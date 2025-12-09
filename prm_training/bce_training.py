import os, json, math, random
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModel,
    Trainer, TrainingArguments, PreTrainedModel
)
import wandb
from peft import LoraConfig, get_peft_model, PeftModel
from pp_dataset import PRMDataset, PRMPackCollator
from prm_model import ProcessRewardModel
from transformers import set_seed as _hf_set_seed
from normalization import (fit_norm_params, apply_normalization, global_info, NormParams, inject_hard_label, filter_entries_by_mi_steps)

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
# os.environ["CUDA_VISIBLE_DEVICES"]= "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["WANDB_PROJECT"]="mi_prm"
os.environ["WANDB_WATCH"]="false"

# ------------------------------------------------------------------------------------
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if _hf_set_seed is not None:
        _hf_set_seed(seed)

def assert_optimizer_has_reward_head(wrapper: ProcessRewardModel, optimizer):
    head_params = {id(p) for p in wrapper.reward_head.parameters()}
    in_opt = set()
    for g in optimizer.param_groups:
        for p in g["params"]:
            in_opt.add(id(p))
    missing = [n for n,p in wrapper.reward_head.named_parameters() if id(p) not in in_opt]
    if missing:
        raise RuntimeError(f"[OPT-ERROR] reward_head params NOT in optimizer: {missing}")
    print("[OPT-OK] reward_head is in optimizer param groups.")

def print_trainable_summary(module: nn.Module):
    total = 0; head = 0
    for n,p in module.named_parameters():
        if p.requires_grad:
            total += p.numel()
            if n.startswith("reward_head."):
                head += p.numel()
    pct = 100.0 * head / max(total,1)
    print(f"[TRAINABLE] total={total:,} | reward_head={head:,} ({pct:.2f}%)")

def _sum_from_mask_prmds(ds_subset, base_ds):
    pos_sum, neg_sum = 0.0, 0.0
    for i in (ds_subset.indices if hasattr(ds_subset, "indices") else range(len(ds_subset))):
        inc = np.asarray(base_ds.samples[i]["is_incorrect"], dtype=float)  # 1=incorrect, 0=correct
        pos_sum += float((1.0 - inc).sum())   # correct 수
        neg_sum += float(inc.sum())           # incorrect 수
    return max(pos_sum, 1e-6), max(neg_sum, 1e-6)

def estimate_pos_weight_and_bias(train_ds, full_ds, *, clamp: float | None = None):
    pos_sum, neg_sum = _sum_from_mask_prmds(train_ds, full_ds)    # pos_sum = Σ y_i, neg_sum = Σ (1 - y_i)
    p = float(pos_sum / max(pos_sum + neg_sum, 1e-6))
    w_neg = 1.0
    w_pos = float(neg_sum / max(pos_sum, 1e-6))
    if clamp is not None:
        w_pos = float(min(w_pos, clamp))
    # bias initialization
    q = (w_pos * p) / (w_pos * p + w_neg * (1.0 - p))
    q = float(min(max(q, 1e-6), 1.0 - 1e-6))
    init_bias = math.log(q / (1.0 - q))
    return w_pos, p, q, init_bias

def _num_steps_in_entry(e: dict) -> int:
    if isinstance(e.get("completion"), list):
        return len(e["completion"])
    if isinstance(e.get("steps"), list):
        return len(e["steps"])
    return 0

def _total_steps(entries: list[dict]) -> int:
    return sum(_num_steps_in_entry(e) for e in entries)

def _ensure_raw_view(entries, reward_key):
    for e in entries:
        if reward_key not in e and f"raw_{reward_key}" in e:
            e[reward_key] = e[f"raw_{reward_key}"]

# ------------------------------------------------------------------------------------
class PRMTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        scores_flat, lengths = model.scores_at_rw(input_ids, attention_mask, inputs["rw_positions"]) # scores_flat: (N_rw,)
        if scores_flat.numel() == 0:
            loss = torch.zeros([], device=input_ids.device, requires_grad=True)
            return (loss, {"preds": [], "tgts": []}) if return_outputs else loss
        tgts = torch.cat(inputs["targets_list"], dim=0).to(scores_flat.device).float()  # (N_rw,)
        probs = torch.sigmoid(scores_flat)

        use_pos_weight = getattr(self.args, "use_pos_weight", True)
        if use_pos_weight:
            # pos_weight (soft-class label) for imbalanced dataset
            w_pos = torch.tensor(getattr(self.args, "pos_weight_global", 1.0), device=scores_flat.device)
            w_neg = torch.tensor(1.0, device=scores_flat.device)
            weight = tgts * w_pos + (1.0 - tgts) * w_neg
            base = F.binary_cross_entropy_with_logits(scores_flat, tgts, reduction="none")
            loss = (base * weight).mean()
            out = {"preds": probs.detach(), "tgts": tgts.detach(), "w_pos": float(torch.as_tensor(w_pos)), "w_neg": float(torch.as_tensor(w_neg))}
        else:
            base = F.binary_cross_entropy_with_logits(scores_flat, tgts, reduction="none")
            loss = base.mean()
            out = {"preds": probs.detach(), "tgts": tgts.detach()}
        
        if hasattr(self, "state") and (self.state.global_step % max(1, self.args.logging_steps) == 0):
            with torch.no_grad():
                self.log({
                    "train/pos_weight_global": float(w_pos) if use_pos_weight else 1.0,
                    "train/targets_mean": float(tgts.mean()),
                    "train/preds_mean": float(probs.mean()),
                    "train/preds_std": float(probs.std()),
                })
        
        return (loss, out) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool = None, ignore_keys=None,):
        """Always compute loss using our custom compute_loss (no labels key needed)."""
        model.eval()
        with torch.no_grad():
            loss, out = self.compute_loss(model, inputs, return_outputs=True)
        loss = loss.detach()
        return (loss, None, None)
    
    def save_model(self, output_dir: str = None, _internal_call: bool = False):
        # Trainer는 _save() 내부에서 save_model()을 호출
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        lr_backbone = self.args.learning_rate              # LoRA용
        lr_head     = getattr(self.args, "learning_rate_head", lr_backbone)
        lr_embed    = getattr(self.args, "learning_rate_embed", lr_backbone * 0.1)

        no_decay_keys = ("bias", "layer_norm.weight", "layernorm.weight", "ln_", "norm.weight")
        head_params, embed_params, lora_params, bk_decay, bk_nodecay = [], [], [], [], []
        emb_param_ids = {id(p) for p in self.model.backbone.get_input_embeddings().parameters()}

        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            name_l = n.lower()
            if n.startswith("reward_head."):
                head_params.append(p); continue
            if id(p) in emb_param_ids:
                embed_params.append(p); continue
            if "lora_" in name_l:
                lora_params.append(p); continue  # ➜ LoRA는 decay 0 권장
            if any(k in name_l for k in no_decay_keys):
                bk_nodecay.append(p)
            else:
                bk_decay.append(p)

        self.optimizer = torch.optim.AdamW([
            {"params": bk_decay,     "lr": lr_backbone, "weight_decay": self.args.weight_decay},
            {"params": bk_nodecay,   "lr": lr_backbone, "weight_decay": 0.0},
            {"params": lora_params,  "lr": lr_backbone, "weight_decay": 0.0},
            {"params": head_params,  "lr": lr_head,     "weight_decay": 0.0},
            {"params": embed_params, "lr": lr_embed,    "weight_decay": 0.0},
        ], betas=(0.9, 0.999), eps=1e-8)
        
        return self.optimizer

# ------------------------------------------------------------------------------------
@dataclass
class TrainConfig:
    model_name: str =  "Qwen/Qwen3-4B-base" # "Qwen/Qwen2.5-Math-1.5B-Instruct", "meta-llama/Meta-Llama-3.1-8B", "mistralai/Mistral-7B-v0.3", Qwen/Qwen3-4B-base
    rw_token: str = "<RW>"
    add_rw_token: bool = True
    max_length: int = 2400
    # trainarguments
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 3
    gradient_checkpointing: bool = True
    num_train_epochs: int = 2
    learning_rate: float = 2e-5
    learning_rate_head: float = 3e-4      # reward_head 크게
    learning_rate_embed: float = 2e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.08
    logging_steps: int = 100
    save_steps: int = 200
    eval_steps: int = 50
    bf16: bool = True
    seed: int = 42
    # loss
    norm_type: str = "logquant" # (logquant, zsig, exp_cdf)
    # data type
    reward_key: Optional[str] = "base_reward" # ("mi_loo","mi_shapley","mi_margin","mi_cmi","hard_label", "base_reward", "gold_label")
    base_type: Optional[str] = "qval"  # ("qval", "pav")
    val_ratio: float = 0.15
    mi_abs_max: float = 50
    # lora
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # imbalance dataset
    use_bias_init: bool = False
    use_pos_weight: bool = True

# ------------------------------------------------------------------------------------
def main(cfg: TrainConfig):
    set_all_seeds(cfg.seed)
    if cfg.reward_key.startswith("base_") and cfg.base_type is not None:
        output_dir = f"/home/leena/rs_prm/checkpoints/qw3_4b/{cfg.base_type}"
    else:
        output_dir = f"/home/leena/rs_prm/checkpoints/qw3_4b/{cfg.reward_key}/{cfg.norm_type}"
    os.makedirs(output_dir, exist_ok=True)

    # 1) Tokenizer & Model
    tok = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        device_map="auto",
    )

    if cfg.gradient_checkpointing:
        base.gradient_checkpointing_enable()
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()
        if hasattr(base.config, "use_cache"):
            base.config.use_cache = False

    if cfg.add_rw_token and cfg.rw_token not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": [cfg.rw_token]})
        base.resize_token_embeddings(len(tok))
    
    if cfg.use_lora:
        lconf = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",)
        base = get_peft_model(base, lconf)
        base.print_trainable_parameters()

    rw_id = tok.convert_tokens_to_ids(cfg.rw_token) if cfg.add_rw_token else None
    model = ProcessRewardModel(base, rw_token_id=rw_id, rw_token=cfg.rw_token)
    emb_module = model.backbone.get_input_embeddings()   # nn.Embedding
    emb_module.weight.requires_grad_(True)
    print("Model Strucutre:", model, flush=True)
    print_trainable_summary(model)

    # 2) Dataset Loading & preprocessing (normalization)
    if cfg.reward_key.startswith("mi_") and cfg.base_type is not None:
        with open("/home/leena/rs_prm/samples/mi/gsm8k_mi_qw3_4b.json", "r", encoding="utf-8") as f1:
            gsm8k_entries = json.load(f1)
        with open("/home/leena/rs_prm/samples/mi/math_mi_qw3_4b.json", "r", encoding="utf-8") as f2:
            math_entries = json.load(f2)
        with open("/home/leena/rs_prm/samples/mi/omni_mi_qw3_4b.json", "r", encoding="utf-8") as f3:
            omni_entries = json.load(f3)
    else:
        with open(f"/home/leena/rs_prm/samples/baselines/{cfg.base_type}/gsm8k_{cfg.base_type}_qw3_4b.json", "r", encoding="utf-8") as f1:
            gsm8k_entries = json.load(f1)
        with open(f"/home/leena/rs_prm/samples/baselines/{cfg.base_type}/math_{cfg.base_type}_qw3_4b.json", "r", encoding="utf-8") as f2:
            math_entries = json.load(f2)
        with open(f"/home/leena/rs_prm/samples/baselines/{cfg.base_type}/omni_{cfg.base_type}_qw3_4b.json", "r", encoding="utf-8") as f3:
            omni_entries = json.load(f3)
    
    # 3) Filtering (RAW) → Normalize (Filtered)
    norm_type = (cfg.norm_type).lower()
    if cfg.reward_key.startswith("mi_"):
        _ensure_raw_view(gsm8k_entries, cfg.reward_key)
        _ensure_raw_view(math_entries,  cfg.reward_key)
        _ensure_raw_view(omni_entries,  cfg.reward_key)
        b_gs = _total_steps(gsm8k_entries)
        b_ma = _total_steps(math_entries)
        b_om = _total_steps(omni_entries)
        gsm8k_f = filter_entries_by_mi_steps(gsm8k_entries, reward_key=cfg.reward_key, abs_max=getattr(cfg, "mi_abs_max", None),)
        math_f = filter_entries_by_mi_steps(math_entries, reward_key=cfg.reward_key, abs_max=getattr(cfg, "mi_abs_max", None),)
        omni_f = filter_entries_by_mi_steps(omni_entries, reward_key=cfg.reward_key, abs_max=getattr(cfg, "mi_abs_max", None),)
        a_gs = _total_steps(gsm8k_f); a_ma = _total_steps(math_f); a_om = _total_steps(omni_f)
        print(f"[INFO] RAW filter GSM8K: {b_gs} -> {a_gs} | MATH: {b_ma} -> {a_ma} | OMNI: {b_om} -> {a_om}")
        
        entries_all = gsm8k_f + math_f + omni_f
        if len(entries_all) == 0 or _total_steps(entries_all) == 0:
            raise RuntimeError("No data left after RAW filtering. Relax thresholds or check raw keys.")

        # normalization
        norm_params_gs = fit_norm_params(gsm8k_f, reward_key=cfg.reward_key, norm_type=norm_type, q_low=0.01, q_high=0.99, eps=0.01, temp=1.0)
        norm_params_ma = fit_norm_params(math_f,  reward_key=cfg.reward_key, norm_type=norm_type, q_low=0.01, q_high=0.99, eps=0.01, temp=1.0)
        norm_params_om = fit_norm_params(omni_f,  reward_key=cfg.reward_key, norm_type=norm_type, q_low=0.01, q_high=0.99, eps=0.01, temp=1.0)
        # norm_params: NormParams = fit_norm_params(entries_all, reward_key=cfg.reward_key, norm_type=norm_type, q_low=0.01, q_high=0.99, eps=0.01, temp=1.0)
        norm_gs = apply_normalization(gsm8k_f, reward_key=cfg.reward_key, params=norm_params_gs, keep_raw=True)
        norm_ma = apply_normalization(math_f,  reward_key=cfg.reward_key, params=norm_params_ma, keep_raw=True)
        norm_om = apply_normalization(omni_f,  reward_key=cfg.reward_key, params=norm_params_om, keep_raw=True)
        print(f"[NORM] type={norm_type}")
        entries = norm_gs + norm_ma + norm_om

    else: # hard_label 등
        gsm8k_f, math_f, omni_f = gsm8k_entries, math_entries, omni_entries
        entries = gsm8k_f + math_f + omni_f
    
    # gain global stats
    gs_m, gs_std, gs_qan = global_info(gsm8k_f, reward_key=cfg.reward_key)
    ma_m, ma_std, ma_qan = global_info(math_f, reward_key=cfg.reward_key)
    om_m, om_std, om_qan = global_info(omni_f, reward_key=cfg.reward_key)
    print(f"[STATS] GSM8K mean/std: {gs_m:.4f}/{gs_std:.4f}")
    print(f"[STATS] MATH  mean/std: {ma_m:.4f}/{ma_std:.4f}")
    print(f"[STATS] OMNI  mean/std: {om_m:.4f}/{om_std:.4f}")
    
    # 4) Training stability 
    full_ds  = PRMDataset(entries, tok, reward_key=cfg.reward_key, add_rw_token=cfg.add_rw_token, rw_token=cfg.rw_token, max_length=cfg.max_length)
    n = len(full_ds)
    n_val = max(1, int(n * cfg.val_ratio))
    n_train = max(1, n - n_val)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed))
    data_collator = PRMPackCollator(pad_token_id=tok.pad_token_id, rw_token_id=tok.convert_tokens_to_ids(cfg.rw_token), strict=False)
    print(f"Load model and {cfg.reward_key} type dataset!", flush=True)

    # pos_weight (soft label) & bias init
    w_pos, p, q_eff, init_bias = estimate_pos_weight_and_bias(train_ds, full_ds)
    print(f"[BAL] w_pos={w_pos:.3f} (w_neg=1.0) | p={p:.4f} | q_eff={q_eff:.4f}")

    if cfg.use_bias_init:
        final_linear = None
        if isinstance(model.reward_head, nn.Linear):
            final_linear = model.reward_head
        elif isinstance(model.reward_head, nn.Sequential) and isinstance(model.reward_head[-1], nn.Linear):
            final_linear = model.reward_head[-1]
        if final_linear is not None and final_linear.bias is not None:
            ib = torch.tensor(init_bias, device=final_linear.bias.device, dtype=final_linear.bias.dtype)
            final_linear.bias.data.copy_(ib)
            print(f"[INIT] reward_head.bias <- {init_bias:.4f}")

    # 5) TrainingArguments
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        gradient_checkpointing=cfg.gradient_checkpointing,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        num_train_epochs=cfg.num_train_epochs,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=1,
        eval_strategy ="steps",
        eval_steps=cfg.eval_steps, # if val_ds is not None else None,
        bf16=cfg.bf16,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        report_to=["wandb"],
        run_name = f"qw3_4b_{cfg.reward_key}_{cfg.base_type}_{cfg.norm_type}",
    )
    setattr(args, "learning_rate_head", cfg.learning_rate_head)
    setattr(args, "learning_rate_embed", cfg.learning_rate_embed)
    setattr(args, "pos_weight_global", float(w_pos))
    setattr(args, "use_pos_weight", cfg.use_pos_weight)

    # 6) Trainer
    trainer = PRMTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    # check if the model has a reward head and optimizer is set up correctly
    _ = trainer.create_optimizer()
    assert_optimizer_has_reward_head(trainer.model, trainer.optimizer)
    print_trainable_summary(trainer.model)

    # 7) Train
    print(f"Start Training with {cfg.norm_type} loss!", flush=True)
    torch.cuda.empty_cache()
    trainer.train()
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)

    # 8) Save final
    save_dir = os.path.join(output_dir, "final_model")
    model.save_pretrained(save_dir) 
    tok.save_pretrained(save_dir)
    if cfg.use_lora:
        print("Saving LoRA adapter weights...")
        # Save LoRA adapter weights
        adapter_dir = os.path.join(save_dir, "adapter")
        model.backbone.save_pretrained(adapter_dir, safe_serialization=True)
    wandb.finish()


if __name__ == "__main__":
    cfg = TrainConfig()
    main(cfg)

