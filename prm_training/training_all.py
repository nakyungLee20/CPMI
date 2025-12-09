import os, json, math, random
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments)
import wandb
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import set_seed as _hf_set_seed

# project class
from pp_dataset import PRMDataset, PRMPackCollator
from prm_model import ProcessRewardModel
from normalization import (global_info, filter_entries_by_mi_steps)

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
        corr = np.asarray(base_ds.samples[i]["is_correct"], dtype=float)  # 1=correct, 0=incorrect
        pos_sum += float(corr.sum())           # correct 수 (y=1)
        neg_sum += float((1.0 - corr).sum())   # incorrect 수 (y=0)
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

        loss_type = getattr(self.args, "loss_type", "bcelogits").lower()
        use_pos_weight = getattr(self.args, "use_pos_weight", True)
        if use_pos_weight:
            w_pos = torch.tensor(getattr(self.args, "pos_weight_global", 1.0), device=scores_flat.device)
            w_neg = torch.tensor(1.0, device=scores_flat.device)
            weight = tgts * w_pos + (1.0 - tgts) * w_neg  # 샘플 가중치
        else:
            weight = None

        if loss_type in ("bcelogits", "bce"):
            base = F.binary_cross_entropy_with_logits(scores_flat, tgts, reduction="none")
            loss = (base * weight).mean() if weight is not None else base.mean()
            w_neg_val = float(torch.as_tensor(w_neg)) if use_pos_weight else 1.0
            w_pos_val = float(torch.as_tensor(w_pos)) if use_pos_weight else 1.0
            out = {"preds": probs.detach(), "tgts": tgts.detach(), "w_pos": w_pos_val, "w_neg": w_neg_val}
        elif loss_type == "mse": # in probability space
            base = F.mse_loss(probs, tgts, reduction="none") 
            loss = (base * weight).mean() if weight is not None else base.mean()
            out = {"preds": probs.detach(), "tgts": tgts.detach()}
        elif loss_type == "mse_logit": # in logit space
            base = F.mse_loss(scores_flat, tgts, reduction="none")
            loss = (base * weight).mean() if weight is not None else base.mean()
            out = {"preds": scores_flat.detach(), "tgts": tgts.detach()}
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
        if hasattr(self, "state") and (self.state.global_step % max(1, self.args.logging_steps) == 0):
            with torch.no_grad():
                self.log({ 
                    # "train/loss": float(loss.detach().item()), 
                    "train/targets_mean": float(tgts.mean()),
                    "train/preds_mean": float(probs.mean()),
                    "train/preds_std": float(probs.std()),
                    "train/pos_weight_global": float(w_pos) if use_pos_weight else 1.0,
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
    max_length: int = 2500
    # trainarguments
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 3
    gradient_checkpointing: bool = True
    num_train_epochs: int = 2
    learning_rate: float = 2e-5
    learning_rate_head: float = 3e-4
    learning_rate_embed: float = 2e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    logging_steps: int = 200
    save_steps: int = 400
    eval_steps: int = 100
    bf16: bool = True
    seed: int = 42
    # loss
    norm_type: str = "logquant"             # (logquant, zsig)
    loss_type: str = "bcelogits"                  # bcelogits | mse | mse_logit
    mse_nonlinearity: str = "sigmoid"
    # data type
    reward_key: Optional[str] = "step_reward"    # ("mi_loo","mi_shapley","mi_margin","mi_cmi", "step_reward", "base_reward", "gold_label")
    base_type: Optional[str] = ""       # ("qval", "pav", "merge_*")
    val_ratio: float = 0.15
    mi_abs_max: float = 60
    # lora
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # imbalance dataset
    use_bias_init: bool = False
    use_pos_weight: bool = True
    dataset_json: Optional[str] = None      # path to a specific dataset JSON
    exp_name: Optional[str] = None 

def _patch_cfg_from_cli(cfg: TrainConfig) -> TrainConfig:
    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--dataset_json", type=str, default=None)
    ap.add_argument("--exp_name", type=str, default=None)
    args, _ = ap.parse_known_args()
    if args.dataset_json:  cfg.dataset_json = args.dataset_json
    if args.exp_name:      cfg.exp_name = args.exp_name
    return cfg

# ------------------------------------------------------------------------------------
def main(cfg: TrainConfig):
    set_all_seeds(cfg.seed)
    suffix = {"bcelogits":"bce","mse":"mse","mse_logit":"mlogit"}.get(cfg.loss_type, cfg.loss_type)
    if cfg.reward_key.startswith("base_") and cfg.base_type is not None:
        output_dir = f"/home/leena/rs_prm/checkpoints/hard_80k/qw3_4b/{cfg.base_type}_{suffix}"
    elif cfg.reward_key.startswith("step_"):
        output_dir = f"/home/leena/rs_prm/checkpoints/hard_80k/qw3_4b/{cfg.exp_name}_{suffix}"
    else:
        output_dir = f"/home/leena/rs_prm/checkpoints/hard_80k/qw3_4b/{cfg.reward_key}_{suffix}"
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
        with open("/home/leena/rs_prm/datasets/hard_80k/mi_sum2/ms_mi_qw3_4b.json", "r", encoding="utf-8") as f1:
            raw_entries = json.load(f1)
    elif cfg.reward_key == "gold_label":
        with open("/home/leena/rs_prm/datasets/hard_80k/mi_sum2/ms_mi_qw3_4b.json", "r", encoding="utf-8") as f2:
            raw_entries = json.load(f2)
    elif cfg.reward_key == "step_reward":
        dataset_path = cfg.dataset_json
        with open(dataset_path, "r", encoding="utf-8") as f3:
            raw_entries = json.load(f3)
        print(f"Loaded from {dataset_path}.")
    elif cfg.reward_key.startswith("pll_"):
        with open("/home/leena/rs_prm/datasets/hard_80k/cpmi_prompt/ms_cpmi_prompt_qw3_4b.json", "r", encoding="utf-8") as f4:
            raw_entries = json.load(f4)
    else:
        with open(f"/home/leena/rs_prm/datasets/hard_80k/{cfg.base_type}/ms_{cfg.base_type}_qw3_4b.json", "r", encoding="utf-8") as f5:
            raw_entries = json.load(f5)
    
    # 3) Filtering (RAW) → Normalize (Filtered)
    norm_type = (cfg.norm_type).lower()
    if cfg.reward_key.startswith("mi_"):
        _ensure_raw_view(raw_entries, cfg.reward_key)
        b_ms = _total_steps(raw_entries)
        ms_f = filter_entries_by_mi_steps(raw_entries, reward_key=cfg.reward_key, abs_max=getattr(cfg, "mi_abs_max", None),)
        a_ms = _total_steps(ms_f)
        print(f"[INFO] RAW filter MathShepherd: {b_ms} -> {a_ms}")
        if len(ms_f) == 0 or _total_steps(ms_f) == 0:
            raise RuntimeError("No data left after RAW filtering. Relax thresholds or check raw keys.")
        # normalization
        if cfg.loss_type in ("mse"):
            from normalization import fit_mse_mi_params, apply_mse_mi_normalization
            mse_params = fit_mse_mi_params(ms_f, reward_key=cfg.reward_key, q_low=0.01, q_high=0.99, eps=1e-6, temp=1.0)
            entries = apply_mse_mi_normalization(ms_f, reward_key=cfg.reward_key, params=mse_params, keep_raw=True, nonlinearity=cfg.mse_nonlinearity)
            print(f"[NORM-MSE] robust-zscore + {cfg.mse_nonlinearity} → [0,1]")
        elif cfg.loss_type == "mse_logit":
            from normalization import fit_mse_logit_params, apply_mse_logit_standardize
            logit_params = fit_mse_logit_params(ms_f, reward_key=cfg.reward_key, q_low=0.01, q_high=0.99, eps=1e-6, temp=1.0)
            entries = apply_mse_logit_standardize(ms_f, reward_key=cfg.reward_key, params=logit_params, keep_raw=True)
            print("[NORM-LOGIT] robust-zscore only (no tanh/sigmoid)")
        else:
            from normalization import fit_norm_params, apply_normalization
            norm_params_ms = fit_norm_params(ms_f, reward_key=cfg.reward_key, norm_type=norm_type, q_low=0.01, q_high=0.99, eps=0.01, temp=1.0)
            entries = apply_normalization(ms_f, reward_key=cfg.reward_key, params=norm_params_ms, keep_raw=True)
            print(f"[NORM] type={norm_type}")
        # gain global stats
        ms_m, ms_std, ms_qan = global_info(ms_f, reward_key=cfg.reward_key)
        print(f"[STATS] MS mean/std: {ms_m:.4f}/{ms_std:.4f}")
    else: # gold_label, base_type 등
        print("No normalization!")
        entries= raw_entries
    
    # 4) Training stability 
    full_ds  = PRMDataset(entries, tok, reward_key=cfg.reward_key, add_rw_token=cfg.add_rw_token, rw_token=cfg.rw_token, max_length=cfg.max_length)
    n = len(full_ds)
    n_val = max(1, int(n * cfg.val_ratio))
    n_train = max(1, n - n_val)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.seed))
    data_collator = PRMPackCollator(pad_token_id=tok.pad_token_id, rw_token_id=tok.convert_tokens_to_ids(cfg.rw_token), strict=False)
    print(f"Load model and {cfg.reward_key} {cfg.exp_name} type dataset!", flush=True)

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
        run_name = f"ms_qw3_4b_{suffix}_{cfg.exp_name}",
    )
    setattr(args, "learning_rate_head", cfg.learning_rate_head)
    setattr(args, "learning_rate_embed", cfg.learning_rate_embed)
    setattr(args, "pos_weight_global", float(w_pos))
    
    # pos_weight (soft label) & bias init
    if cfg.loss_type in ("mse", "mse_logit"):
        setattr(args, "use_pos_weight", False)
    else:
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
    print(f"Start Training with {cfg.loss_type} loss!", flush=True)
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
    cfg = _patch_cfg_from_cli(cfg)
    main(cfg)

