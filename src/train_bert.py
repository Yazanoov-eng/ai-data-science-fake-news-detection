import argparse
import os
import random
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset

from .config import TrainConfig
from .evaluate import compute_metrics, save_json

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def tokenize_fn(tokenizer, max_length):
    def _fn(examples):
        return tokenizer(examples["content"], truncation=True, padding="max_length", max_length=max_length)
    return _fn

def hf_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return compute_metrics(labels, preds)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, choices=["isot","kaggle"], required=True)
    ap.add_argument("--data_dir", type=str, default="outputs")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    cfg = TrainConfig()
    cfg.epochs = args.epochs
    cfg.max_length = args.max_length
    cfg.batch_size = args.batch_size
    cfg.lr = args.lr

    set_seed(cfg.seed)

    train_csv = os.path.join(args.data_dir, f"{args.dataset}_train.csv")
    val_csv   = os.path.join(args.data_dir, f"{args.dataset}_val.csv")
    test_csv  = os.path.join(args.data_dir, f"{args.dataset}_test.csv")

    ds = load_dataset("csv", data_files={"train": train_csv, "validation": val_csv, "test": test_csv})

    tokenizer = AutoTokenizer.from_pretrained(cfg.bert_model_name, use_fast=True)
    ds_tok = ds.map(tokenize_fn(tokenizer, cfg.max_length), batched=True)

    ds_tok = ds_tok.rename_column("label", "labels")
    ds_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(cfg.bert_model_name, num_labels=2)

    out_dir = os.path.join(args.data_dir, f"bert_{args.dataset}_ckpt")
    training_args = TrainingArguments(
        output_dir=out_dir,
        #evaluation_strategy="epoch",
        #save_strategy="epoch",
        #load_best_model_at_end=True,
        #metric_for_best_model="f1",
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        logging_steps=50,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        tokenizer=tokenizer,
        compute_metrics=hf_metrics
    )

    trainer.train()

    test_metrics = trainer.evaluate(ds_tok["test"])
    cleaned = {}
    for k, v in test_metrics.items():
        if k.startswith("eval_") and isinstance(v, (int, float)):
            cleaned[k.replace("eval_", "")] = float(v)

    out = {
        "model": "Fine-tuned BERT classifier",
        "dataset": args.dataset,
        "test": cleaned
    }
    out_path = os.path.join(args.data_dir, f"results_bert_{args.dataset}.json")
    save_json(out, out_path)
    print(f"[OK] Wrote results -> {out_path}")

if __name__ == "__main__":
    main()
