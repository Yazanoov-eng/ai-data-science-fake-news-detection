import argparse
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_isot(true_csv: str, fake_csv: str) -> pd.DataFrame:
    true_df = pd.read_csv(true_csv)
    fake_df = pd.read_csv(fake_csv)

    true_df["label"] = 0  # real
    fake_df["label"] = 1  # fake

    df = pd.concat([true_df, fake_df], ignore_index=True)

    df["title"] = df["title"].fillna("").map(clean_text)
    df["text"]  = df["text"].fillna("").map(clean_text)
    df["content"] = (df["title"] + "\n" + df["text"]).str.strip()
    df = df[df["content"].str.len() > 0]

    return df[["content", "label"]].sample(frac=1, random_state=42).reset_index(drop=True)

def build_kaggle(kaggle_csv: str) -> pd.DataFrame:
    df = pd.read_csv(kaggle_csv)

    # Drop index-like columns if present
    for col in ["Unnamed: 0", "id"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    df["title"] = df.get("title", "").fillna("").map(clean_text)
    df["text"]  = df.get("text", "").fillna("").map(clean_text)

    if "author" in df.columns:
        df["author"] = df["author"].fillna("").map(clean_text)
        df["content"] = (df["title"] + "\n" + df["author"] + "\n" + df["text"]).str.strip()
    else:
        df["content"] = (df["title"] + "\n" + df["text"]).str.strip()

    df = df[df["content"].str.len() > 0]
    df["label"] = df["label"].astype(int)  # usually 0=real, 1=fake

    return df[["content", "label"]].sample(frac=1, random_state=42).reset_index(drop=True)

def split_save(df: pd.DataFrame, prefix: str, out_dir: str, val_size: float = 0.15, test_size: float = 0.15):
    os.makedirs(out_dir, exist_ok=True)

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42, stratify=df["label"]
    )

    val_rel = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_df, test_size=val_rel, random_state=42, stratify=train_df["label"]
    )

    train_df.to_csv(os.path.join(out_dir, f"{prefix}_train.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, f"{prefix}_val.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, f"{prefix}_test.csv"), index=False)

    print(f"[OK] {prefix} splits: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isot_true", type=str, required=True)
    ap.add_argument("--isot_fake", type=str, required=True)
    ap.add_argument("--kaggle", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="outputs")
    args = ap.parse_args()

    isot = build_isot(args.isot_true, args.isot_fake)
    kag  = build_kaggle(args.kaggle)

    split_save(isot, "isot", args.out_dir)
    split_save(kag, "kaggle", args.out_dir)

if __name__ == "__main__":
    main()
