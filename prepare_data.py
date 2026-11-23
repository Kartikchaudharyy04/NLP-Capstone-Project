"""Stages 2-5: dataset download, cleaning, preprocessing, EDA.
Run: python prepare_data.py"""
import io, os, re, zipfile, unicodedata, urllib.request
from collections import Counter
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

URL = "http://www.manythings.org/anki/fra-eng.zip"   # Tatoeba / Anki pairs
MAX_WORDS, SAMPLE = 12, 40000

def normalize(s: str) -> str:
    """Unicode-normalise, lowercase, split punctuation, drop odd symbols."""
    s = unicodedata.normalize("NFC", str(s)).lower().strip()
    s = re.sub(r"([?.!,¿])", r" \1 ", s)
    s = re.sub(r"[^a-zà-ÿœæç?.!,'¿ ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def main():
    os.makedirs("data", exist_ok=True); os.makedirs("plots", exist_ok=True)
    if not os.path.exists("data/fra.txt"):
        print("Downloading dataset...")
        z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read()))
        z.extract("fra.txt", "data")
    df = pd.read_csv("data/fra.txt", sep="\t", header=None,
                     names=["eng", "fra", "meta"], usecols=[0, 1])
    print("Raw pairs:", len(df))
    print(df.head(), "\nMissing:\n", df.isna().sum())

    # ---- Stage 3: cleaning
    df = df.dropna().drop_duplicates()
    df["eng"], df["fra"] = df.eng.map(normalize), df.fra.map(normalize)
    df = df[(df.eng != "") & (df.fra != "")].drop_duplicates()
    # ---- Stage 4: preprocessing (tokenisation = whitespace after normalisation;
    # stop-words are KEPT on purpose: removing them would break translation)
    df["n_en"], df["n_fr"] = df.eng.str.split().str.len(), df.fra.str.split().str.len()
    df = df[(df.n_en <= MAX_WORDS) & (df.n_fr <= MAX_WORDS)]
    df = df.sample(min(SAMPLE, len(df)), random_state=42)
    df[["eng", "fra"]].to_csv("data/clean.csv", index=False)
    print("Clean pairs:", len(df))

    # ---- Stage 5: EDA
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].hist(df.n_en, bins=range(1, MAX_WORDS + 2), color="#4c72b0"); ax[0].set_title("English length")
    ax[1].hist(df.n_fr, bins=range(1, MAX_WORDS + 2), color="#dd8452"); ax[1].set_title("French length")
    plt.tight_layout(); plt.savefig("plots/length_dist.png", dpi=120); plt.close()
    for col, name in [("eng", "English"), ("fra", "French")]:
        c = Counter(w for t in df[col] for w in t.split())
        print(f"{name} vocab: {len(c)}")
        w, n = zip(*c.most_common(15))
        plt.figure(figsize=(7, 3.5)); plt.bar(w, n); plt.xticks(rotation=60)
        plt.title(f"Top words - {name}"); plt.tight_layout()
        plt.savefig(f"plots/top_{name.lower()}.png", dpi=120); plt.close()
    print("Saved data/clean.csv and plots/")

if __name__ == "__main__":
    main()
