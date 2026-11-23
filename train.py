"""Stages 6-8: feature engineering (tokenizer->integer sequences->learned embeddings),
Seq2Seq LSTM model, training and evaluation.  Run: python train.py"""
import json, os, pickle
from collections import Counter
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

VOCAB, EMB, UNITS, EPOCHS, BATCH = 8000, 256, 256, 15, 128
os.makedirs("model", exist_ok=True)

df = pd.read_csv("data/clean.csv")
src = df.eng.tolist(); tgt = ("<start> " + df.fra + " <end>").tolist()
tr_s, te_s, tr_t, te_t = train_test_split(src, tgt, test_size=0.05, random_state=42)

# Feature engineering: word -> integer ids (embedding layer learns dense vectors)
st = Tokenizer(num_words=VOCAB, oov_token="<unk>", filters="")
tt = Tokenizer(num_words=VOCAB, oov_token="<unk>", filters="")
st.fit_on_texts(tr_s); tt.fit_on_texts(tr_t)
max_s = max(len(s.split()) for s in tr_s); max_t = max(len(t.split()) for t in tr_t)
X = pad_sequences(st.texts_to_sequences(tr_s), maxlen=max_s, padding="post")
Y = pad_sequences(tt.texts_to_sequences(tr_t), maxlen=max_t, padding="post")
dec_in, dec_out = Y[:, :-1], Y[:, 1:]          # teacher forcing

# Model: encoder LSTM -> context (h,c) -> decoder LSTM -> softmax over French vocab
e_in = layers.Input((None,))
e_emb = layers.Embedding(VOCAB, EMB, mask_zero=True, name="enc_emb")(e_in)
_, h, c = layers.LSTM(UNITS, return_state=True, name="enc_lstm")(e_emb)
d_in = layers.Input((None,))
d_emb = layers.Embedding(VOCAB, EMB, mask_zero=True, name="dec_emb")(d_in)
d_seq, _, _ = layers.LSTM(UNITS, return_sequences=True, return_state=True,
                          name="dec_lstm")(d_emb, initial_state=[h, c])
out = layers.Dense(VOCAB, activation="softmax", name="dec_dense")(layers.Dropout(0.2)(d_seq))
model = Model([e_in, d_in], out)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

hist = model.fit([X, dec_in], dec_out, batch_size=BATCH, epochs=EPOCHS, validation_split=0.1,
                 callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
model.save("model/seq2seq.keras")
pickle.dump(st, open("model/src_tok.pkl", "wb")); pickle.dump(tt, open("model/tgt_tok.pkl", "wb"))
json.dump({"units": UNITS, "max_src": max_s, "max_tgt": max_t - 1}, open("model/config.json", "w"))

plt.plot(hist.history["loss"], label="train"); plt.plot(hist.history["val_loss"], label="val")
plt.legend(); plt.title("Loss"); plt.savefig("plots/loss.png", dpi=120)

# ---- Evaluation on held-out sentences
from translator import Translator
T = Translator()
N = 500
preds = [T.translate(s).split() for s in te_s[:N]]
refs = [t.replace("<start> ", "").replace(" <end>", "").split() for t in te_t[:N]]
bleu = corpus_bleu([[r] for r in refs], preds, smoothing_function=SmoothingFunction().method1)
tp = sum(sum((Counter(p) & Counter(r)).values()) for p, r in zip(preds, refs))
prec = tp / max(1, sum(map(len, preds))); rec = tp / max(1, sum(map(len, refs)))
f1 = 2 * prec * rec / max(1e-9, prec + rec)
Xt = pad_sequences(st.texts_to_sequences(te_s), maxlen=max_s, padding="post")
Yt = pad_sequences(tt.texts_to_sequences(te_t), maxlen=max_t, padding="post")
_, acc = model.evaluate([Xt, Yt[:, :-1]], Yt[:, 1:], verbose=0)
metrics = dict(token_accuracy=acc, bleu=bleu, precision=prec, recall=rec, f1=f1)
json.dump(metrics, open("model/metrics.json", "w"), indent=2)
print(metrics)
for s, p, r in list(zip(te_s, preds, refs))[:5]:
    print(f"EN: {s}\nPRED: {' '.join(p)}\nREF : {' '.join(r)}\n")
