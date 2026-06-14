import re
import nltk
import numpy as np
from collections import Counter

from sklearn.preprocessing import LabelEncoder

from nltk.tokenize import word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# ==========================
# TEXT CLEANING
# ==========================

def clean_text(text):

    if text is None:
        return ""

    text = str(text)

    text = text.lower()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ==========================
# TOKENIZATION
# ==========================

def tokenize(text):

    if not text:
        return []

    return word_tokenize(text)


# ==========================
# BUILD VOCAB
# ==========================

def build_vocab(
    texts,
    max_vocab_size
):

    counter = Counter()

    for text in texts:

        tokens = tokenize(text)

        counter.update(tokens)

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for word, _ in counter.most_common(
        max_vocab_size - 2
    ):

        vocab[word] = len(vocab)

    return vocab


# ==========================
# ENCODE TEXT
# ==========================

def encode_text(
    text,
    vocab,
    max_len
):

    tokens = tokenize(text)

    encoded = [
        vocab.get(
            token,
            vocab["<UNK>"]
        )
        for token in tokens
    ]

    encoded = encoded[:max_len]

    if len(encoded) < max_len:

        encoded += [0] * (
            max_len - len(encoded)
        )

    return encoded


# ==========================
# PREPROCESS
# ==========================

def preprocess(
    data,
    max_vocab_size,
    max_seq_len
):

    texts = []
    labels = []

    for row in data:

        subject = clean_text(
            row.get("subject")
        )

        body = clean_text(
            row.get("body")
        )

        text = (
            subject + " " + body
        ).strip()

        queue = row.get("queue")

        if text and queue:

            texts.append(text)

            labels.append(queue)

    print(
        f"Valid Samples: {len(texts)}"
    )

    vocab = build_vocab(
        texts,
        max_vocab_size
    )

    X = np.array(
        [
            encode_text(
                text,
                vocab,
                max_seq_len
            )
            for text in texts
        ]
    )

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(
        labels
    )

    return (
        X,
        y,
        vocab,
        label_encoder
    )