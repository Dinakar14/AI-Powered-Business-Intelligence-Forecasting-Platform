import torch
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from sklearn.model_selection import train_test_split

from data_loader import load_data
from preprocessing import preprocess

from model import TicketLSTM

from config import *

# ==========================
# LOAD DATA
# ==========================

data = load_data()

X, y, vocab, label_encoder = preprocess(
    data,
    MAX_VOCAB_SIZE,
    MAX_SEQ_LEN
)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# LOAD MODEL
# ==========================

model = TicketLSTM(
    len(vocab),
    EMBEDDING_DIM,
    HIDDEN_DIM,
    len(label_encoder.classes_)
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()

# ==========================
# PREDICTION
# ==========================

with torch.no_grad():

    preds = model(
        torch.LongTensor(X_val)
    )

    preds = preds.argmax(
        dim=1
    ).numpy()

# ==========================
# METRICS
# ==========================

print("\nClassification Report\n")

print(
    classification_report(
        y_val,
        preds,
        target_names=label_encoder.classes_
    )
)

print("\nConfusion Matrix\n")

print(
    confusion_matrix(
        y_val,
        preds
    )
)