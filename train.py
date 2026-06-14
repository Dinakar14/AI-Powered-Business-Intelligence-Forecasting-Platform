import os
import joblib
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score

from torch.utils.data import (
TensorDataset,
DataLoader,
WeightedRandomSampler
)

from data_loader import load_data
from preprocessing import preprocess
from model import TicketLSTM
from config import *

print("EPOCHS =", EPOCHS)

os.makedirs("saved", exist_ok=True)

print("Loading dataset...")

data = load_data()

X, y, vocab, label_encoder = preprocess(
data,
MAX_VOCAB_SIZE,
MAX_SEQ_LEN
)

joblib.dump(vocab, VOCAB_PATH)
joblib.dump(label_encoder, LABEL_ENCODER_PATH)

X_train, X_val, y_train, y_val = train_test_split(
X,
y,
test_size=0.2,
random_state=42,
stratify=y
)

device = torch.device(
"cuda" if torch.cuda.is_available()
else "cpu"
)

weights = compute_class_weight(
class_weight="balanced",
classes=np.unique(y_train),
y=y_train
)

weights = torch.tensor(
weights,
dtype=torch.float
).to(device)

train_dataset = TensorDataset(
torch.LongTensor(X_train),
torch.LongTensor(y_train)
)

val_dataset = TensorDataset(
torch.LongTensor(X_val),
torch.LongTensor(y_val)
)

class_counts = np.bincount(y_train)

class_weights = 1.0 / class_counts

sample_weights = class_weights[y_train]

sample_weights = torch.DoubleTensor(
sample_weights
)

sampler = WeightedRandomSampler(
sample_weights,
len(sample_weights),
replacement=True
)

train_loader = DataLoader(
train_dataset,
batch_size=BATCH_SIZE,
sampler=sampler
)

val_loader = DataLoader(
val_dataset,
batch_size=BATCH_SIZE
)

model = TicketLSTM(
vocab_size=len(vocab),
embed_dim=EMBEDDING_DIM,
hidden_dim=HIDDEN_DIM,
output_dim=len(label_encoder.classes_)
)

model.to(device)

criterion = nn.CrossEntropyLoss(
weight=weights
)

optimizer = optim.Adam(
model.parameters(),
lr=LEARNING_RATE
)

best_acc = 0

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    for xb, yb in train_loader:

        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()

        outputs = model(xb)

        loss = criterion(
            outputs,
            yb
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    model.eval()

    preds_all = []
    actuals_all = []

    top5_correct = 0
    total = 0

    with torch.no_grad():

        for xb, yb in val_loader:

            xb = xb.to(device)

            outputs = model(xb)

            preds = outputs.argmax(
                dim=1
            ).cpu().numpy()

            preds_all.extend(preds)

            actuals_all.extend(
                yb.numpy()
            )

            top5 = torch.topk(
                outputs,
                k=5,
                dim=1
            ).indices.cpu()

            for i in range(len(yb)):
                if yb[i].item() in top5[i]:
                    top5_correct += 1

            total += len(yb)

    acc = accuracy_score(
        actuals_all,
        preds_all
    )

    top5_acc = top5_correct / total

    print(
        f"Epoch {epoch+1}/{EPOCHS}"
        f" | Loss={running_loss:.4f}"
        f" | Val Acc={acc:.4f}"
        f" | Top5={top5_acc:.4f}"
    )

    if acc > best_acc:

        best_acc = acc

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print("Best Model Saved")

print(
    f"\nBest Validation Accuracy: {best_acc:.4f}"
)
