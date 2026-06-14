import torch
import torch.nn.functional as F
import joblib

from model import TicketLSTM
from preprocessing import clean_text, encode_text
from config import *

# ==========================
# LOAD VOCAB & LABEL ENCODER
# ==========================

try:
    vocab = joblib.load(VOCAB_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

except Exception as e:
    raise Exception(
        f"Error loading vocab or label encoder: {e}"
    )

# ==========================
# LOAD MODEL
# ==========================

try:

    model = TicketLSTM(
        vocab_size=len(vocab),
        embed_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=len(label_encoder.classes_)
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu"
        )
    )

    model.eval()

except Exception as e:

    raise Exception(
        f"Error loading model: {e}"
    )


# ==========================
# PREDICT FUNCTION
# ==========================

def predict_ticket(ticket_text, debug=False):

    # Clean text
    ticket_text = clean_text(ticket_text)

    if len(ticket_text.strip()) == 0:
        return "Invalid Ticket"

    # Encode
    encoded = encode_text(
        ticket_text,
        vocab,
        MAX_SEQ_LEN
    )

    encoded = torch.LongTensor(
        [encoded]
    )

    with torch.no_grad():

        logits = model(encoded)

        probabilities = F.softmax(
            logits,
            dim=1
        )

        predicted_idx = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_idx
        ].item()

    predicted_queue = (
        label_encoder
        .inverse_transform(
            [predicted_idx]
        )[0]
    )

    if debug:

        print("\n========== DEBUG ==========")

        print(
            "Input Ticket:",
            ticket_text
        )

        print(
            "\nClass Probabilities:"
        )

        for idx, cls in enumerate(
            label_encoder.classes_
        ):

            print(
                f"{cls}: "
                f"{probabilities[0][idx].item():.4f}"
            )

        print(
            "\nPredicted:",
            predicted_queue
        )

        print(
            "Confidence:",
            round(
                confidence * 100,
                2
            ),
            "%"
        )

        print("===========================\n")

    return predicted_queue


# ==========================
# TEST
# ==========================

if __name__ == "__main__":

    sample_ticket = input(
        "Enter Ticket: "
    )

    result = predict_ticket(
        sample_ticket,
        debug=True
    )

    print(
        "Predicted Queue:",
        result
    )