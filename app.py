import streamlit as st

from predict import predict_ticket

from gemini_reply import generate_reply

from database import (
    save_ticket,
    get_all_tickets,
    clear_history
)

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Ticket Classification System",
    page_icon="🎫",
    layout="wide"
)

# ==========================
# TITLE
# ==========================

st.title(
    "🎫 Automatic Ticket Classification System"
)

st.markdown(
    """
Classify customer tickets using LSTM and generate
AI-powered customer replies.
"""
)

# ==========================
# INPUT
# ==========================

ticket = st.text_area(
    "Enter Customer Ticket",
    height=200
)

# ==========================
# PREDICT BUTTON
# ==========================

if st.button("Predict Ticket"):

    if ticket.strip() == "":

        st.warning(
            "Please enter a ticket."
        )

    else:

        with st.spinner(
            "Processing..."
        ):

            queue = predict_ticket(
                ticket
            )

            reply = generate_reply(
                ticket,
                queue
            )

            save_ticket(
                ticket,
                queue,
                reply
            )

        st.success(
            "Prediction Completed"
        )

        st.subheader(
            "Predicted Queue"
        )

        st.info(
            queue
        )

        st.subheader(
            "Generated Reply"
        )

        st.write(
            reply
        )

# ==========================
# HISTORY SECTION
# ==========================

st.markdown("---")

st.header(
    "Ticket History"
)

history = get_all_tickets()

if history:

    for row in history:

        st.expander(
            f"Ticket #{row[0]}"
        ).write(
            {
                "Ticket":
                row[1],

                "Queue":
                row[2],

                "Reply":
                row[3],

                "Date":
                row[4]
            }
        )

else:

    st.info(
        "No tickets found."
    )

# ==========================
# CLEAR HISTORY
# ==========================

if st.button(
    "Clear History"
):

    clear_history()

    st.success(
        "History Cleared"
    )

    st.rerun()