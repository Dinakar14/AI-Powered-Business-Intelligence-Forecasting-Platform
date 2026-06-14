import google.generativeai as genai

from config import GEMINI_API_KEY

genai.configure(
    api_key=GEMINI_API_KEY
)

def generate_reply(
    ticket_text,
    queue
):

    prompt = f"""
You are a professional customer support executive.

Ticket Category:
{queue}

Customer Ticket:
{ticket_text}

Generate a polite, empathetic acknowledgment reply.

Keep it professional.
"""

    try:

        model = genai.GenerativeModel(
            "gemini-2.0-flash"
        )

        response = model.generate_content(
            prompt
        )

        return response.text

    except Exception:

        return f"""
Thank you for contacting support regarding your {queue} issue.

We have successfully received your request and our support team will review it shortly.

Thank you for your patience.
"""