import sqlite3
from datetime import datetime

# Database Connection
conn = sqlite3.connect(
    "tickets.db",
    check_same_thread=False
)

cursor = conn.cursor()


# Create Table
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS tickets(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        ticket_text TEXT,

        predicted_queue TEXT,

        generated_reply TEXT,

        created_at TEXT

    )
    """
)

conn.commit()


# ==========================
# SAVE TICKET
# ==========================

def save_ticket(
    ticket_text,
    predicted_queue,
    generated_reply
):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cursor.execute(
        """
        INSERT INTO tickets(
            ticket_text,
            predicted_queue,
            generated_reply,
            created_at
        )

        VALUES(?,?,?,?)
        """,
        (
            ticket_text,
            predicted_queue,
            generated_reply,
            timestamp
        )
    )

    conn.commit()


# ==========================
# GET ALL HISTORY
# ==========================

def get_all_tickets():

    cursor.execute(
        """
        SELECT *
        FROM tickets
        ORDER BY id DESC
        """
    )

    return cursor.fetchall()


# ==========================
# DELETE HISTORY
# ==========================

def clear_history():

    cursor.execute(
        "DELETE FROM tickets"
    )

    conn.commit()


# ==========================
# TEST DATABASE
# ==========================

if __name__ == "__main__":

    print(
        "Database initialized successfully!"
    )