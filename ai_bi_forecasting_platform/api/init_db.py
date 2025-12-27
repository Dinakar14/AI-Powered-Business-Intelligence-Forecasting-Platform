from sqlalchemy.orm import Session
from api.database import SessionLocal
from api.models import User
from api.auth import get_password_hash

def init_db():
    db: Session = SessionLocal()

    admin = db.query(User).filter(User.username == "admin").first()
    if admin:
        print("⚠️ Admin already exists")
        db.close()
        return

    admin_user = User(
        username="admin",
        password=get_password_hash("admin123"),
        role="admin"
    )

    db.add(admin_user)
    db.commit()
    db.close()

    print("✅ Admin created")

if __name__ == "__main__":
    init_db()
