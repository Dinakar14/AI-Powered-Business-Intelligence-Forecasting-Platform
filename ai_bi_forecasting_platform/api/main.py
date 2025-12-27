from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from api.database import SessionLocal, engine
from api.models import User, Sales
from api.auth import hash_password, verify_password, create_access_token
from api.database import Base

Base.metadata.create_all(bind=engine)

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/register")
def register(username: str, password: str, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(400, "User exists")

    user = User(
        username=username,
        password=hash_password(password),
        role="user"
    )
    db.add(user)
    db.commit()
    return {"message": "User created"}

@app.post("/login")
def login(username: str, password: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()

    if not user or not verify_password(password, user.password):
        raise HTTPException(401, "Invalid credentials")

    token = create_access_token({"sub": user.username})
    return {"access_token": token, "role": user.role}
