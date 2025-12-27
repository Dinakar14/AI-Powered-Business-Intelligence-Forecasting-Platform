from sqlalchemy import Column, Integer, String, Float, Date
from .database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    role = Column(String(20), default="user")

class Sales(Base):
    __tablename__ = "sales"

    id = Column(Integer, primary_key=True)
    sale_date = Column(Date)
    revenue = Column(Float)
    region = Column(String(50))
    product = Column(String(50))
