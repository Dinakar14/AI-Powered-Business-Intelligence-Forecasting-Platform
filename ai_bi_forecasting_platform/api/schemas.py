from pydantic import BaseModel

class ChurnInput(BaseModel):
    Units_Sold: float
    Revenue: float
    Customer_Age: int
    Purchase_Frequency: int
    Month: int
    DayOfWeek: int

class RiskInput(BaseModel):
    Units_Sold: float
    Revenue: float
    Purchase_Frequency: int
    Month: int
