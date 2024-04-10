
from pydantic import BaseModel
class FinanceData(BaseModel):
    coin_name:str
    coin_id:str
    currency:str

