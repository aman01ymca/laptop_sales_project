from typing import Any, List, Optional
import datetime

from pydantic import BaseModel
from laptop_sales_model.processing.validation import DataInputSchema


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    #predictions: Optional[List[int]]
    predictions: Optional[int]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                "Make": "Dell", # datetime.datetime.strptime("2012-11-05", "%Y-%m-%d"),  
                "Colour": "White", 
                "Usage (Hours)": 700,
                "USB Ports": 3,
                    }
                ]
            }
        }
