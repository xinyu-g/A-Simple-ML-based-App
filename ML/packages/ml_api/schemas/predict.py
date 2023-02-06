from typing import Any, List, Optional, Dict

from pydantic import BaseModel


class PredictionResults(BaseModel):
    predictions: Optional[Any]


class InputData(BaseModel):
    inputs: List[Any]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "return": "lstm" 
                    }
                ]
            }
        }       