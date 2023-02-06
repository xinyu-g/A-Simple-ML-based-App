import json
from typing import Any

# import numpy as np
# import pandas as pd
from fastapi import APIRouter, HTTPException
# from fastapi.encoders import jsonable_encoder
import os
import sys
sys.path.append('..')
from Model.main import make_prediction
from loguru import logger

import schemas
# from app.config import settings

api_router = APIRouter()



@api_router.post("/predict", status_code=200)
async def predict(input_data) -> Any:
    """
    Make predictions with the Fraud detection model
    """

    # input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))

    # Advanced: You can improve performance of your API by rewriting the
    # `make prediction` function to be async and using await here.
    # logger.info(f"Making prediction on inputs: {input_data.inputs}")
    results = make_prediction(inputs=input_data)

    # if results["errors"] is not None:
    #     logger.warning(f"Prediction validation error: {results.get('errors')}")
    #     raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    logger.info(f"Prediction results: {results.get('predictions')}")

    return {'predictions': results.get('predictions')}