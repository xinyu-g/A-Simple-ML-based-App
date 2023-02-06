

__version__ = "1.0.0"
# import json
# from typing import Any

# # import numpy as np
# # import pandas as pd
# from fastapi import APIRouter, HTTPException
# from fastapi.encoders import jsonable_encoder
# import schemas
# # from app.config import settings

# import sys
# import os
# sys.path.insert(0, '../../')
# from Model.main import make_prediction
# from loguru import logger



# api_router = APIRouter()


# @api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
# async def predict(input_data) -> Any:
#     """
#     Make predictions with the model
#     """

#     # input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))

#     # logger.info(f"Making prediction on inputs: {input_data.inputs}")
#     results = make_prediction(inputs=input_data)

#     # if results["errors"] is not None:
#     #     logger.warning(f"Prediction validation error: {results.get('errors')}")
#     #     raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

#     logger.info(f"Prediction results: {results.get('predictions')}")

#     return results