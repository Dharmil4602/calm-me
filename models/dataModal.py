'''
There are serveral HRV values present for individual user at different timestamps, so, instead of processing individual values, we can process the data in batch.
Data Required for Batch Processing:
1. healthData -> value [this is required for training the model and getting the result]
'''

from pydantic import BaseModel
from datetime import datetime

class StressDataEntry(BaseModel):
    hrvValue: float
    prediction: str
    predictionIdentifier: int
    dateFrom: datetime
    dateTo: datetime

class HRV(BaseModel):
    email: str
    '''Below data, that is array of objects that contains HRV value is required for training the model and getting the result'''
    healthData: list[StressDataEntry]