from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Step 1: Load model
with open("pipeline_v1.bin", "rb") as f_in:
    model = pickle.load(f_in)

# Step 2: Define input data model
class Lead(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

# Step 3: Create FastAPI app
app = FastAPI()

@app.post("/predict")
def predict(lead: Lead):
    X = [lead.dict()]
    probability = model.predict_proba(X)[0, 1]
    return {"conversion_probability": probability}
