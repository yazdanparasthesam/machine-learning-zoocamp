import pickle

# Step 1: Load the trained pipeline
with open("pipeline_v1.bin", "rb") as f_in:
    model = pickle.load(f_in)

# Step 2: Create the sample record
record = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

# Step 3: Predict probability of conversion
X = [record]
prob = model.predict_proba(X)[0, 1]

print(prob)
