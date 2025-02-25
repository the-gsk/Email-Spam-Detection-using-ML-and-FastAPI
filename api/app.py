
import joblib
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI


app = FastAPI()

# Loading the model

try:
    model = joblib.load("./model/spam_classifier_v1.pkl")
    print("The model Loaded Successfully!!")
except Exception as e:
    print(f"There is error in loading Model: {e}")

# Input Schema
class EmailInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Spam Detection API is Running successfully!"}

@app.post("/predict")
def predict_spam(email: EmailInput):
    prediction = model.predict([email.text])[0]
    return {"spam": bool(prediction)}

if __name__ == "__main__":
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
