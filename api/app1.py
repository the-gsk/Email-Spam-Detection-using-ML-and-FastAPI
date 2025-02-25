import uvicorn
from pydantic import baseModel
from fastapi import FastAPI
import joblib

app = FastAPI()

## Loading the model from pickle file

try:
    model = joblib.load("./model/spam_classifier_v1.pkl")
    print("The model successfully loaded and ready to use!!")
except Exception as e:
    print(f"There is an error in loading models: {e}")

# Input Schema
class EmailInput(baseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Spam dection API is running successfully !!"}

@app.post("/predict")
def predict_spam(email: EmailInput):
    prediction = model.predict([email.text])[0]
    return {"spam": bool(prediction)}

if __name__ == "__main__":
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
