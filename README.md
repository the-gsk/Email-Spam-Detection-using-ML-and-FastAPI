# 📧 Email Spam Detection API

## 📌 Project Overview
This project is a **Machine Learning-based Email Spam Detector** using NLP techniques. It includes:
- A trained **Naïve Bayes** model for spam classification.
- A **FastAPI**-based REST API for training, prediction, and hyperparameter tuning.
- **MLflow** for experiment tracking.
- A **Dockerized API** for easy deployment.

---

## 📂 Project Structure
```
Email-Spam-Detection-using-ML-and-Flask/
│── api/
│   ├── app.py            # FastAPI backend
│── model/
│   ├── train.py          # Model training script
│   ├── spam_classifier.pkl  # Saved trained model (Generated after training)
│── data/
│   ├── spam.csv          # Sample dataset
│── requirements.txt      # Required dependencies
│── Dockerfile            # Docker setup
│── README.md             # Project Documentation
```

---

## 🚀 Installation & Setup
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/the-gsk/Email-Spam-Detection-using-ML-and-FastAPI.git
```
```sh
cd Email-Spam-Detection-using-ML-and-FastAPI
```

### 2️⃣ Create a Virtual Environment
```sh
python -m venv venv
```
```sh
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

---

## 🏋️ Train the Model
```sh
python model/train.py
```
✅ This will save the trained model as `model/spam_classifier.pkl`.

---

## 🌐 Run the API
```sh
python api/app.py
```
---- OR Try Run with Uvicorn----
```sh
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```
- 🔹 The API will be available at: [http://localhost:8000](http://localhost:8000)
- 🔹 API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📦 Docker Deployment
### 1️⃣ Build the Docker Image
```sh
docker build -t spam-detection-api .
```
### 2️⃣ Run the Container
```sh
docker run -p 8000:8000 spam-detection-api
```
Now, your API is running inside Docker!

## Test it locally

docker run -p 8000:8000 spam-detection-api

curl http://localhost:8000/best_params

## Docker Hub

-- 1st create docker hub account
```sh  docker login
```

```sh docker tag spam-detection-api herambithape/spam-detection-api:latest 
```

```sh docker push herambithape/spam-detection-api:latest   ```

```sh docker pull herambithape/spam-detection-api:latest ```

```sh docker run -p 8000:8000 herambithape/spam-detection-api:latest      ```





---

## 📤 Push to Docker Hub (Optional)
1️⃣ Login to Docker Hub:
```sh
docker login
```
2️⃣ Tag the Image:
```sh
docker tag spam-detection-api your_dockerhub_username/spam-detection-api
```
3️⃣ Push the Image:
```sh
docker push your_dockerhub_username/spam-detection-api
```
Now, others can pull and run your container. -->

---

## 📲 API Endpoints
### 🔹 **1. Home**
- **`GET /`** → Returns API status.

### 🔹 **2. Predict Spam**
- **`POST /predict`**
- **Request:**
```json
{
  "text": "WINNER!! You have won a free prize. Call now!"
}
```
- **Response:**
```json
{
  "spam": true
}
```

### 🔹 **3. Train Model**
- **`POST /train`** → Retrains the model with new data.

---

## 🛠 Technologies Used
- **Python** (FastAPI, Scikit-learn, Joblib, MLflow)
- **Machine Learning** (Naïve Bayes, NLP, CountVectorizer)
- **Docker** (Containerization)
- **REST API** (FastAPI)

---

## 📌 Future Enhancements
✅ Implement deep learning models like LSTMs.
✅ Add more preprocessing techniques for better accuracy.
✅ Deploy on cloud services like AWS/GCP.

---

## 👨‍💻 Author
🔹 **Gaurav Shankar Kumar**
📧 gauravshankarkumar@gmail.com 
🔹 **Heramb Ithape**
📧 heramb.analytics@gmail.com 

  

---

🎯 **Star the repo if you found this useful! ⭐**

