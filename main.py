from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.sklearn
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Optional
import uvicorn
import os

app = FastAPI(title="Titanic Survival Prediction", version="1.0.0")

#Load model from MLflow
def load_champion_model():
    try:
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "https://opportunistically-unneedful-seymour.ngrok-free.dev")
        mlflow.set_tracking_uri(mlflow_uri)
        model_name="rf-model-titanic"
        alias = "champion"
        model_uri=f"models:/{model_name}@{alias}"
        model = mlflow.sklearn.load_model(model_uri)
        return model
    except:
        #Fallback: try to load latest version
        try:
            model = mlflow.sklearn.load_model("models:/rf-model-titanic/latest")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from MLflow: {str(e)}")

#Initialize model and scaler
try:
    champion_model = load_champion_model()
    scaler = StandardScaler()
except Exception as e:
    print(f"Warning: {e}")
    champion_model = None
    scaler = None

class PassengerData(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str

class PredictionResponse(BaseModel):
    survived: int
    probability: float
    message: str

@app.get("/")
def read_root():
    return {
        "name": "Titanic Survival Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Make survival prediction",
            "/health": "GET - Health check",
            "/docs": "GET - Interactive API documentation"
        },
        "model": "Random Forest Classifier"
    }

@app.get("/docs")
def api_docs():
    return {
        "title": "Titanic Survival Prediction API Documentation",
        "version": "1.0.0",
        "description": "API for predicting passenger survival on the Titanic",
        "endpoints": [
            {
                "path": "/predict",
                "method": "POST",
                "description": "Predict survival for a passenger",
                "request_body": {
                    "Pclass": "Passenger class (1, 2, or 3)",
                    "Sex": "Passenger gender (male or female)",
                    "Age": "Passenger age in years",
                    "SibSp": "Number of siblings/spouses aboard",
                    "Parch": "Number of parents/children aboard",
                    "Fare": "Ticket fare amount",
                    "Embarked": "Port of embarkation (S, C, or Q)"
                },
                "response": {
                    "survived": "0 = Did not survive, 1 = Survived",
                    "probability": "Prediction confidence (0.0-1.0)",
                    "message": "Descriptive prediction message"
                },
                "example": {
                    "Pclass": 1,
                    "Sex": "female",
                    "Age": 25.0,
                    "SibSp": 0,
                    "Parch": 0,
                    "Fare": 250.0,
                    "Embarked": "S"
                }
            },
            {
                "path": "/health",
                "method": "GET",
                "description": "Check API health status"
            }
        ],
        "interactive_docs": "Visit /docs for interactive Swagger UI or /redoc for ReDoc"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(passenger: PassengerData):
    if champion_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Prepare data
        data = {
            'Pclass': passenger.Pclass,
            'Sex': passenger.Sex,
            'Age': passenger.Age,
            'SibSp': passenger.SibSp,
            'Parch': passenger.Parch,
            'Fare': passenger.Fare,
            'Embarked': passenger.Embarked
        }
        
        df = pd.DataFrame([data])
        
        #Encode categorical variables
        le_sex = LabelEncoder()
        le_embarked = LabelEncoder()
        df['Sex'] = le_sex.fit_transform(df['Sex'])
        df['Embarked'] = le_embarked.fit_transform(df['Embarked'])
    
        #Scale features
        X_scaled = scaler.fit_transform(df)
        
        #Make prediction
        prediction = champion_model.predict(X_scaled)[0]
        probability = champion_model.predict_proba(X_scaled)[0][int(prediction)]
        
        survived_message = "Survived" if prediction == 1 else "Did not survive"
        
        return PredictionResponse(
            survived=int(prediction),
            probability=float(probability),
            message=f"Passenger prediction: {survived_message} (confidence: {probability:.2%})"
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": champion_model is not None
    }

def main():
    print("Starting Titanic Survival Prediction API...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    main()
