import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocess import preprocess
import os

def train(df):
    #MLflow tracking URI
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "https://opportunistically-unneedful-seymour.ngrok-free.dev")
    mlflow.set_tracking_uri(mlflow_uri)
    
    #features and target
    Y = df['Survived']
    X = df.drop('Survived', axis=1)
    
    #standard scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    #MLflow experiment
    mlflow.set_experiment("titanic-survival-prediction")
    
    with mlflow.start_run():
        #Train model
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, Y)
        
        #Evaluate - Training metrics only
        Y_pred_train = model.predict(X)
        accuracy = accuracy_score(Y, Y_pred_train)
        precision = precision_score(Y, Y_pred_train)
        recall = recall_score(Y, Y_pred_train)
        f1 = f1_score(Y, Y_pred_train)
        
        #Log metrics
        mlflow.log_metric("training_accuracy", accuracy)
        mlflow.log_metric("training_precision", precision)
        mlflow.log_metric("training_recall", recall)
        mlflow.log_metric("training_f1_score", f1)
        
        # Log parameters
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 15)
        mlflow.log_param("min_samples_split", 2)
        mlflow.log_param("min_samples_leaf", 1)
        mlflow.log_param("class_weight", "balanced")
        
        # Save model
        mlflow.sklearn.log_model(
            sk_model=model,
            name="rf-model-titanic",
            input_example=X,
            registered_model_name="rf-model-titanic",
        )
        
        print(f"\nModel trained and saved to MLflow")
        print(f"Training Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        return

if __name__ == "__main__":
    df = pd.read_csv("dataset/train.csv")
    df = preprocess(df)
    train(df)
