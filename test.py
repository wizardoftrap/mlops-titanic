import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocess import preprocess
import sys
import os

def test(df):
    #Set MLflow tracking URI
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "https://opportunistically-unneedful-seymour.ngrok-free.dev")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("titanic-survival-prediction")
    
    #features and target
    Y = df['Survived']
    X = df.drop('Survived', axis=1)
    
    #standard scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    #latest model
    model_name = "rf-model-titanic"
    model_version = "latest"
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(model_uri)
    
    #Get actual version number for champion alias
    client = mlflow.MlflowClient()
    model_versions = client.search_model_versions(f"name='{model_name}'")
    actual_version = str(max([int(v.version) for v in model_versions]))

    #test model
    Y_pred = model.predict(X)
    accuracy = accuracy_score(Y, Y_pred)
    precision = precision_score(Y, Y_pred)
    recall = recall_score(Y, Y_pred)
    f1 = f1_score(Y, Y_pred)
    
    #Log metrics
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1_score", f1)
    
    print(f"\nTest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    #passing criteria
    MIN_ACCURACY = 0.80
    MIN_F1 = 0.75
    
    test_passed = accuracy >= MIN_ACCURACY and f1 >= MIN_F1
    
    if test_passed:
        print("\nTests PASSED!")
        #champion alias to model version
        try:
            client.set_registered_model_alias(model_name, "champion", actual_version)
            print(f"Champion alias set to model version {actual_version}")
        except Exception as e:
            print(f"Error setting champion alias: {e}")
        return True
    else:
        print("\nTests FAILED!")
        print(f"Accuracy must be >= {MIN_ACCURACY}, got {accuracy:.4f}")
        print(f"F1 must be >= {MIN_F1}, got {f1:.4f}")
        return False

if __name__ == "__main__":
    df = pd.read_csv("dataset/test.csv")
    df = preprocess(df)
    test_passed = test(df)
    
    if not test_passed:
        sys.exit(1)
    sys.exit(0)
