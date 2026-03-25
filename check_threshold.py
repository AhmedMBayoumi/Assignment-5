import mlflow
import os
import sys

def check_threshold():
    if not os.path.exists("model_info.txt"):
        print("Error: model_info.txt not found.")
        sys.exit(1)
        
    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()
    
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("Error: MLFLOW_TRACKING_URI environment variable not set.")
        sys.exit(1)
    
    mlflow.set_tracking_uri(tracking_uri)
    
    try:
        run = mlflow.get_run(run_id)
        accuracy = run.data.metrics.get("accuracy")
        
        if accuracy is None:
            print(f"Error: Accuracy metric not found for Run ID: {run_id}")
            sys.exit(1)
            
        print(f"Model Accuracy: {accuracy}")
        
        threshold = 0.85
        if accuracy < threshold:
            print(f"Failing: Accuracy {accuracy} is below threshold {threshold}")
            sys.exit(1)
        else:
            print(f"Success: Accuracy {accuracy} is above threshold {threshold}")
            sys.exit(0)
            
    except Exception as e:
        print(f"Error connecting to MLflow or fetching run data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    check_threshold()
