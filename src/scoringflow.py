from metaflow import FlowSpec, step
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn import datasets

class ScoringFlow(FlowSpec):

    @step
    def start(self):
        #Load the data
        X, y = datasets.load_wine(return_X_y=True)
        
        self.new_data = X[-10:]
        self.true_labels = y[-10:]
        print("Loaded new data for prediction.")
        self.next(self.load_model)

    @step
    def load_model(self):
        #Load model from MLFlow
        mlflow.set_tracking_uri('https://mlflow-run-745354261248.us-west2.run.app/')
        model_name = "metaflow-wine-model"
        
        model_uri = f"models:/{model_name}/Production"
        self.model = mlflow.sklearn.load_model(model_uri)
        print(f"Loaded model from MLflow: {model_uri}")
        self.next(self.predict)

    @step
    def predict(self):
        #Make predictions using loaded model
        self.predictions = self.model.predict(self.new_data)
        print("Predictions:", self.predictions)
        print("True labels:", self.true_labels)
        self.next(self.end)

    @step
    def end(self):
        print("Scoring complete.")

if __name__ == '__main__':
    ScoringFlow()
