from metaflow import FlowSpec, step, Parameter, kubernetes, resources, conda_base
import mlflow
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

@conda_base(python='3.12')
class ClassifierTrainFlow(FlowSpec):

    test_size = Parameter('test_size', default=0.2)
    random_seed = Parameter('random_seed', default=42)

    @resources(memory="4096", cpu=2)
    @kubernetes
    @step
    def start(self):
        #Load the data
        X, y = datasets.load_wine(return_X_y=True)
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_seed)
        self.next(self.train_knn, self.train_rf)

    @resources(memory="4096", cpu=2)
    @kubernetes
    @step
    def train_knn(self):
        #Train KNN Model
        self.model = KNeighborsClassifier()
        self.model.fit(self.train_data, self.train_labels)
        self.model_name = "knn"
        self.test_data = self.test_data
        self.test_labels = self.test_labels
        self.next(self.choose_best_model)

    @resources(memory="4096", cpu=2)
    @kubernetes
    @step
    def train_rf(self):
        #Train Random Forest Model
        self.model = RandomForestClassifier(random_state=self.random_seed)
        self.model.fit(self.train_data, self.train_labels)
        self.model_name = "random_forest"
        self.test_data = self.test_data
        self.test_labels = self.test_labels
        self.next(self.choose_best_model)

    @resources(memory="4096", cpu=2)
    @kubernetes
    @step
    def choose_best_model(self, inputs):
        #Identify best model based on performance
        mlflow.set_tracking_uri('http://localhost:5001')
        mlflow.set_experiment('metaflow-experiment')

        def evaluate(model_input):
            model = model_input.model
            name = model_input.model_name
            score = model.score(model_input.test_data, model_input.test_labels)
            return name, model, score

        scores = list(map(evaluate, inputs))
        scores.sort(key=lambda x: -x[2])

        self.best_model_name, self.best_model, self.best_score = scores[0]

        #Save best model to MLFlow
        with mlflow.start_run() as run:
            mlflow.sklearn.log_model(
                sk_model=self.best_model,
                artifact_path="model",
                registered_model_name="metaflow-wine-model"
            )
            self.best_run_id = run.info.run_id

        self.next(self.register_best)

    @resources(memory="4096", cpu=2)
    @kubernetes
    @step
    def register_best(self):
        #Register Best Model
        mlflow.set_tracking_uri('http://localhost:5001')
        mlflow.set_experiment("metaflow-experiment")

        print(f"Registered model '{self.best_model_name}' with run ID {self.best_run_id}")
        self.next(self.end)

    @step
    def end(self):
        print(f"Best Model: {self.best_model_name}")
        print(f"Accuracy: {self.best_score:.4f}")

if __name__ == '__main__':
    ClassifierTrainFlow()
