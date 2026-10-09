import mlflow
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

# 1. Load the California Housing dataset
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = california.target

# 2. Split it into 80% Training and 20% Validation
# 3. Make the split reproducible by using a fixed random seed
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)

print("Dataset loaded and split successfully.")

# Define a name for the experiment to group our runs
mlflow.set_experiment("House_Price_Prediction_Experiment")

# Part 2: Train Multiple Experiments with different configurations
configurations = [
    {"run_name": "Run 1", "max_depth": 3, "learning_rate": 0.1},
    {"run_name": "Run 2", "max_depth": 5, "learning_rate": 0.05},
    {"run_name": "Run 3", "max_depth": 7, "learning_rate": 0.01},
]

print("Starting MLflow experiments...")

# Part 3: Log the Model
for config in configurations:
    # Start an MLflow run for each configuration
    with mlflow.start_run(run_name=config["run_name"]):
        
        # 1. Train the model
        model = GradientBoostingRegressor(
            max_depth=config["max_depth"], 
            learning_rate=config["learning_rate"], 
            random_state=42 # fixed random state for reproducibility
        )
        model.fit(X_train, y_train)
        
        # Predict on the validation set to evaluate performance
        y_pred = model.predict(X_val)
        
        # 2. Calculate the validation metrics
        rmse = root_mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        # 3. Log the parameters
        mlflow.log_param("max_depth", config["max_depth"])
        mlflow.log_param("learning_rate", config["learning_rate"])
        
        # 4. Log the metrics
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)
        
        # 5. Save the trained model as an MLflow artifact
        mlflow.sklearn.log_model(
            model, 
            "model", 
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )
        
        print(f"{config['run_name']} completed. RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

print("All experiments finished and logged successfully!")
