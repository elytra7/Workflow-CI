import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import os

# --- KONFIGURASI ---
EXPERIMENT_NAME = "Eksperimen_Kidney_Disease_Ardian"
ARTIFACT_PATH = "model_artifacts"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "kidney_disease_processed", "processed_data.csv")

def load_data(path):
    print(f"Mencoba membaca dataset dari: {path}") 
    

    if not os.path.exists(path):
        raise FileNotFoundError(f"File dataset tidak ditemukan di: {path}")
    
    # 2. BACA FILE 
    df = pd.read_csv(path)
    
    # 3. Pisahkan Fitur dan Target 

    X = df.drop(columns=['Dialysis_Needed'])
    y = df['Dialysis_Needed']
    return X, y

def main():
    # 1. Load Data
    print("Memuat data...")
    X, y = load_data(DATA_PATH)

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Apply SMOTE 
    print("Menerapkan SMOTE pada data training...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # 4. Setup MLflow Experiment

    with mlflow.start_run():
        print("Memulai Hyperparameter Tuning...")
        
        # --- MODEL & HYPERPARAMETER TUNING ---
        rf = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        
        # Grid Search
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
        grid_search.fit(X_train_resampled, y_train_resampled)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"Parameter Terbaik: {best_params}")

        # --- EVALUASI MODEL ---
        y_pred = best_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Accuracy: {accuracy}")

        # --- MANUAL LOGGING ---
        # Log Hyperparameters
        for param, value in best_params.items():
            mlflow.log_param(param, value)
            
        # Log Metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log Model
        print("Menyimpan model ke MLflow...")
        mlflow.sklearn.log_model(best_model, ARTIFACT_PATH)
        
        print("Model dan metrik berhasil disimpan ke MLflow.")

if __name__ == "__main__":
    main()