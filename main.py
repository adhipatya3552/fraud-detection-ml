import os
from src.preprocessing import load_data, preprocess
from src.train import train_model
from src.evaluate import evaluate_model

DATA_PATH = "data/creditcard_2023.csv"

def run_pipeline():
    print("🚀 Starting ML Pipeline...")

    # Step 1: Load data
    df = load_data(DATA_PATH)
    print("✅ Data loaded")

    # Step 2: Preprocess
    X, y = preprocess(df)
    print("✅ Data preprocessed")

    # Step 3: Train model
    model = train_model(X, y)
    print("✅ Model trained")

    # Step 4: Evaluate model
    evaluate_model(model,X, y)
    print("✅ Model evaluated")

    print("🎉 Pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()