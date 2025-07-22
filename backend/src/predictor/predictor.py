import joblib
import pandas as pd


def predict_from_model(model_path, input_data, preprocessing_pipeline=None):
    model = joblib.load(model_path)
    # Convert JSON to DataFrame if needed
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        df = input_data
    else:
        raise ValueError("Unsupported input data type")
    # Optional preprocessing
    if preprocessing_pipeline:
        df = preprocessing_pipeline(df)
    preds = model.predict(df)
    return preds.tolist()
