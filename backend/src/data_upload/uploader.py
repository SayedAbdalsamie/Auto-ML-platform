import os
import pandas as pd


def save_and_analyze_file(file):
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".csv", ".xlsx"]:
        return None, {"error": "Unsupported file type. Only .csv and .xlsx allowed."}
    try:
        if ext == ".csv":
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        info = {
            "message": "File loaded successfully",
            "shape": df.shape,
            "columns": df.columns.tolist(),
        }
        return df, info
    except Exception as e:
        return None, {"error": f"Failed to process file: {str(e)}"}
