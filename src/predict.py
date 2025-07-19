import joblib
import numpy as np

def predict_sales(tv, radio, newspaper, model_path="model/sales_model.pkl"):
    model = joblib.load(model_path)
    input_data = np.array([[tv, radio, newspaper]])
    prediction = model.predict(input_data)
    return prediction[0]
