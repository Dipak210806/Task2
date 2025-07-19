import pandas as pd
import os
from src.train_model import train_and_save_model
from src.predict import predict_sales

# Step 1: Create folders if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)

# Step 2: Download advertising dataset
url = "https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/Advertising.csv"
df = pd.read_csv(url)
df.to_csv("data/advertising.csv", index=False)

print("✅ Dataset downloaded and saved.")
print(df.head())
print("Total rows:", len(df))

# Step 3: Train the model
train_and_save_model("data/advertising.csv", "model/sales_model.pkl")

# Step 4: Predict sales
tv, radio, newspaper = 100, 50, 20
prediction = predict_sales(tv, radio, newspaper)
print(f"\n📈 Predicted Sales for TV={tv}, Radio={radio}, Newspaper={newspaper} is: {prediction:.2f}")
