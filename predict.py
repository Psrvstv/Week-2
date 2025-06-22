import joblib
import pandas as pd

model = joblib.load("../models/water_quality_model.pkl")

sample = pd.DataFrame([{ 
    "pH": 7.0,
    "Hardness": 200.0,
    "Solids": 15000.0,
    "Chloramines": 7.0,
    "Sulfate": 300.0,
    "Conductivity": 500.0,
    "Organic_carbon": 10.0,
    "Trihalomethanes": 85.0,
    "Turbidity": 3.0
}])

prediction = model.predict(sample)
print("Potable" if prediction[0] == 1 else "Not Potable")
