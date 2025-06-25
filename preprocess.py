import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(csv_path="data/heart.csv"):
    df = pd.read_csv(csv_path)
    X = pd.get_dummies(df.drop("HeartDisease", axis=1))
    y = df["HeartDisease"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, X.columns
