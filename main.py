# main.py
from src.preprocess import load_data, clean_data
from src.model import train_model, evaluate_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def main():
    df = load_data("data/water_quality.csv")
    df = clean_data(df)

    X = df.drop("Potability", axis=1)
    y = df["Potability"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model = train_model(model, X_train, y_train)
    accuracy, cm = evaluate_model(model, X_test, y_test)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    main()
