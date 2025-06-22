import pandas as pd
from sklearn.impute import SimpleImputer

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    imputer = SimpleImputer(strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df_imputed
