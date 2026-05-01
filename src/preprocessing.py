import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):
    # Drop unnecessary column
    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    # Scale Amount
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df[['Amount']])

    X = df.drop('Class', axis=1)
    y = df['Class']

    return X, y