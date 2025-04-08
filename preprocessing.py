
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

def load_data():
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

def save_data(df):
    df.to_csv('data/processed_diabetes.csv', index=False)
    
if __name__ == "__main__":
    df = load_data()
    save_data(df)