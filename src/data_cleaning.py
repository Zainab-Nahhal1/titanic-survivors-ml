import pandas as pd

def clean_data(df):
    df = df.drop_duplicates()
    # avoid chained assignment warnings by assigning back
    df = df.copy()
    if 'Age' in df.columns:
        df['Age'] = df['Age'].fillna(df['Age'].median())
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    if 'Fare' in df.columns:
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    if 'Cabin' in df.columns:
        df = df.drop(columns=['Cabin'])
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    if 'Embarked' in df.columns:
        df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    return df

def load_and_clean_data(train_path='data/train.csv', test_path='data/test.csv'):
    """Load train/test CSVs from data/ and return X, y, X_test, test_df.
    Make sure you place train.csv and test.csv in the data/ folder before running.
    """
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train_clean = clean_data(train)
    test_clean = clean_data(test)
    X = train_clean.drop(columns=[c for c in ['Survived','Name','Ticket','PassengerId'] if c in train_clean.columns])
    y = train_clean['Survived']
    X_test = test_clean.drop(columns=[c for c in ['Name','Ticket','PassengerId'] if c in test_clean.columns])
    return X, y, X_test, test_clean
