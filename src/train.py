from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import pickle

def train_model(X, y):
    # 🔥 Handle imbalance
    smote = SMOTE()
    X, y = smote.fit_resample(X, y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestClassifier(n_estimators=50, n_jobs=-1)
    model.fit(X_train, y_train)

    # Save model
    pickle.dump(model, open("models/model.pkl", "wb"))

    return model