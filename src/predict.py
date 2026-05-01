import pickle
import numpy as np

model = pickle.load(open("models/model.pkl", "rb"))

def predict(data):
    data = np.array(data).reshape(1, -1)
    return model.predict(data)