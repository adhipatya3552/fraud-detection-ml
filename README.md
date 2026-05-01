# 💳 Credit Card Fraud Detection — ML Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Engine-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-green?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue?style=for-the-badge&logo=docker&logoColor=white)
![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?style=for-the-badge&logo=render&logoColor=white)
![SMOTE](https://img.shields.io/badge/SMOTE-Imbalance%20Handling-purple?style=for-the-badge)

**An end-to-end Machine Learning system that detects fraudulent credit card transactions using real-world anonymized data — featuring a full modular ML pipeline, a FastAPI backend for real-time predictions, an interactive Streamlit web UI, and Docker-based cloud deployment on Render.**

🔗 **Live API:** [https://fraud-detection-ml-jp1y.onrender.com](https://fraud-detection-ml-jp1y.onrender.com)
🔗 **Live Web App:** [Streamlit Cloud](https://fraud-detection-ml-adhipatya3552.streamlit.app)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Features](#-features)
- [How It Works — Step by Step](#-how-it-works--step-by-step)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Getting Started](#-getting-started)
- [Running the Project](#-running-the-project)
- [API Usage](#-api-usage)
- [Streamlit Web App](#-streamlit-web-app)
- [Cloud Deployment](#-cloud-deployment)
- [Docker Deployment](#-docker-deployment)
- [Model Evaluation & Results](#-model-evaluation--results)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Module Reference](#-module-reference)
- [Known Limitations](#-known-limitations)
- [Roadmap](#-roadmap)

---

## 📌 Overview

Credit card fraud is one of the most critical problems in financial technology. Even a tiny percentage of fraudulent transactions can result in millions of dollars in losses for banks, businesses, and customers alike. Building systems that can detect fraud automatically — in real time — is a key challenge in applied machine learning.

This project builds a **complete, production-style machine learning pipeline** that:

- Loads and preprocesses anonymized real-world credit card transaction data
- Handles the heavily imbalanced nature of fraud datasets using **SMOTE** (Synthetic Minority Over-sampling Technique)
- Trains a **Random Forest classifier** to distinguish fraudulent from legitimate transactions
- Evaluates the model using classification reports, confusion matrices, and ROC curves
- Saves the trained model using **pickle** for reuse without retraining
- Exposes predictions through a **FastAPI REST API** that returns a fraud probability score in real time
- Provides an interactive **Streamlit web app** where users can input transaction features and immediately get a fraud prediction
- Packages everything inside a **Dockerfile** for easy deployment

> The goal is not just to train a model, but to simulate how a real-world ML fraud detection system is designed, built, and deployed — from raw data to a working API.

---

## 🌍 Live Demo

| Service | URL | Status |
|---------|-----|--------|
| **🌐 Web App** | [Streamlit Cloud](https://fraud-detection-ml-adhipatya3552.streamlit.app) | ✅ Live |
| **🔗 API** | [https://fraud-detection-ml-jp1y.onrender.com](https://fraud-detection-ml-jp1y.onrender.com) | ✅ Live |
| **📖 API Docs** | [Swagger UI](https://fraud-detection-ml-jp1y.onrender.com/docs) | ✅ Live |

> ⚠️ **Note:** The Render free tier spins down after inactivity. The first request after idle may take ~30 seconds while the service restarts.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📊 **Data Preprocessing** | Drops unnecessary columns, scales `Amount` using StandardScaler |
| ⚖️ **Imbalance Handling** | Uses SMOTE to synthetically oversample the minority fraud class |
| 🤖 **Model Training** | Trains a Random Forest classifier with parallelized fitting (`n_jobs=-1`) |
| 📈 **Evaluation Metrics** | Outputs precision, recall, F1-score, and confusion matrix |
| 📉 **Visualization** | Saves confusion matrix heatmap and ROC curve as `.png` files |
| 🌐 **FastAPI Backend** | `/predict` endpoint returns fraud label + probability score |
| 💳 **Streamlit UI** | Interactive web app with 29 input fields, sample loaders, and live predictions |
| 🧪 **Sample Testing** | Load real normal or fraud samples from the dataset directly in the UI |
| 📦 **Modular Pipeline** | Each stage (load, preprocess, train, evaluate, predict) is in its own file |
| 🐳 **Docker Ready** | Dockerfile included for containerized API deployment |
| ☁️ **Cloud Deployed** | API deployed on Render, Web App deployed on Streamlit Cloud |
| 💾 **Model Persistence** | Trained model saved as `model.pkl` and reloaded without retraining |

---

## ⚙️ How It Works — Step by Step

### 1. 📥 Data Loading & Preprocessing (`src/preprocessing.py`)

The raw dataset is loaded as a Pandas DataFrame from `data/creditcard_2023.csv`.

Here's what preprocessing does:

1. The `id` column is dropped if it exists — it's just a row identifier with no predictive value
2. The `Amount` column is scaled using **StandardScaler** — this normalizes the amount feature to have zero mean and unit variance, which is important because the other 28 features (V1–V28) are already PCA-transformed and have much smaller ranges
3. The `Class` column is separated as the target variable `y` (0 = legitimate, 1 = fraud)
4. All remaining columns become the feature matrix `X`

```python
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

X = df.drop('Class', axis=1)
y = df['Class']
```

> **Why scale Amount?** If you don't scale Amount, the model's distance-based calculations can be heavily biased by large values. Scaling ensures it contributes proportionally alongside V1–V28.

---

### 2. ⚖️ Handling Class Imbalance with SMOTE (`src/train.py`)

Fraud datasets are almost always highly imbalanced — fraudulent transactions represent a tiny fraction of total transactions. If you train a model on this as-is, it will learn to simply predict "not fraud" for everything and still achieve 99%+ accuracy, which is useless.

To fix this, **SMOTE (Synthetic Minority Over-sampling Technique)** is applied:

1. SMOTE looks at the minority class (fraud cases) and generates **synthetic new fraud samples** by interpolating between existing fraud examples in feature space
2. After SMOTE, the dataset has a balanced 50-50 split between legitimate and fraudulent transactions
3. Only then is the model trained — so it gets equal exposure to both classes

```python
smote = SMOTE()
X, y = smote.fit_resample(X, y)
```

> **Why not just undersample?** Undersampling (discarding majority class data) wastes real information. SMOTE instead creates new data, so the model sees more patterns without information loss.

---

### 3. 🤖 Model Training — Random Forest (`src/train.py`)

After SMOTE resampling, the data is split 80/20 into training and test sets:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

A **Random Forest Classifier** is then trained on the training set:

```python
model = RandomForestClassifier(n_estimators=50, n_jobs=-1)
model.fit(X_train, y_train)
```

**Why Random Forest?**

| Reason | Explanation |
|--------|-------------|
| **Ensemble method** | Combines 50 decision trees — each tree votes, and the majority wins |
| **Handles non-linear data** | Credit card features don't have clean linear relationships |
| **Robust to overfitting** | Individual trees overfit, but the ensemble averages them out |
| **Feature importance** | Can tell you which features (V1–V28, Amount) matter most |
| **`n_jobs=-1`** | Uses all CPU cores for parallel training — much faster on large data |

The trained model is saved using `pickle` so it can be reloaded by the API and Streamlit app without retraining every time:

```python
pickle.dump(model, open("models/model.pkl", "wb"))
```

---

### 4. 📈 Evaluation (`src/evaluate.py`)

After training, the model is evaluated on the held-out 20% test set. The evaluation step generates three types of output:

**A) Classification Report (printed to terminal)**

```
              precision    recall  f1-score   support
           0       ...       ...      ...        ...
           1       ...       ...      ...        ...
```

- **Precision** — Out of all transactions predicted as fraud, how many actually were fraud?
- **Recall** — Out of all actual fraud cases, how many did the model catch?
- **F1-score** — Harmonic mean of precision and recall (the most useful single metric for imbalanced classification)

**B) Confusion Matrix (saved as `outputs/confusion_matrix.png`)**

A heatmap showing True Positives, True Negatives, False Positives, and False Negatives at a glance. Plotted using Seaborn.

**C) ROC Curve (saved as `outputs/roc_curve.png`)**

Shows the trade-off between True Positive Rate and False Positive Rate across different decision thresholds. The AUC score (Area Under the Curve) is displayed in the legend — a higher AUC means the model separates fraud from non-fraud more cleanly.

---

### 5. 🌐 Real-Time Prediction via FastAPI (`api/app.py`)

The trained model is loaded by the FastAPI app at startup. A Pydantic model defines the expected request body:

```python
class Transaction(BaseModel):
    features: list[float]
```

The `/predict` endpoint receives 29 float values (V1–V28 + scaled Amount), reshapes them, runs `predict_proba()` to get the fraud probability, and returns:

```json
{
  "fraud": 1,
  "probability": 0.87,
  "message": "High risk transaction"
}
```

The threshold is set at 0.5 — if the model's estimated probability of fraud exceeds 50%, the transaction is flagged.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        USER INTERACTION                          │
│              Streamlit UI  ──or──  Direct API Call               │
└──────────────────────────┬───────────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │   FastAPI /predict      │
              │   (api/app.py)          │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   Load model.pkl        │
              │   (models/model.pkl)    │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   Random Forest         │
              │   .predict_proba()      │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   Fraud / Not Fraud     │
              │   + Probability Score   │
              └─────────────────────────┘
```

**Training Pipeline Flow:**

```
data/creditcard_2023.csv
         │
         ▼
  preprocessing.py  ──▶  Drop id, Scale Amount, Split X/y
         │
         ▼
    train.py  ──▶  SMOTE resampling  ──▶  Train/Test Split  ──▶  RandomForest.fit()
         │
         ▼
  models/model.pkl  (saved via pickle)
         │
         ▼
   evaluate.py  ──▶  Classification Report
                ──▶  outputs/confusion_matrix.png
                ──▶  outputs/roc_curve.png
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.10+ | Core programming language |
| **Data Processing** | Pandas, NumPy | Loading, cleaning, and transforming transaction data |
| **ML** | Scikit-Learn | StandardScaler, RandomForestClassifier, train_test_split, metrics |
| **Imbalance Handling** | imbalanced-learn (SMOTE) | Oversampling the minority fraud class |
| **Model Storage** | Pickle | Serializing and loading the trained model |
| **API Backend** | FastAPI + Uvicorn | REST endpoint for real-time fraud prediction |
| **Request Validation** | Pydantic | Validates the incoming feature list structure |
| **Web UI** | Streamlit | Interactive browser-based transaction testing app |
| **Visualization** | Matplotlib, Seaborn | Confusion matrix heatmap and ROC curve |
| **EDA** | Jupyter Notebook | Exploratory data analysis of the dataset |
| **Containerization** | Docker | Packaging the API for deployment |
| **Cloud Hosting** | Render | Production deployment of the FastAPI backend |
| **App Hosting** | Streamlit Cloud | Production deployment of the web UI |

---

## 📁 Project Structure

```
fraud-detection-ml/
│
├── api/
│   └── app.py                  # FastAPI app — loads model, exposes /predict endpoint
│
├── app/
│   └── streamlit_app.py        # Streamlit UI — 29 input fields, sample loaders, live predictions
│
├── data/
│   └── creditcard_2023.csv     # Dataset (downloaded from Kaggle — not included in repo)
│
├── models/
│   └── model.pkl               # Trained Random Forest model (generated by main.py)
│
├── notebooks/
│   └── eda.ipynb               # Jupyter notebook — class distribution, dataset exploration
│
├── outputs/
│   ├── confusion_matrix.png    # Heatmap of model predictions vs actual labels
│   └── roc_curve.png           # ROC curve with AUC score
│
├── src/
│   ├── preprocessing.py        # Loads CSV, drops id, scales Amount, returns X and y
│   ├── train.py                # SMOTE + train/test split + RandomForest training + model save
│   ├── evaluate.py             # Metrics, confusion matrix plot, ROC curve plot
│   └── predict.py              # Loads model.pkl and runs inference on a single input
│
├── Dockerfile                  # Container setup for deploying the FastAPI backend
├── requirements.txt            # Python dependencies
├── main.py                     # Pipeline entry point — runs all 4 stages in sequence
└── README.md                   # This file
```

---

## 📊 Dataset

### Source

This project uses the **Credit Card Fraud Detection Dataset 2023**, published on Kaggle by **[Nidula Elgiriyewithana](https://www.kaggle.com/nelgiriyewithana)**.

> 🔗 **Kaggle Link:** [https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)

The dataset contains anonymized credit card transaction records from European cardholders in 2023. It was created specifically for training fraud detection models and includes both fraudulent and legitimate transaction records.

### Key Facts

- Over **550,000 transaction records**
- Features **V1 through V28** are the result of **PCA (Principal Component Analysis)** transformation applied to protect cardholder privacy — the original feature names and meanings cannot be recovered
- The `Amount` field represents the transaction amount in the original scale
- The `Class` field is the target: `0` = legitimate transaction, `1` = fraudulent transaction
- Unlike the older 2019 version of this dataset, **this 2023 version has a more balanced class distribution**, making it a better benchmark dataset

### Column Structure

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer | Row identifier — dropped during preprocessing |
| `V1` – `V28` | Float | Anonymized PCA-transformed features — the actual transaction attributes |
| `Amount` | Float | Transaction amount (in original currency units) — scaled before training |
| `Class` | Integer | Target label — `0` for legitimate, `1` for fraud |

### Why This Dataset?

| Reason | Detail |
|--------|--------|
| **Real-world origin** | Based on actual European credit card transactions |
| **Privacy-safe** | PCA transformation ensures no cardholder data is exposed |
| **Benchmark quality** | Widely used in the ML community for fraud detection research |
| **Balanced enough** | Unlike older fraud datasets, this version doesn't require extreme resampling ratios |
| **Clear target** | Binary `Class` label makes it a clean supervised classification problem |

### Preprocessing Applied

| Step | What Happens |
|------|--------------|
| Drop `id` | Removed — no predictive value, just a sequential row number |
| Scale `Amount` | Transformed with StandardScaler so it matches the scale of V1–V28 |
| SMOTE | Applied during training to balance the class distribution synthetically |

> ⚠️ **Note:** The dataset file (`creditcard_2023.csv`) is not included in this repository due to its size. Download it from the Kaggle link above and place it at `data/creditcard_2023.csv` before running the project.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or above
- pip
- Docker (optional — only needed for containerized deployment)
- A Kaggle account (to download the dataset)

### 1. Clone the Repository

```bash
git clone https://github.com/adhipatya3552/fraud-detection-ml.git
cd fraud-detection-ml
```

### 2. Create a Virtual Environment

It is strongly recommended to use a virtual environment to keep all dependencies isolated from your system Python installation.

```bash
# Create virtual environment
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:

```
pandas
numpy
streamlit
fastapi
uvicorn
seaborn
matplotlib
python-multipart
imbalanced-learn
scikit-learn
```

### 4. Download the Dataset from Kaggle

Go to the Kaggle dataset page:

> 🔗 [https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)

Download the CSV file and place it inside the `data/` folder:

```
data/creditcard_2023.csv
```

> ⚠️ **The dataset is required to run the ML pipeline locally.** `main.py` expects it at exactly this path. The deployed Streamlit app works without the dataset — the sample loader buttons are gracefully disabled when the CSV is not available.

---

## ▶️ Running the Project

### Option 1 — Run the Full ML Pipeline

This runs all four stages in sequence: load → preprocess → train → evaluate.

```bash
python main.py
```

You will see output like this:

```
🚀 Starting ML Pipeline...
✅ Data loaded
✅ Data preprocessed
✅ Model trained
✅ Model evaluated
🎉 Pipeline completed successfully!
```

After this completes:
- `models/model.pkl` — the trained Random Forest model (used by the API and Streamlit app)
- `outputs/confusion_matrix.png` — visual evaluation of model predictions
- `outputs/roc_curve.png` — ROC curve with AUC score

> ⚠️ **Run `main.py` first** before starting the API or the Streamlit app. Both depend on `models/model.pkl` being present.

---

### Option 2 — Run the FastAPI Backend (Local)

Start the fraud detection REST API locally:

```bash
uvicorn api.app:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

You can also visit `http://127.0.0.1:8000/docs` in your browser for the **auto-generated Swagger UI** — FastAPI generates interactive API documentation automatically.

> 💡 **Or use the live deployed API:** [https://fraud-detection-ml-jp1y.onrender.com](https://fraud-detection-ml-jp1y.onrender.com)

---

### Option 3 — Run the Streamlit Web App (Local)

Run both the FastAPI backend and the Streamlit UI together (they need to run simultaneously in two separate terminals):

**Terminal 1 — Start the API:**
```bash
uvicorn api.app:app --reload
```

**Terminal 2 — Start the Streamlit app:**
```bash
streamlit run app/streamlit_app.py
```

The Streamlit UI will open in your browser at `http://localhost:8501`.

> 💡 **Or use the live deployed app:** [Streamlit Cloud](https://fraud-detection-ml-adhipatya3552.streamlit.app) — no local setup needed.

---

## 🌐 API Usage

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check — returns a running status message |
| `POST` | `/predict` | Accepts transaction features, returns fraud prediction + probability |

### POST `/predict`

**Request Body:**

```json
{
  "features": [0.1, -1.2, 0.5, 0.3, ..., 150.0]
}
```

The `features` list must contain exactly **29 float values** in this order:
- V1, V2, V3, ..., V28 (the 28 PCA-transformed features)
- Amount (the transaction amount, already scaled)

**Response:**

```json
{
  "fraud": 1,
  "probability": 0.87,
  "message": "High risk transaction"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `fraud` | Integer | `1` = fraud detected, `0` = normal transaction |
| `probability` | Float | Model's estimated probability that this is fraud (0.0 to 1.0) |
| `message` | String | Human-readable summary of the result |

**Example using `curl` (Live API):**

```bash
curl -X POST "https://fraud-detection-ml-jp1y.onrender.com/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [1.2, -0.5, 0.3, 0.8, -1.1, 0.6, 0.2, -0.4, 0.7, 0.1,
                       -0.3, 0.9, -0.2, 0.4, -0.8, 0.5, 0.1, -0.6, 0.3, 0.7,
                       -0.1, 0.2, -0.5, 0.4, 0.8, -0.3, 0.6, -0.7, 120.50]}'
```

---

## 💻 Streamlit Web App

> 🔗 **Live App:** [https://fraud-detection-ml-adhipatya3552.streamlit.app](https://fraud-detection-ml-adhipatya3552.streamlit.app)

The Streamlit app (`app/streamlit_app.py`) provides a visual interface for testing the fraud detection model without touching the API directly. It is deployed on **Streamlit Community Cloud** and connects to the Render-hosted FastAPI backend.

### What It Does

- Displays **29 number input fields** — one for each feature (V1–V28 + Amount)
- Offers **4 quick-action buttons** at the top:
  - 📥 **Normal Sample** — loads a real non-fraud transaction from the dataset into the fields (requires local CSV; disabled on cloud)
  - ⚠️ **Fraud Sample** — loads a real fraud transaction from the dataset into the fields (requires local CSV; disabled on cloud)
  - 🔄 **Reset** — resets all fields to 0.0
  - 🎲 **Random Values** — fills all fields with random values between -5 and 5
- When you click **🚀 Predict Fraud**, it sends the 29 values to the deployed FastAPI `/predict` endpoint on Render
- Displays the result as either a green success box (✅ NORMAL TRANSACTION) or a red error box (⚠️ FRAUD DETECTED)
- Shows the **confidence probability** alongside the verdict
- Renders a **progress bar** proportional to the fraud probability score

### How the Sample Loaders Work

The app attempts to read from `data/creditcard_2023.csv` using `@st.cache_data`. When running locally with the dataset present, clicking "Normal Sample" or "Fraud Sample" filters the dataset by `Class == 0` or `Class == 1`, picks a random row, strips the `id` and `Class` columns, and populates all 29 input fields with those real values.

When deployed on Streamlit Cloud (where the 325MB CSV is not included in the repo), the sample loader buttons are **gracefully disabled** — the app handles the missing file without crashing. You can still use **🎲 Random Values** or manually enter values to test predictions.

---

## ☁️ Cloud Deployment

This project is fully deployed on the cloud with two services:

### API — Render

The FastAPI backend is deployed on **[Render](https://render.com)** using Docker.

| Detail | Value |
|--------|-------|
| **Platform** | Render (Free Tier) |
| **Runtime** | Docker |
| **URL** | [https://fraud-detection-ml-jp1y.onrender.com](https://fraud-detection-ml-jp1y.onrender.com) |
| **API Docs** | [https://fraud-detection-ml-jp1y.onrender.com/docs](https://fraud-detection-ml-jp1y.onrender.com/docs) |
| **Auto-Deploy** | Yes — deploys automatically on every push to `main` |
| **Region** | Singapore |

> ⚠️ Render free tier services spin down after ~15 minutes of inactivity. The first request after idle may take ~30 seconds.

### Web App — Streamlit Cloud

The Streamlit frontend is deployed on **[Streamlit Community Cloud](https://streamlit.io/cloud)**.

| Detail | Value |
|--------|-------|
| **Platform** | Streamlit Community Cloud |
| **Main File** | `app/streamlit_app.py` |
| **URL** | [https://fraud-detection-ml-adhipatya3552.streamlit.app](https://fraud-detection-ml-adhipatya3552.streamlit.app) |
| **Auto-Deploy** | Yes — deploys automatically on every push to `main` |

---

## 🐳 Docker Deployment

The project includes a `Dockerfile` that packages the FastAPI API into a container. This is used by Render for production deployment and can also be used locally.

### Build the Image

```bash
docker build -t fraud-detection-api .
```

### Run the Container

```bash
docker run -p 10000:10000 fraud-detection-api
```

The API will be accessible at `http://localhost:10000`.

> ⚠️ **Important:** Before building the Docker image, make sure `models/model.pkl` exists (run `python main.py` first). The model file must be present inside the image because the container copies the entire project directory.

### Dockerfile Breakdown

```dockerfile
FROM python:3.10          # Uses the official Python 3.10 base image

WORKDIR /app              # Sets the working directory inside the container

COPY . .                  # Copies all project files into the container

RUN pip install -r requirements.txt   # Installs all dependencies

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "10000"]
# Starts the FastAPI server on port 10000 (Render's expected port)
```

---

## 📉 Model Evaluation & Results

After training, the following evaluation artifacts are saved to the `outputs/` folder:

### Confusion Matrix (`outputs/confusion_matrix.png`)

A heatmap showing:
- **True Negatives (top-left):** Legitimate transactions correctly classified as legitimate
- **False Positives (top-right):** Legitimate transactions incorrectly flagged as fraud
- **False Negatives (bottom-left):** Actual fraud missed by the model (most costly!)
- **True Positives (bottom-right):** Fraud cases correctly caught

### ROC Curve (`outputs/roc_curve.png`)

Shows the model's ability to distinguish between fraud and non-fraud across all decision thresholds. The **AUC (Area Under the Curve)** score is annotated — a score close to 1.0 indicates near-perfect separation.

### Classification Report (terminal output)

Printed after training with:
- **Precision** for each class
- **Recall** for each class
- **F1-score** for each class
- **Macro and weighted averages**

> In fraud detection, **recall for class 1 (fraud)** is especially important — missing a fraud case has a higher real-world cost than a false alarm.

---

## 📓 Exploratory Data Analysis

The notebook at `notebooks/eda.ipynb` provides a first look at the dataset before any modelling is done.

| Analysis | What It Shows |
|----------|---------------|
| `df.head()` | First 5 rows — feature names and sample values |
| `df.info()` | Column types, non-null counts, and memory usage |
| `df['Class'].value_counts()` | How many legitimate vs. fraudulent transactions exist |
| Count plot | Bar chart of class distribution — visually shows the imbalance |

To run the notebook:

```bash
jupyter notebook notebooks/eda.ipynb
```

> Install Jupyter if not already available: `pip install jupyter`

---

## 📦 Module Reference

### `main.py`

The pipeline entry point. Calls all four stages in order:

```python
load_data() → preprocess() → train_model() → evaluate_model()
```

```bash
python main.py
```

---

### `src/preprocessing.py`

**`load_data(path)`**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Path to the CSV dataset file |

Returns: A Pandas DataFrame.

**`preprocess(df)`**

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `DataFrame` | Raw loaded DataFrame |

Drops `id`, scales `Amount`, and returns `(X, y)`.

---

### `src/train.py`

**`train_model(X, y)`**

| Parameter | Type | Description |
|-----------|------|-------------|
| `X` | `DataFrame` | Feature matrix |
| `y` | `Series` | Target labels |

Applies SMOTE, splits 80/20, trains `RandomForestClassifier(n_estimators=50, n_jobs=-1)`, saves to `models/model.pkl`, and returns the trained model.

---

### `src/evaluate.py`

**`evaluate_model(model, X, y)`**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | Sklearn estimator | Trained Random Forest model |
| `X` | `DataFrame` | Full feature matrix |
| `y` | `Series` | Full target labels |

Performs a fresh 80/20 split (same `random_state=42`), prints classification report, and saves confusion matrix and ROC curve to `outputs/`.

---

### `src/predict.py`

Loads `models/model.pkl` at import time and exposes a single function:

**`predict(data)`**

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `list` | List of 29 feature values |

Returns: `numpy.ndarray` with the predicted class label.

---

### `api/app.py`

FastAPI application with:
- `GET /` — returns `{"message": "Fraud Detection API Running"}`
- `POST /predict` — accepts `{"features": [...]}`, returns `{"fraud": int, "probability": float, "message": str}`

---

## ⚠️ Known Limitations

| Issue | Details |
|-------|---------|
| **No feature scaling at inference time** | The `Amount` value sent to the API is expected to already be scaled — the API does not apply StandardScaler itself; the Streamlit app sends raw values |
| **SMOTE applied to full dataset** | SMOTE is run before the train/test split in `train.py`, which technically risks data leakage — ideally SMOTE should only be applied to training data |
| **Model reloaded on every API start** | The model is loaded at module level in `api/app.py` — if `model.pkl` doesn't exist, the API will crash on startup |
| **Static model** | The model does not update automatically as new fraud patterns emerge — retraining via `main.py` is needed manually |
| **No input validation for 29 features** | The API accepts any-length list; if the list is not exactly 29 values, `reshape(1, -1)` will not fail gracefully |
| **Streamlit and API must run separately (local)** | When running locally, both processes need separate terminals. On cloud, they are independent services |
| **No authentication on API** | The `/predict` endpoint has no API key or rate limiting |
| **Render cold starts** | Free tier services spin down after inactivity — first request after idle takes ~30 seconds |

---

## 🗺️ Roadmap

- [x] Data preprocessing pipeline (drop id, scale Amount)
- [x] SMOTE-based class imbalance handling
- [x] Random Forest model training with pickle serialization
- [x] Classification report + confusion matrix + ROC curve evaluation
- [x] FastAPI backend with structured Pydantic input validation
- [x] Streamlit web UI with real sample loaders and probability display
- [x] Docker support for API containerization
- [x] EDA notebook for dataset exploration
- [x] Deploy the FastAPI backend to Render (Docker-based cloud deployment)
- [x] Deploy the Streamlit app to Streamlit Community Cloud
- [ ] Apply StandardScaler inside the API at inference time (not just during training)
- [ ] Fix SMOTE to apply only on training data (after train/test split)
- [ ] Add input validation to ensure exactly 29 features are provided to the API
- [ ] Add feature importance visualization (which V-features matter most)
- [ ] Experiment with XGBoost or LightGBM for comparison
- [ ] Add cross-validation during training for more robust evaluation
- [ ] Add model retraining endpoint to the API
- [ ] Add API authentication with an API key header

---

<div align="center">

Built with ❤️ using Python, Scikit-Learn, FastAPI, Streamlit, and deployed on Render.

</div>
