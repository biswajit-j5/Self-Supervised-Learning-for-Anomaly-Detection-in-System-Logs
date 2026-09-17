# Self-Supervised-Learning-for-Anomaly-Detection-in-System-Logs-
🔎 Project Overview

System logs contain valuable information about the behavior and health of software and distributed systems. However, large volumes of logs make it difficult to manually identify unusual events.

This project demonstrates an automated approach to log anomaly detection. Instead of depending entirely on manually labeled anomaly data, it first uses HDBSCAN clustering to create pseudo-labels and then trains a GRU-based neural network using those labels.

Pipeline

Raw System Logs
       ↓
Data Loading & Preprocessing
       ↓
TF-IDF Feature Extraction
       ↓
FastICA Dimensionality Reduction
       ↓
HDBSCAN Clustering
       ↓
Pseudo Labels
       ↓
GRU Neural Network
       ↓
Anomaly Prediction
       ↓
Evaluation & Visualization

🎯 Problem Statement

Traditional anomaly detection systems often require labeled examples of normal and abnormal behavior. Creating high-quality labels for large-scale system logs can be expensive and time-consuming.

The goal of this project is to build a pipeline that can:

Process system log messages.

Convert text logs into numerical representations.

Discover natural patterns in the data.

Automatically generate pseudo-labels.

Train a deep-learning model using those labels.

Predict whether new log entries are potentially anomalous.

Provide basic visualizations and an anomaly score.

🎯 Objectives

Detect unusual patterns in system logs.

Reduce dependence on manually labeled training data.

Extract useful features from textual log messages.

Reduce feature dimensionality before clustering.

Generate pseudo-labels automatically.

Use a GRU neural network for binary anomaly classification.

Evaluate the model using standard classification metrics.

Demonstrate anomaly detection on new log entries.

✨ Key Features

1. Text Feature Extraction

Uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert log messages into numerical feature vectors.

2. Dimensionality Reduction

Uses FastICA (Fast Independent Component Analysis) to reduce the feature space from the TF-IDF representation to a smaller set of components.

3. Unsupervised Clustering

Uses HDBSCAN to identify clusters and noise points.

In the current implementation:

HDBSCAN cluster = -1  →  Anomaly
Other clusters       →  Normal

This creates pseudo-labels without requiring manually assigned anomaly labels.

4. GRU-Based Classification

A Gated Recurrent Unit (GRU) neural network is trained using the generated pseudo-labels.

5. Model Evaluation

The notebook evaluates predictions using:

Accuracy

Precision

Recall

F1-score

Classification report

Confusion matrix

Training/validation loss

6. New Log Prediction

The trained pipeline can transform new log messages and predict an anomaly score.

7. Basic Explainability

For an anomalous log, the notebook displays high-scoring TF-IDF terms as a simple indication of words contributing to the representation.

8. Authentication Demonstration

The notebook contains a simulated authentication section followed by anomaly checking. This is a demonstration only and is not intended to represent production-grade authentication.

🛠 Technology Stack

Technology

Purpose

Python

Main programming language

Pandas

Data loading and manipulation

NumPy

Numerical operations

Matplotlib

Data visualization

Seaborn

Statistical visualization

Scikit-learn

TF-IDF, FastICA, train/test split and evaluation metrics

HDBSCAN

Density-based clustering and pseudo-label generation

TensorFlow / Keras

GRU deep-learning model

Jupyter Notebook

Development and experimentation

Google Colab

Optional cloud environment for running the notebook

HTML/CSS/JavaScript

Demonstration login UI inside the notebook

🧠 Machine Learning Techniques

TF-IDF

TF-IDF converts text into numerical features based on how important words are within the collection of log messages.

The project currently uses:

TfidfVectorizer(max_features=500)

Therefore, at most 500 text features are retained.

FastICA

FastICA is used for dimensionality reduction and feature transformation.

The project uses:

FastICA(n_components=20, random_state=42)

The TF-IDF representation is transformed into 20 independent components.

HDBSCAN

HDBSCAN is a density-based clustering algorithm that can identify clusters as well as noise.

The project uses:

HDBSCAN(min_cluster_size=15)

The current pseudo-labeling rule is:

labels = np.where(clusters == -1, 1, 0)

Where:

1 = Anomaly
0 = Normal

Important: These are pseudo-labels created by the clustering procedure, not manually verified ground-truth labels.

GRU Neural Network

The GRU model is built using Keras:

GRU(64)
   ↓
Dropout(0.2)
   ↓
Dense(32, ReLU)
   ↓
Dense(1, Sigmoid)

The model is compiled with:

Optimizer: Adam
Loss: Binary Cross-Entropy
Metric: Accuracy

The current training configuration is:

Epochs: 5
Batch size: 64
Validation/Test split: 20%

📊 Dataset

The project includes a structured CSV file based on the HDFS log dataset.

The included CSV contains approximately 104,815 log records and 9 columns.

Dataset Columns

Column

Description

LineId

Identifier for the log line

Date

Date information

Time

Time information

Pid

Process ID

Level

Log level such as INFO

Component

System component that generated the log

Content

Actual log message

EventId

Identifier for the log event

EventTemplate

Generalized event template

The notebook primarily uses:

df["Content"]

as the textual input for anomaly detection.

📁 Project Structure

Self-Supervised-Learning-for-Anomaly-Detection-in-System-Logs/
│
├── HDFS_100k.log_structured.csv
│
├── system_logs_anomaly_detection.ipynb
│
├── README.md
│
└── .gitignore              # Recommended addition

File Descriptions

HDFS_100k.log_structured.csv
Contains the structured system log dataset used by the notebook.

system_logs_anomaly_detection.ipynb
Contains the complete implementation, including preprocessing, feature extraction, clustering, GRU training, evaluation, anomaly prediction, authentication demonstration, and UI.

README.md
Project documentation and setup instructions.

📦 Requirements

Install the required Python packages:

pip install pandas numpy matplotlib seaborn scikit-learn hdbscan tensorflow jupyter

Or create a requirements.txt file containing:

pandas
numpy
matplotlib
seaborn
scikit-learn
hdbscan
tensorflow
jupyter

Then install them with:

pip install -r requirements.txt

⚙️ Installation

Step 1: Clone the repository

git clone <YOUR_GITHUB_REPOSITORY_URL>

Step 2: Open the project folder

cd Self-Supervised-Learning-for-Anomaly-Detection-in-System-Logs

Step 3: Install dependencies

pip install -r requirements.txt

Step 4: Start Jupyter Notebook

jupyter notebook

Step 5: Open

system_logs_anomaly_detection.ipynb

Run the notebook cells from top to bottom.

☁️ Running on Google Colab

The notebook can also be run using Google Colab.

Open Google Colab.

Upload system_logs_anomaly_detection.ipynb.

Upload HDFS_100k.log_structured.csv.

Install missing packages if necessary:

!pip install hdbscan

Run the notebook cells sequentially.

🔄 How the System Works

Step 1 — Load the Dataset

The CSV file is loaded using Pandas:

df = pd.read_csv("HDFS_100k.log_structured.csv")
logs = df["Content"].astype(str)

Step 2 — Convert Logs to TF-IDF

vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(logs).toarray()

This converts textual log messages into numerical vectors.

Step 3 — Reduce Dimensions

ica = FastICA(n_components=20, random_state=42)
X_reduced = ica.fit_transform(X)

The feature representation is reduced to 20 components.

Step 4 — Generate Pseudo Labels

clusterer = hdbscan.HDBSCAN(min_cluster_size=15)
clusters = clusterer.fit_predict(X_reduced)

labels = np.where(clusters == -1, 1, 0)

HDBSCAN noise points are treated as potential anomalies.

Step 5 — Prepare Data for GRU

The reduced features are reshaped into the 3D structure expected by a GRU:

(samples, timesteps, features)

The current implementation uses one timestep:

X_seq = X_reduced.reshape(
    (X_reduced.shape[0], 1, X_reduced.shape[1])
)

The data is then divided into training and testing sets.

Step 6 — Train the GRU

The neural network learns to classify the pseudo-labels generated by HDBSCAN.

Input
 ↓
GRU - 64 units
 ↓
Dropout - 20%
 ↓
Dense - 32 units
 ↓
Sigmoid Output

Step 7 — Predict Anomalies

The model outputs a value between 0 and 1.

The notebook currently uses a threshold of:

> 0.5 → Anomaly
≤ 0.5 → Normal

📈 Model Evaluation

The notebook generates several evaluation outputs.

Classification Report

Reports:

Precision

Recall

F1-score

Support

Confusion Matrix

Shows:

True Negative
False Positive
False Negative
True Positive

Training Loss

The notebook compares training loss and validation loss to visualize the training process.

Accuracy

The notebook also reports:

Training accuracy

Validation accuracy

Test accuracy

Note: Because the targets are generated by HDBSCAN rather than manually verified ground truth, these metrics primarily measure how well the GRU reproduces the clustering-derived pseudo-labels. They should not automatically be interpreted as real-world anomaly-detection accuracy.

🧪 Online Learning Simulation

The notebook includes a basic simulation of processing new logs:

new_logs = logs.sample(100)

The new logs go through the same pipeline:

New Logs
   ↓
TF-IDF Transform
   ↓
FastICA Transform
   ↓
GRU Prediction
   ↓
Anomaly Count

This demonstrates how an already-trained pipeline could process incoming log messages.

This is a simulation, not a true continuously retrained online-learning system.

🔐 Authentication and UI

The notebook contains a demonstration authentication step and an HTML/CSS/JavaScript login interface.

The purpose is to demonstrate how an anomaly detection workflow could be connected to a user-facing interface.

Important Security Note

The authentication implementation is not production-ready.

It contains hard-coded demonstration credentials and a simulated IP address. Real applications should use:

Secure password hashing

Environment variables or a secrets manager

HTTPS

Proper session management

Database-backed authentication

Access control

Secure logging

Do not use the demonstration credentials or authentication code as a real security system.

⚠️ Important Notes

1. Pseudo-labels are not ground truth

The project defines HDBSCAN noise points as anomalies. A noise point is not necessarily a real-world anomaly.

Therefore:

HDBSCAN noise ≠ guaranteed real anomaly

The generated labels should be treated as automatically generated training targets.

2. The current GRU setup has one timestep

The notebook reshapes every sample into:

1 timestep × 20 features

This means the current implementation does not model long sequences of consecutive log events in the way a typical sequence-based GRU system would.

A future version could group logs into windows or sessions so that the GRU receives multiple timesteps.

3. The dataset is already structured

The project uses the structured CSV version of the HDFS logs rather than parsing completely raw log files from scratch.

4. Authentication is only a demonstration

The login functionality should not be used for real authentication.

🚧 Limitations

Pseudo-labels are generated from clustering and may contain incorrect labels.

HDBSCAN's noise class is assumed to represent anomalies.

The current GRU uses only one timestep per sample.

The model is trained for only 5 epochs in the notebook.

No hyperparameter optimization is included.

The current online-learning section does not retrain the model continuously.

The authentication system is simulated.

Real-time log ingestion is not implemented.

Production deployment and monitoring are not included.

The project does not establish that every detected anomaly is a confirmed security incident.

🚀 Future Enhancements

Possible improvements include:

Machine Learning

Use manually verified ground-truth labels for evaluation.

Tune HDBSCAN parameters automatically.

Compare HDBSCAN with DBSCAN, Isolation Forest, One-Class SVM, and autoencoders.

Perform hyperparameter tuning for the GRU.

Add early stopping and model checkpointing.

Sequence Modeling

Instead of one timestep, create sequences such as:

Log 1 → Log 2 → Log 3 → Log 4 → Log 5

and feed these sequences into the GRU.

This would allow the model to learn temporal relationships between log events.

Real-Time Detection

Build a pipeline such as:

Application/System
       ↓
Log Stream
       ↓
Log Collector
       ↓
Feature Extraction
       ↓
Trained Model
       ↓
Anomaly Alert
       ↓
Dashboard

Deployment

The model could be exposed through:

Flask

FastAPI

Streamlit

and deployed using:

Docker

Cloud platforms

Kubernetes

Explainability

Add stronger explainability techniques such as:

SHAP

LIME

Feature importance analysis

Event-template analysis

Security

Replace the simulated authentication with a proper secure authentication system.

💡 Possible Use Cases

The approach can be adapted for:

Distributed-system monitoring

Server log monitoring

Application monitoring

Infrastructure monitoring

Error detection

Operational anomaly detection

Cybersecurity log analysis

Early identification of unusual system behavior

🏗️ Overall Architecture

                 ┌─────────────────────┐
                 │   HDFS Log Dataset  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Data Preprocessing  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │      TF-IDF         │
                 │ Feature Extraction  │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │      FastICA        │
                 │ Dimensionality      │
                 │ Reduction           │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │      HDBSCAN        │
                 │    Clustering       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │  Pseudo Labels      │
                 │ Normal / Anomaly    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │       GRU           │
                 │ Deep Learning Model │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Anomaly Prediction  │
                 └──────────┬──────────┘
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
          ┌──────────────┐    ┌───────────────┐
          │ Evaluation   │    │ New Log Check │
          └──────────────┘    └───────────────┘

📌 Results

The notebook automatically generates the model's metrics when it is executed.

Because results depend on the runtime environment, library versions, model initialization, and data processing, the README intentionally does not claim fixed accuracy values.

After running the notebook, you can add your verified results here:

Test Accuracy: XX.XX%
Precision: XX.XX%
Recall: XX.XX%
F1-Score: XX.XX%

🤝 Contribution

Contributions are welcome.

A typical workflow is:

git checkout -b feature/new-improvement

Make your changes, test them, and create a pull request.

Possible contribution areas:

Better sequence construction

Improved anomaly labeling

New ML models

Real-time log ingestion

Dashboard development

Explainability

Secure authentication

Deployment support
