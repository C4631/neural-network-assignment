# ============================================================================
# FINAL ANN TRAINING SCRIPT — IMPUTATION + SMOTE + NORMALIZATION + AUTO ENCODING
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# --------------------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------------------

data = pd.read_csv("heart_disease.csv")
print("Loaded CSV columns:", data.columns.tolist())

# --------------------------------------------------------------
# 2. AUTO-DETECT LABEL COLUMN
# --------------------------------------------------------------

possible_labels = ["target", "output", "num", "diagnosis", "class", "heart disease status"]

label_col = None
for col in data.columns:
    if col.lower() in possible_labels:
        label_col = col
        break

if label_col is None:
    label_col = data.columns[-1]

print("Detected label column:", label_col)

# --------------------------------------------------------------
# 3. ENCODE STRING COLUMNS
# --------------------------------------------------------------

le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == "object":
        print(f"Encoding column: {col}")
        data[col] = le.fit_transform(data[col].astype(str))

# --------------------------------------------------------------
# 4. SPLIT FEATURES + LABEL
# --------------------------------------------------------------

X = data.drop(label_col, axis=1)
y = data[label_col]

# --------------------------------------------------------------
# 5. HANDLE MISSING VALUES (IMPUTATION)
# --------------------------------------------------------------

print("\nChecking missing values before imputation:")
print(X.isna().sum())

imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# --------------------------------------------------------------
# 6. SMOTE OVERSAMPLING
# --------------------------------------------------------------

print("\nBefore SMOTE distribution:", np.bincount(y))

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_imputed, y)

print("After SMOTE distribution:", np.bincount(y_resampled))

# --------------------------------------------------------------
# 7. NORMALIZATION
# --------------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resampled)

# --------------------------------------------------------------
# 8. TRAIN-TEST SPLIT
# --------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# --------------------------------------------------------------
# 9. MODEL
# --------------------------------------------------------------

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dropout(0.1),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# --------------------------------------------------------------
# 10. TRAINING
# --------------------------------------------------------------

es = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    callbacks=[es],
    verbose=1
)

# --------------------------------------------------------------
# 11. PLOTS
# --------------------------------------------------------------

plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label="Training Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# --------------------------------------------------------------
# 12. EVALUATION
# --------------------------------------------------------------

y_pred = (model.predict(X_test) > 0.5).astype("int32")

print("===================================================")
print("                  FINAL MODEL RESULTS")
print("===================================================")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
