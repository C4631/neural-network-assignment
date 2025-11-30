# ===================================================================
# FULL ANN TRAINING SCRIPT — AUTO-DETECT LABEL + AUTO ENCODE STRINGS
# ===================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

# fallback: use last column
if label_col is None:
    label_col = data.columns[-1]

print("Detected label column:", label_col)

# --------------------------------------------------------------
# 3. AUTO-ENCODE ALL STRING COLUMNS
# --------------------------------------------------------------

le = LabelEncoder()

for col in data.columns:
    if data[col].dtype == "object":     # if column contains text
        print(f"Encoding column: {col}")
        data[col] = le.fit_transform(data[col].astype(str))

# --------------------------------------------------------------
# 4. SPLIT FEATURES + LABEL
# --------------------------------------------------------------

X = data.drop(label_col, axis=1)
y = data[label_col]

# --------------------------------------------------------------
# 5. TRAIN-TEST SPLIT
# --------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------------------------------
# 6. BUILD MODEL
# --------------------------------------------------------------

model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# --------------------------------------------------------------
# 7. COMPILE MODEL
# --------------------------------------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# --------------------------------------------------------------
# 8. TRAIN MODEL
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
# 9. PLOT TRAINING CURVES
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
# 10. EVALUATE MODEL
# --------------------------------------------------------------

y_pred = (model.predict(X_test) > 0.5).astype("int32")

print("===================================================")
print("               FINAL MODEL RESULTS")
print("===================================================")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
