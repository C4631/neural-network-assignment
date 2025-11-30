# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 14:42:15 2025

@author: Josh
"""

# ==============================================================
# FULL ANN MODEL TRAINING SCRIPT (Person C)
# ==============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# --------------------------------------------------------------
# 1. LOAD CLEANED DATA  (Ensure Person A gives you this CSV)
# --------------------------------------------------------------

data = pd.read_csv("heart_disease_cleaned.csv")   # change name if needed

# Separate features & label
X = data.drop("target", axis=1)
y = data["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------------------------------------------
# 2. BUILD MODEL (Person B Architecture — adjust layers if needed)
# --------------------------------------------------------------

model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# --------------------------------------------------------------
# 3. COMPILE MODEL
# --------------------------------------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# --------------------------------------------------------------
# 4. TRAINING (With Early Stopping)
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
# 5. PLOT TRAINING CURVES
# --------------------------------------------------------------

# Accuracy curve
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label="Training Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Loss curve
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# --------------------------------------------------------------
# 6. MODEL EVALUATION
# --------------------------------------------------------------

y_pred = (model.predict(X_test) > 0.5).astype("int32")

print("====================================")
print("        MODEL EVALUATION")
print("====================================")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

