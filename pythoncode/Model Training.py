import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc


# ======================================================
# 1. LOAD CLEANED DATA
# ======================================================
df = pd.read_csv("cleaned_heart_disease.csv")
print("Loaded cleaned CSV columns:", df.columns.tolist())


# ======================================================
# 2. IDENTIFY LABEL COLUMN
# ======================================================
label_col = "Heart Disease Status"
y = df[label_col]
X = df.drop(label_col, axis=1)


# ======================================================
# 3. ENCODE CATEGORICAL COLUMNS
# ======================================================
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print("Categorical columns detected:", categorical_cols)

for col in categorical_cols:
    print("Encoding column:", col)
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Encode label if it is still string
if y.dtype == "object":
    y = LabelEncoder().fit_transform(y.astype(str))


# ======================================================
# 4. APPLY SMOTE TO BALANCE DATA
# ======================================================
print("\nBefore SMOTE distribution:", np.bincount(y))

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

print("After SMOTE distribution:", np.bincount(y_resampled))


# ======================================================
# 5. TRAIN/TEST SPLIT
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.20, random_state=42
)


# ======================================================
# 6. BUILD ANN MODEL
# ======================================================
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# ======================================================
# 7. TRAIN MODEL
# ======================================================
history = model.fit(
    X_train, y_train,
    validation_split=0.20,
    epochs=80,
    batch_size=32,
    verbose=1
)


# ======================================================
# 8. EVALUATE MODEL
# ======================================================
pred_probs = model.predict(X_test)
pred_classes = (pred_probs > 0.5).astype(int)

acc = accuracy_score(y_test, pred_classes)
cm = confusion_matrix(y_test, pred_classes)
cr = classification_report(y_test, pred_classes)

print("\n===================================================")
print("                FINAL MODEL RESULTS")
print("===================================================")
print("Accuracy:", acc)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", cr)


# ======================================================
# 9. GENERATE ALL REQUIRED GRAPHS
# ======================================================

# --------------------------
# ACCURACY GRAPH
# --------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()


# --------------------------
# LOSS GRAPH
# --------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


# --------------------------
# CONFUSION MATRIX HEATMAP
# --------------------------
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# --------------------------
# ROC CURVE
# --------------------------
y_probs = model.predict(X_test).ravel()
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()