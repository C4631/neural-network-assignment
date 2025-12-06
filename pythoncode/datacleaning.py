# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 22:52:09 2025

@author: user
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

standardCol = ['Gender','Exercise Habits','Smoking','Family Heart Disease','Diabetes','High Blood Pressure',
               'Low HDL Cholesterol', 'High LDL Cholesterol','Alcohol Consumption','Stress Level',
               'Sugar Consumption','Heart Disease Status']
numScaleCol = ['Age','Blood Pressure','Cholesterol Level','BMI','Sleep Hours','Triglyceride Level',
               'Fasting Blood Sugar','CRP Level','Homocysteine Level']

df = pd.read_csv("heart_disease.csv")

df = df.drop_duplicates()
df = df.dropna()

for col in standardCol:
    df[col] = df[col].astype(str).str.strip().str.lower()

for col in numScaleCol:
    mean = df[col].mean()
    std = df[col].std()
    df = df[(df[col] > mean - 3*std) & (df[col] < mean + 3*std)]

scaler = StandardScaler()
df[numScaleCol] = scaler.fit_transform(df[numScaleCol])

df.to_csv("cleaned_heart_disease.csv", index=False)