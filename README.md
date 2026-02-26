# Diabetes Risk Prediction - MLP

### [Diabetes Risk Prediction Dataset](https://www.kaggle.com/datasets/vishardmehta/diabetes-risk-prediction-dataset)

Author: [Kevin Thomas](mailto:ket189@pitt.edu)

License: MIT

## Install Libraries


```python
# !python -m pip install --upgrade pip
# %pip install ipywidgets
# %pip install pandas matplotlib seaborn
# %pip install scikit-learn
# # %pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124  # Windows with CUDA 12.4
# %pip install torch  # MacOS or CPU-only
# %pip install shap
# %pip install black
# %pip install black[jupyter]
# %pip install nbqa
# %pip install scipy
```

## Import Libraries


```python
import warnings
import sqlite3
from itertools import combinations
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    fbeta_score,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", message=".*global RNG.*", category=FutureWarning)
```

## Seed


```python
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
```




    <torch._C.Generator at 0x119287030>



## Device


```python
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
DEVICE
```




    device(type='mps')



## Parameters


```python
H1 = 32
H2 = 16
OUT_FEATURES = 2
TEST_SIZE = 0.3
DATA_PATH = "diabetes_risk_dataset.csv"
TARGET_VAR = "diabetes_risk_score"
CLASS_NAMES = ["Low Risk", "High Risk"]
CLASS_MAP = {0: "Low Risk", 1: "High Risk"}
```

## Hyperparameters


```python
LR = 0.001
EPOCHS = 1000
LOG_INTERVAL = 10
BATCH_SIZE = 64
CHUNK_SIZE = None  # Set to int (e.g., 100000) for large files, None for small files
DROPOUT = 0.3  # Dropout rate (0.0 = no dropout, 0.2-0.5 typical for regularization)
VAL_SIZE = 0.15  # Fraction of training data reserved for validation
PATIENCE = 50  # Early stopping patience (epochs without improvement)
```

## Load Dataset


```python
if CHUNK_SIZE is None:
    df = pd.read_csv(DATA_PATH)
else:
    chunks = []
    for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    del chunks  # Free memory
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Patient_ID</th>
      <th>age</th>
      <th>gender</th>
      <th>bmi</th>
      <th>blood_pressure</th>
      <th>fasting_glucose_level</th>
      <th>insulin_level</th>
      <th>HbA1c_level</th>
      <th>cholesterol_level</th>
      <th>triglycerides_level</th>
      <th>physical_activity_level</th>
      <th>daily_calorie_intake</th>
      <th>sugar_intake_grams_per_day</th>
      <th>sleep_hours</th>
      <th>stress_level</th>
      <th>family_history_diabetes</th>
      <th>waist_circumference_cm</th>
      <th>diabetes_risk_score</th>
      <th>diabetes_risk_category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>77</td>
      <td>Female</td>
      <td>33.8</td>
      <td>154</td>
      <td>93</td>
      <td>12.1</td>
      <td>5.2</td>
      <td>242</td>
      <td>194</td>
      <td>Low</td>
      <td>2169</td>
      <td>78.4</td>
      <td>8.1</td>
      <td>4</td>
      <td>No</td>
      <td>101.1</td>
      <td>52.3</td>
      <td>Prediabetes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>54</td>
      <td>Male</td>
      <td>19.2</td>
      <td>123</td>
      <td>94</td>
      <td>4.6</td>
      <td>5.4</td>
      <td>212</td>
      <td>76</td>
      <td>High</td>
      <td>1881</td>
      <td>16.5</td>
      <td>6.6</td>
      <td>3</td>
      <td>No</td>
      <td>60.0</td>
      <td>3.7</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>25</td>
      <td>Male</td>
      <td>33.7</td>
      <td>141</td>
      <td>150</td>
      <td>10.8</td>
      <td>6.9</td>
      <td>247</td>
      <td>221</td>
      <td>Low</td>
      <td>2811</td>
      <td>147.9</td>
      <td>6.7</td>
      <td>10</td>
      <td>Yes</td>
      <td>114.7</td>
      <td>87.3</td>
      <td>High Risk</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>23</td>
      <td>Female</td>
      <td>32.8</td>
      <td>140</td>
      <td>145</td>
      <td>11.6</td>
      <td>6.8</td>
      <td>195</td>
      <td>193</td>
      <td>Low</td>
      <td>2826</td>
      <td>98.3</td>
      <td>4.4</td>
      <td>9</td>
      <td>Yes</td>
      <td>96.6</td>
      <td>76.1</td>
      <td>High Risk</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>70</td>
      <td>Male</td>
      <td>33.7</td>
      <td>165</td>
      <td>90</td>
      <td>18.3</td>
      <td>5.6</td>
      <td>217</td>
      <td>170</td>
      <td>Moderate</td>
      <td>2610</td>
      <td>65.8</td>
      <td>9.1</td>
      <td>5</td>
      <td>Yes</td>
      <td>107.4</td>
      <td>47.7</td>
      <td>Prediabetes</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5995</th>
      <td>5996</td>
      <td>58</td>
      <td>Male</td>
      <td>21.8</td>
      <td>158</td>
      <td>89</td>
      <td>6.3</td>
      <td>5.3</td>
      <td>198</td>
      <td>132</td>
      <td>High</td>
      <td>1995</td>
      <td>44.1</td>
      <td>7.6</td>
      <td>7</td>
      <td>No</td>
      <td>82.7</td>
      <td>0.0</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>5996</th>
      <td>5997</td>
      <td>73</td>
      <td>Male</td>
      <td>47.3</td>
      <td>193</td>
      <td>109</td>
      <td>33.7</td>
      <td>6.1</td>
      <td>250</td>
      <td>209</td>
      <td>Low</td>
      <td>3401</td>
      <td>61.4</td>
      <td>5.3</td>
      <td>10</td>
      <td>Yes</td>
      <td>150.0</td>
      <td>100.0</td>
      <td>High Risk</td>
    </tr>
    <tr>
      <th>5997</th>
      <td>5998</td>
      <td>35</td>
      <td>Male</td>
      <td>31.0</td>
      <td>139</td>
      <td>90</td>
      <td>15.1</td>
      <td>5.3</td>
      <td>190</td>
      <td>164</td>
      <td>Moderate</td>
      <td>3022</td>
      <td>86.7</td>
      <td>6.8</td>
      <td>3</td>
      <td>Yes</td>
      <td>102.7</td>
      <td>22.3</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>5998</th>
      <td>5999</td>
      <td>58</td>
      <td>Male</td>
      <td>26.2</td>
      <td>136</td>
      <td>88</td>
      <td>6.3</td>
      <td>5.5</td>
      <td>223</td>
      <td>126</td>
      <td>High</td>
      <td>2311</td>
      <td>28.2</td>
      <td>8.4</td>
      <td>5</td>
      <td>No</td>
      <td>90.6</td>
      <td>3.9</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>5999</th>
      <td>6000</td>
      <td>54</td>
      <td>Male</td>
      <td>39.3</td>
      <td>158</td>
      <td>201</td>
      <td>10.7</td>
      <td>8.3</td>
      <td>240</td>
      <td>223</td>
      <td>Low</td>
      <td>2442</td>
      <td>113.1</td>
      <td>5.7</td>
      <td>6</td>
      <td>No</td>
      <td>123.8</td>
      <td>100.0</td>
      <td>High Risk</td>
    </tr>
  </tbody>
</table>
<p>6000 rows × 19 columns</p>
</div>




```python
df.info()
```

    <class 'pandas.DataFrame'>
    RangeIndex: 6000 entries, 0 to 5999
    Data columns (total 19 columns):
     #   Column                      Non-Null Count  Dtype  
    ---  ------                      --------------  -----  
     0   Patient_ID                  6000 non-null   int64  
     1   age                         6000 non-null   int64  
     2   gender                      6000 non-null   str    
     3   bmi                         6000 non-null   float64
     4   blood_pressure              6000 non-null   int64  
     5   fasting_glucose_level       6000 non-null   int64  
     6   insulin_level               6000 non-null   float64
     7   HbA1c_level                 6000 non-null   float64
     8   cholesterol_level           6000 non-null   int64  
     9   triglycerides_level         6000 non-null   int64  
     10  physical_activity_level     6000 non-null   str    
     11  daily_calorie_intake        6000 non-null   int64  
     12  sugar_intake_grams_per_day  6000 non-null   float64
     13  sleep_hours                 6000 non-null   float64
     14  stress_level                6000 non-null   int64  
     15  family_history_diabetes     6000 non-null   str    
     16  waist_circumference_cm      6000 non-null   float64
     17  diabetes_risk_score         6000 non-null   float64
     18  diabetes_risk_category      6000 non-null   str    
    dtypes: float64(7), int64(8), str(4)
    memory usage: 890.8 KB


## Clean Dataset

### Complete Missingness Analysis


```python
missing_df = pd.DataFrame(
    {
        "Variable": df.columns,
        "Missing_Count": df.isna().sum(),
        "Missing_Pct": (df.isna().sum() / len(df) * 100).round(1),
    }
).sort_values("Missing_Pct", ascending=False)
missing_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Variable</th>
      <th>Missing_Count</th>
      <th>Missing_Pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Patient_ID</th>
      <td>Patient_ID</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>physical_activity_level</th>
      <td>physical_activity_level</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>diabetes_risk_score</th>
      <td>diabetes_risk_score</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>waist_circumference_cm</th>
      <td>waist_circumference_cm</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>family_history_diabetes</th>
      <td>family_history_diabetes</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>stress_level</th>
      <td>stress_level</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>sleep_hours</th>
      <td>sleep_hours</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>sugar_intake_grams_per_day</th>
      <td>sugar_intake_grams_per_day</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>daily_calorie_intake</th>
      <td>daily_calorie_intake</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>triglycerides_level</th>
      <td>triglycerides_level</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>age</th>
      <td>age</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>cholesterol_level</th>
      <td>cholesterol_level</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>HbA1c_level</th>
      <td>HbA1c_level</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>insulin_level</th>
      <td>insulin_level</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>fasting_glucose_level</th>
      <td>fasting_glucose_level</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>blood_pressure</th>
      <td>blood_pressure</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>bmi</th>
      <td>bmi</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>gender</th>
      <td>gender</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>diabetes_risk_category</th>
      <td>diabetes_risk_category</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



### Create Dataset `df_clean` w/ Cleaned Data


```python
df_clean = df.copy()
df_clean
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Patient_ID</th>
      <th>age</th>
      <th>gender</th>
      <th>bmi</th>
      <th>blood_pressure</th>
      <th>fasting_glucose_level</th>
      <th>insulin_level</th>
      <th>HbA1c_level</th>
      <th>cholesterol_level</th>
      <th>triglycerides_level</th>
      <th>physical_activity_level</th>
      <th>daily_calorie_intake</th>
      <th>sugar_intake_grams_per_day</th>
      <th>sleep_hours</th>
      <th>stress_level</th>
      <th>family_history_diabetes</th>
      <th>waist_circumference_cm</th>
      <th>diabetes_risk_score</th>
      <th>diabetes_risk_category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>77</td>
      <td>Female</td>
      <td>33.8</td>
      <td>154</td>
      <td>93</td>
      <td>12.1</td>
      <td>5.2</td>
      <td>242</td>
      <td>194</td>
      <td>Low</td>
      <td>2169</td>
      <td>78.4</td>
      <td>8.1</td>
      <td>4</td>
      <td>No</td>
      <td>101.1</td>
      <td>52.3</td>
      <td>Prediabetes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>54</td>
      <td>Male</td>
      <td>19.2</td>
      <td>123</td>
      <td>94</td>
      <td>4.6</td>
      <td>5.4</td>
      <td>212</td>
      <td>76</td>
      <td>High</td>
      <td>1881</td>
      <td>16.5</td>
      <td>6.6</td>
      <td>3</td>
      <td>No</td>
      <td>60.0</td>
      <td>3.7</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>25</td>
      <td>Male</td>
      <td>33.7</td>
      <td>141</td>
      <td>150</td>
      <td>10.8</td>
      <td>6.9</td>
      <td>247</td>
      <td>221</td>
      <td>Low</td>
      <td>2811</td>
      <td>147.9</td>
      <td>6.7</td>
      <td>10</td>
      <td>Yes</td>
      <td>114.7</td>
      <td>87.3</td>
      <td>High Risk</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>23</td>
      <td>Female</td>
      <td>32.8</td>
      <td>140</td>
      <td>145</td>
      <td>11.6</td>
      <td>6.8</td>
      <td>195</td>
      <td>193</td>
      <td>Low</td>
      <td>2826</td>
      <td>98.3</td>
      <td>4.4</td>
      <td>9</td>
      <td>Yes</td>
      <td>96.6</td>
      <td>76.1</td>
      <td>High Risk</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>70</td>
      <td>Male</td>
      <td>33.7</td>
      <td>165</td>
      <td>90</td>
      <td>18.3</td>
      <td>5.6</td>
      <td>217</td>
      <td>170</td>
      <td>Moderate</td>
      <td>2610</td>
      <td>65.8</td>
      <td>9.1</td>
      <td>5</td>
      <td>Yes</td>
      <td>107.4</td>
      <td>47.7</td>
      <td>Prediabetes</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5995</th>
      <td>5996</td>
      <td>58</td>
      <td>Male</td>
      <td>21.8</td>
      <td>158</td>
      <td>89</td>
      <td>6.3</td>
      <td>5.3</td>
      <td>198</td>
      <td>132</td>
      <td>High</td>
      <td>1995</td>
      <td>44.1</td>
      <td>7.6</td>
      <td>7</td>
      <td>No</td>
      <td>82.7</td>
      <td>0.0</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>5996</th>
      <td>5997</td>
      <td>73</td>
      <td>Male</td>
      <td>47.3</td>
      <td>193</td>
      <td>109</td>
      <td>33.7</td>
      <td>6.1</td>
      <td>250</td>
      <td>209</td>
      <td>Low</td>
      <td>3401</td>
      <td>61.4</td>
      <td>5.3</td>
      <td>10</td>
      <td>Yes</td>
      <td>150.0</td>
      <td>100.0</td>
      <td>High Risk</td>
    </tr>
    <tr>
      <th>5997</th>
      <td>5998</td>
      <td>35</td>
      <td>Male</td>
      <td>31.0</td>
      <td>139</td>
      <td>90</td>
      <td>15.1</td>
      <td>5.3</td>
      <td>190</td>
      <td>164</td>
      <td>Moderate</td>
      <td>3022</td>
      <td>86.7</td>
      <td>6.8</td>
      <td>3</td>
      <td>Yes</td>
      <td>102.7</td>
      <td>22.3</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>5998</th>
      <td>5999</td>
      <td>58</td>
      <td>Male</td>
      <td>26.2</td>
      <td>136</td>
      <td>88</td>
      <td>6.3</td>
      <td>5.5</td>
      <td>223</td>
      <td>126</td>
      <td>High</td>
      <td>2311</td>
      <td>28.2</td>
      <td>8.4</td>
      <td>5</td>
      <td>No</td>
      <td>90.6</td>
      <td>3.9</td>
      <td>Low Risk</td>
    </tr>
    <tr>
      <th>5999</th>
      <td>6000</td>
      <td>54</td>
      <td>Male</td>
      <td>39.3</td>
      <td>158</td>
      <td>201</td>
      <td>10.7</td>
      <td>8.3</td>
      <td>240</td>
      <td>223</td>
      <td>Low</td>
      <td>2442</td>
      <td>113.1</td>
      <td>5.7</td>
      <td>6</td>
      <td>No</td>
      <td>123.8</td>
      <td>100.0</td>
      <td>High Risk</td>
    </tr>
  </tbody>
</table>
<p>6000 rows × 19 columns</p>
</div>



### Drop Variables in `df_clean` w/ Missings


```python
# df_clean.drop(columns=[
#     "",
#     ""],
#     inplace=True)
```

### Drop Variables w/ No Predictive Value


```python
df_clean.drop(columns=["Patient_ID"], inplace=True, errors="ignore")
```

### Remove Leakage Features


```python
df_clean.drop(columns=["diabetes_risk_category"], inplace=True, errors="ignore")
```

### Binarize Target Variable

- `diabetes_risk_score = 1`: High Risk (> median)
- `diabetes_risk_score = 0`: Low Risk (<= median)


```python
df_clean["diabetes_risk_score"] = (
    df_clean["diabetes_risk_score"] > df_clean["diabetes_risk_score"].median()
).astype(int)
df_clean["diabetes_risk_score"].value_counts()
```




    diabetes_risk_score
    0    3001
    1    2999
    Name: count, dtype: int64



## Feature Engineering

### Classify `bmi_category` Based on BMI Thresholds


```python
df_clean["bmi_category"] = pd.cut(
    df_clean["bmi"],
    bins=[0, 18.5, 25, 30, np.inf],
    labels=["underweight", "normal", "overweight", "obese"],
)
df_clean["bmi_category"]
```




    0            obese
    1           normal
    2            obese
    3            obese
    4            obese
               ...    
    5995        normal
    5996         obese
    5997         obese
    5998    overweight
    5999         obese
    Name: bmi_category, Length: 6000, dtype: category
    Categories (4, str): ['underweight' < 'normal' < 'overweight' < 'obese']



### Classify `glucose_category` Based on Fasting Glucose Levels


```python
df_clean["glucose_category"] = pd.cut(
    df_clean["fasting_glucose_level"],
    bins=[0, 100, 126, np.inf],
    labels=["normal", "prediabetic", "diabetic"],
)
df_clean["glucose_category"]
```




    0            normal
    1            normal
    2          diabetic
    3          diabetic
    4            normal
               ...     
    5995         normal
    5996    prediabetic
    5997         normal
    5998         normal
    5999       diabetic
    Name: glucose_category, Length: 6000, dtype: category
    Categories (3, str): ['normal' < 'prediabetic' < 'diabetic']



## Save Cleaned Dataset


```python
df_clean.to_csv("drp_dataset_clean.csv", index=False)
```

## Load Cleaned Dataset


```python
df_eda = pd.read_csv("drp_dataset_clean.csv")
df_eda
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>gender</th>
      <th>bmi</th>
      <th>blood_pressure</th>
      <th>fasting_glucose_level</th>
      <th>insulin_level</th>
      <th>HbA1c_level</th>
      <th>cholesterol_level</th>
      <th>triglycerides_level</th>
      <th>physical_activity_level</th>
      <th>daily_calorie_intake</th>
      <th>sugar_intake_grams_per_day</th>
      <th>sleep_hours</th>
      <th>stress_level</th>
      <th>family_history_diabetes</th>
      <th>waist_circumference_cm</th>
      <th>diabetes_risk_score</th>
      <th>bmi_category</th>
      <th>glucose_category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>77</td>
      <td>Female</td>
      <td>33.8</td>
      <td>154</td>
      <td>93</td>
      <td>12.1</td>
      <td>5.2</td>
      <td>242</td>
      <td>194</td>
      <td>Low</td>
      <td>2169</td>
      <td>78.4</td>
      <td>8.1</td>
      <td>4</td>
      <td>No</td>
      <td>101.1</td>
      <td>1</td>
      <td>obese</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>54</td>
      <td>Male</td>
      <td>19.2</td>
      <td>123</td>
      <td>94</td>
      <td>4.6</td>
      <td>5.4</td>
      <td>212</td>
      <td>76</td>
      <td>High</td>
      <td>1881</td>
      <td>16.5</td>
      <td>6.6</td>
      <td>3</td>
      <td>No</td>
      <td>60.0</td>
      <td>0</td>
      <td>normal</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25</td>
      <td>Male</td>
      <td>33.7</td>
      <td>141</td>
      <td>150</td>
      <td>10.8</td>
      <td>6.9</td>
      <td>247</td>
      <td>221</td>
      <td>Low</td>
      <td>2811</td>
      <td>147.9</td>
      <td>6.7</td>
      <td>10</td>
      <td>Yes</td>
      <td>114.7</td>
      <td>1</td>
      <td>obese</td>
      <td>diabetic</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23</td>
      <td>Female</td>
      <td>32.8</td>
      <td>140</td>
      <td>145</td>
      <td>11.6</td>
      <td>6.8</td>
      <td>195</td>
      <td>193</td>
      <td>Low</td>
      <td>2826</td>
      <td>98.3</td>
      <td>4.4</td>
      <td>9</td>
      <td>Yes</td>
      <td>96.6</td>
      <td>1</td>
      <td>obese</td>
      <td>diabetic</td>
    </tr>
    <tr>
      <th>4</th>
      <td>70</td>
      <td>Male</td>
      <td>33.7</td>
      <td>165</td>
      <td>90</td>
      <td>18.3</td>
      <td>5.6</td>
      <td>217</td>
      <td>170</td>
      <td>Moderate</td>
      <td>2610</td>
      <td>65.8</td>
      <td>9.1</td>
      <td>5</td>
      <td>Yes</td>
      <td>107.4</td>
      <td>1</td>
      <td>obese</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5995</th>
      <td>58</td>
      <td>Male</td>
      <td>21.8</td>
      <td>158</td>
      <td>89</td>
      <td>6.3</td>
      <td>5.3</td>
      <td>198</td>
      <td>132</td>
      <td>High</td>
      <td>1995</td>
      <td>44.1</td>
      <td>7.6</td>
      <td>7</td>
      <td>No</td>
      <td>82.7</td>
      <td>0</td>
      <td>normal</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>5996</th>
      <td>73</td>
      <td>Male</td>
      <td>47.3</td>
      <td>193</td>
      <td>109</td>
      <td>33.7</td>
      <td>6.1</td>
      <td>250</td>
      <td>209</td>
      <td>Low</td>
      <td>3401</td>
      <td>61.4</td>
      <td>5.3</td>
      <td>10</td>
      <td>Yes</td>
      <td>150.0</td>
      <td>1</td>
      <td>obese</td>
      <td>prediabetic</td>
    </tr>
    <tr>
      <th>5997</th>
      <td>35</td>
      <td>Male</td>
      <td>31.0</td>
      <td>139</td>
      <td>90</td>
      <td>15.1</td>
      <td>5.3</td>
      <td>190</td>
      <td>164</td>
      <td>Moderate</td>
      <td>3022</td>
      <td>86.7</td>
      <td>6.8</td>
      <td>3</td>
      <td>Yes</td>
      <td>102.7</td>
      <td>0</td>
      <td>obese</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>5998</th>
      <td>58</td>
      <td>Male</td>
      <td>26.2</td>
      <td>136</td>
      <td>88</td>
      <td>6.3</td>
      <td>5.5</td>
      <td>223</td>
      <td>126</td>
      <td>High</td>
      <td>2311</td>
      <td>28.2</td>
      <td>8.4</td>
      <td>5</td>
      <td>No</td>
      <td>90.6</td>
      <td>0</td>
      <td>overweight</td>
      <td>normal</td>
    </tr>
    <tr>
      <th>5999</th>
      <td>54</td>
      <td>Male</td>
      <td>39.3</td>
      <td>158</td>
      <td>201</td>
      <td>10.7</td>
      <td>8.3</td>
      <td>240</td>
      <td>223</td>
      <td>Low</td>
      <td>2442</td>
      <td>113.1</td>
      <td>5.7</td>
      <td>6</td>
      <td>No</td>
      <td>123.8</td>
      <td>1</td>
      <td>obese</td>
      <td>diabetic</td>
    </tr>
  </tbody>
</table>
<p>6000 rows × 19 columns</p>
</div>



## SQL Exploration


```python
# Create in-memory SQLite database from cleaned dataset
conn = sqlite3.connect(":memory:")
df_eda.to_sql("diabetes", conn, index=False, if_exists="replace")
```




    6000



### Helper Function for Query Execution


```python
def run_query(query, description, n_rows=20):
    """
    Execute a SQL query against the in-memory SQLite database and display results.

    Parameters:
        query (str): SQL query string to execute.
        description (str): Human-readable description of what the query does.
        n_rows (int): Maximum number of rows to display (default: 20).

    Returns:
        None
    """
    print(f"\n{'='*80}")
    print(f"QUERY: {description}")
    print(f"{'='*80}")
    print(f"\nSQL:\n{query}\n")
    result = pd.read_sql_query(query, conn)
    print(f"Results ({len(result)} rows):")
    display(result.head(n_rows))
```

### Risk Distribution

Understand the distribution of `diabetes_risk_score` across the dataset.


```python
query = """
SELECT
    diabetes_risk_score,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM diabetes), 2) as percentage
FROM diabetes
GROUP BY diabetes_risk_score
ORDER BY count DESC;
"""
run_query(query, "Overall risk distribution")
```

    
    ================================================================================
    QUERY: Overall risk distribution
    ================================================================================
    
    SQL:
    
    SELECT
        diabetes_risk_score,
        COUNT(*) as count,
        ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM diabetes), 2) as percentage
    FROM diabetes
    GROUP BY diabetes_risk_score
    ORDER BY count DESC;
    
    
    Results (2 rows):



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>diabetes_risk_score</th>
      <th>count</th>
      <th>percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3001</td>
      <td>50.02</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2999</td>
      <td>49.98</td>
    </tr>
  </tbody>
</table>
</div>


### Feature Statistics by Risk Level

What are the average measurements for each risk level?


```python
query = """
SELECT
    diabetes_risk_score,
    ROUND(AVG(age), 2) as avg_age,
    ROUND(AVG(bmi), 2) as avg_bmi,
    ROUND(AVG(blood_pressure), 2) as avg_blood_pressure,
    ROUND(AVG(fasting_glucose_level), 2) as avg_fasting_glucose,
    ROUND(AVG(HbA1c_level), 2) as avg_HbA1c,
    ROUND(AVG(insulin_level), 2) as avg_insulin,
    COUNT(*) as sample_count
FROM diabetes
GROUP BY diabetes_risk_score
ORDER BY diabetes_risk_score;
"""
run_query(query, "Average feature measurements by risk level")
```

    
    ================================================================================
    QUERY: Average feature measurements by risk level
    ================================================================================
    
    SQL:
    
    SELECT
        diabetes_risk_score,
        ROUND(AVG(age), 2) as avg_age,
        ROUND(AVG(bmi), 2) as avg_bmi,
        ROUND(AVG(blood_pressure), 2) as avg_blood_pressure,
        ROUND(AVG(fasting_glucose_level), 2) as avg_fasting_glucose,
        ROUND(AVG(HbA1c_level), 2) as avg_HbA1c,
        ROUND(AVG(insulin_level), 2) as avg_insulin,
        COUNT(*) as sample_count
    FROM diabetes
    GROUP BY diabetes_risk_score
    ORDER BY diabetes_risk_score;
    
    
    Results (2 rows):



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>diabetes_risk_score</th>
      <th>avg_age</th>
      <th>avg_bmi</th>
      <th>avg_blood_pressure</th>
      <th>avg_fasting_glucose</th>
      <th>avg_HbA1c</th>
      <th>avg_insulin</th>
      <th>sample_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>48.55</td>
      <td>27.12</td>
      <td>133.84</td>
      <td>88.87</td>
      <td>5.27</td>
      <td>10.13</td>
      <td>3001</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>55.91</td>
      <td>38.76</td>
      <td>155.13</td>
      <td>123.35</td>
      <td>6.30</td>
      <td>20.03</td>
      <td>2999</td>
    </tr>
  </tbody>
</table>
</div>


### BMI Category Distribution by Risk Level

How does the engineered `bmi_category` feature distribute across risk levels?


```python
query = """
SELECT
    diabetes_risk_score,
    bmi_category,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY diabetes_risk_score), 2) as pct_within_risk
FROM diabetes
GROUP BY diabetes_risk_score, bmi_category
ORDER BY diabetes_risk_score, bmi_category;
"""
run_query(query, "BMI category distribution by risk level")
```

    
    ================================================================================
    QUERY: BMI category distribution by risk level
    ================================================================================
    
    SQL:
    
    SELECT
        diabetes_risk_score,
        bmi_category,
        COUNT(*) as count,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY diabetes_risk_score), 2) as pct_within_risk
    FROM diabetes
    GROUP BY diabetes_risk_score, bmi_category
    ORDER BY diabetes_risk_score, bmi_category;
    
    
    Results (8 rows):



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>diabetes_risk_score</th>
      <th>bmi_category</th>
      <th>count</th>
      <th>pct_within_risk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>normal</td>
      <td>951</td>
      <td>31.69</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>obese</td>
      <td>825</td>
      <td>27.49</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>overweight</td>
      <td>1158</td>
      <td>38.59</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>underweight</td>
      <td>67</td>
      <td>2.23</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>normal</td>
      <td>10</td>
      <td>0.33</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>obese</td>
      <td>2906</td>
      <td>96.90</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>overweight</td>
      <td>82</td>
      <td>2.73</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>underweight</td>
      <td>1</td>
      <td>0.03</td>
    </tr>
  </tbody>
</table>
</div>


### Glucose Category Distribution by Risk Level

How does the engineered `glucose_category` feature distribute across risk levels?


```python
query = """
SELECT
    diabetes_risk_score,
    glucose_category,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY diabetes_risk_score), 2) as pct_within_risk
FROM diabetes
GROUP BY diabetes_risk_score, glucose_category
ORDER BY diabetes_risk_score, glucose_category;
"""
run_query(query, "Glucose category distribution by risk level")
```

    
    ================================================================================
    QUERY: Glucose category distribution by risk level
    ================================================================================
    
    SQL:
    
    SELECT
        diabetes_risk_score,
        glucose_category,
        COUNT(*) as count,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY diabetes_risk_score), 2) as pct_within_risk
    FROM diabetes
    GROUP BY diabetes_risk_score, glucose_category
    ORDER BY diabetes_risk_score, glucose_category;
    
    
    Results (6 rows):



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>diabetes_risk_score</th>
      <th>glucose_category</th>
      <th>count</th>
      <th>pct_within_risk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>diabetic</td>
      <td>4</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>normal</td>
      <td>2653</td>
      <td>88.40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>prediabetic</td>
      <td>344</td>
      <td>11.46</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>diabetic</td>
      <td>996</td>
      <td>33.21</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>normal</td>
      <td>865</td>
      <td>28.84</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>prediabetic</td>
      <td>1138</td>
      <td>37.95</td>
    </tr>
  </tbody>
</table>
</div>


### Feature Ranges by Risk Level

What are the min/max feature values for each risk level?


```python
query = """
SELECT
    diabetes_risk_score,
    MIN(age) as min_age,
    MAX(age) as max_age,
    MIN(bmi) as min_bmi,
    MAX(bmi) as max_bmi,
    MIN(fasting_glucose_level) as min_glucose,
    MAX(fasting_glucose_level) as max_glucose
FROM diabetes
GROUP BY diabetes_risk_score
ORDER BY diabetes_risk_score;
"""
run_query(query, "Feature ranges by risk level")
```

    
    ================================================================================
    QUERY: Feature ranges by risk level
    ================================================================================
    
    SQL:
    
    SELECT
        diabetes_risk_score,
        MIN(age) as min_age,
        MAX(age) as max_age,
        MIN(bmi) as min_bmi,
        MAX(bmi) as max_bmi,
        MIN(fasting_glucose_level) as min_glucose,
        MAX(fasting_glucose_level) as max_glucose
    FROM diabetes
    GROUP BY diabetes_risk_score
    ORDER BY diabetes_risk_score;
    
    
    Results (2 rows):



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>diabetes_risk_score</th>
      <th>min_age</th>
      <th>max_age</th>
      <th>min_bmi</th>
      <th>max_bmi</th>
      <th>min_glucose</th>
      <th>max_glucose</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>20</td>
      <td>84</td>
      <td>16.0</td>
      <td>41.0</td>
      <td>60</td>
      <td>143</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>20</td>
      <td>84</td>
      <td>16.4</td>
      <td>50.0</td>
      <td>60</td>
      <td>281</td>
    </tr>
  </tbody>
</table>
</div>


### Risk Identification Patterns

Which features best identify each risk level?


```python
query = """
SELECT
    diabetes_risk_score,
    COUNT(*) as total,
    SUM(CASE WHEN bmi > 30 THEN 1 ELSE 0 END) as obese_count,
    ROUND(SUM(CASE WHEN bmi > 30 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as obese_pct,
    SUM(CASE WHEN family_history_diabetes = 'Yes' THEN 1 ELSE 0 END) as family_history_count,
    ROUND(SUM(CASE WHEN family_history_diabetes = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as family_history_pct
FROM diabetes
GROUP BY diabetes_risk_score
ORDER BY diabetes_risk_score;
"""
run_query(query, "Risk identification patterns")
```

    
    ================================================================================
    QUERY: Risk identification patterns
    ================================================================================
    
    SQL:
    
    SELECT
        diabetes_risk_score,
        COUNT(*) as total,
        SUM(CASE WHEN bmi > 30 THEN 1 ELSE 0 END) as obese_count,
        ROUND(SUM(CASE WHEN bmi > 30 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as obese_pct,
        SUM(CASE WHEN family_history_diabetes = 'Yes' THEN 1 ELSE 0 END) as family_history_count,
        ROUND(SUM(CASE WHEN family_history_diabetes = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as family_history_pct
    FROM diabetes
    GROUP BY diabetes_risk_score
    ORDER BY diabetes_risk_score;
    
    
    Results (2 rows):



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>diabetes_risk_score</th>
      <th>total</th>
      <th>obese_count</th>
      <th>obese_pct</th>
      <th>family_history_count</th>
      <th>family_history_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3001</td>
      <td>825</td>
      <td>27.49</td>
      <td>488</td>
      <td>16.26</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2999</td>
      <td>2906</td>
      <td>96.90</td>
      <td>674</td>
      <td>22.47</td>
    </tr>
  </tbody>
</table>
</div>



```python
conn.close()
```

## Exploratory Data Analysis


```python
df_eda.dtypes
```




    age                             int64
    gender                            str
    bmi                           float64
    blood_pressure                  int64
    fasting_glucose_level           int64
    insulin_level                 float64
    HbA1c_level                   float64
    cholesterol_level               int64
    triglycerides_level             int64
    physical_activity_level           str
    daily_calorie_intake            int64
    sugar_intake_grams_per_day    float64
    sleep_hours                   float64
    stress_level                    int64
    family_history_diabetes           str
    waist_circumference_cm        float64
    diabetes_risk_score             int64
    bmi_category                      str
    glucose_category                  str
    dtype: object



### Create `CATEGORICAL_FEATURES` & `CONTINUOUS_FEATURES`


```python
CATEGORICAL_FEATURES = [
    "gender",
    "physical_activity_level",
    "family_history_diabetes",
    "bmi_category",
    "glucose_category",
]
CATEGORICAL_FEATURES
```




    ['gender',
     'physical_activity_level',
     'family_history_diabetes',
     'bmi_category',
     'glucose_category']




```python
CONTINUOUS_FEATURES = [
    "age",
    "bmi",
    "blood_pressure",
    "fasting_glucose_level",
    "insulin_level",
    "HbA1c_level",
    "cholesterol_level",
    "triglycerides_level",
    "daily_calorie_intake",
    "sugar_intake_grams_per_day",
    "sleep_hours",
    "stress_level",
    "waist_circumference_cm",
]
CONTINUOUS_FEATURES
```




    ['age',
     'bmi',
     'blood_pressure',
     'fasting_glucose_level',
     'insulin_level',
     'HbA1c_level',
     'cholesterol_level',
     'triglycerides_level',
     'daily_calorie_intake',
     'sugar_intake_grams_per_day',
     'sleep_hours',
     'stress_level',
     'waist_circumference_cm']



### Create `TARGET_VARS`


```python
TARGET_VARS_list = df_clean[TARGET_VAR].value_counts()
TARGET_VARS = TARGET_VARS_list.index.tolist()
TARGET_VARS
```




    [0, 1]



### Verify Class Balance


```python
# Verify class balance (important: accuracy is misleading if imbalanced)
print("Class Distribution:")
print(TARGET_VARS_list)
print(f"\nPercentages:")
print((TARGET_VARS_list / TARGET_VARS_list.sum() * 100).round(1).astype(str) + "%")
# Balance ratio: min_class / max_class (1.0 = perfect, <0.5 = significantly imbalanced)
balance_ratio = TARGET_VARS_list.min() / TARGET_VARS_list.max()
print(f"\nBalance Ratio: {balance_ratio:.2f}")
if balance_ratio >= 0.8:
    print("Assessment: Well-Balanced ✅")
elif balance_ratio >= 0.5:
    print("Assessment: Moderately Imbalanced (consider stratified sampling)")
else:
    print("Assessment: Severely Imbalanced (use F1/AUC over accuracy, consider SMOTE)")
```

    Class Distribution:
    diabetes_risk_score
    0    3001
    1    2999
    Name: count, dtype: int64
    
    Percentages:
    diabetes_risk_score
    0    50.0%
    1    50.0%
    Name: count, dtype: str
    
    Balance Ratio: 1.00
    Assessment: Well-Balanced ✅


### Visualize the Marginal Distributions

#### Count Plots — Marginal Distributions of Each Categorical Variable, w/ Facets for Each Categorical Variable


```python
grid = sns.catplot(
    data=df_eda.melt(value_vars=CATEGORICAL_FEATURES),
    x="value",
    col="variable",
    col_wrap=3,
    kind="count",
    sharex=False,
    sharey=False,
    height=4,
    aspect=1.5,
)
for ax in grid.axes.flatten():
    for label in ax.get_xticklabels():
        label.set_rotation(90)
        label.set_ha("center")
grid.figure.tight_layout()
plt.subplots_adjust(hspace=0.4, wspace=0.2)
plt.show()
```


    
![png](README_files/DRP-MLP_67_0.png)
    


#### Count Plots with Facets - Explore Relationships Between Categorical Variables


```python
pairs = list(combinations(CATEGORICAL_FEATURES, 2))
for x_var, col_var in pairs:
    grid = sns.catplot(
        data=df_eda,
        x=x_var,
        col=col_var,
        col_wrap=3,
        kind="count",
        sharex=False,
        sharey=False,
        height=3,
        aspect=1.5,
    )
    for ax in grid.axes.flatten():
        for label in ax.get_xticklabels():
            label.set_rotation(90)
            label.set_ha("center")
    grid.figure.tight_layout()
    grid.figure.subplots_adjust(hspace=1.0, wspace=0.2)
    plt.show()
```


    
![png](README_files/DRP-MLP_69_0.png)
    



    
![png](README_files/DRP-MLP_69_1.png)
    



    
![png](README_files/DRP-MLP_69_2.png)
    



    
![png](README_files/DRP-MLP_69_3.png)
    



    
![png](README_files/DRP-MLP_69_4.png)
    



    
![png](README_files/DRP-MLP_69_5.png)
    



    
![png](README_files/DRP-MLP_69_6.png)
    



    
![png](README_files/DRP-MLP_69_7.png)
    



    
![png](README_files/DRP-MLP_69_8.png)
    



    
![png](README_files/DRP-MLP_69_9.png)
    


### Histogram Plots — Marginal Distributions of Each Numerical Variable, w/ Facets for Each Numerical Variable


```python
df_eda_lf = (
    df_eda.reset_index()
    .rename(columns={"index": "rowid"})
    .melt(id_vars=["rowid"] + CATEGORICAL_FEATURES, value_vars=CONTINUOUS_FEATURES)
)
df_eda_lf
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rowid</th>
      <th>gender</th>
      <th>physical_activity_level</th>
      <th>family_history_diabetes</th>
      <th>bmi_category</th>
      <th>glucose_category</th>
      <th>variable</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Female</td>
      <td>Low</td>
      <td>No</td>
      <td>obese</td>
      <td>normal</td>
      <td>age</td>
      <td>77.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Male</td>
      <td>High</td>
      <td>No</td>
      <td>normal</td>
      <td>normal</td>
      <td>age</td>
      <td>54.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Male</td>
      <td>Low</td>
      <td>Yes</td>
      <td>obese</td>
      <td>diabetic</td>
      <td>age</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Female</td>
      <td>Low</td>
      <td>Yes</td>
      <td>obese</td>
      <td>diabetic</td>
      <td>age</td>
      <td>23.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Male</td>
      <td>Moderate</td>
      <td>Yes</td>
      <td>obese</td>
      <td>normal</td>
      <td>age</td>
      <td>70.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>77995</th>
      <td>5995</td>
      <td>Male</td>
      <td>High</td>
      <td>No</td>
      <td>normal</td>
      <td>normal</td>
      <td>waist_circumference_cm</td>
      <td>82.7</td>
    </tr>
    <tr>
      <th>77996</th>
      <td>5996</td>
      <td>Male</td>
      <td>Low</td>
      <td>Yes</td>
      <td>obese</td>
      <td>prediabetic</td>
      <td>waist_circumference_cm</td>
      <td>150.0</td>
    </tr>
    <tr>
      <th>77997</th>
      <td>5997</td>
      <td>Male</td>
      <td>Moderate</td>
      <td>Yes</td>
      <td>obese</td>
      <td>normal</td>
      <td>waist_circumference_cm</td>
      <td>102.7</td>
    </tr>
    <tr>
      <th>77998</th>
      <td>5998</td>
      <td>Male</td>
      <td>High</td>
      <td>No</td>
      <td>overweight</td>
      <td>normal</td>
      <td>waist_circumference_cm</td>
      <td>90.6</td>
    </tr>
    <tr>
      <th>77999</th>
      <td>5999</td>
      <td>Male</td>
      <td>Low</td>
      <td>No</td>
      <td>obese</td>
      <td>diabetic</td>
      <td>waist_circumference_cm</td>
      <td>123.8</td>
    </tr>
  </tbody>
</table>
<p>78000 rows × 8 columns</p>
</div>




```python
sns.displot(
    data=df_eda_lf,
    x="value",
    col="variable",
    col_wrap=4,
    kind="hist",
    facet_kws={"sharex": False, "sharey": False},
    common_bins=False,
)
plt.show()
```


    
![png](README_files/DRP-MLP_72_0.png)
    


### Histogram Plots with Facets - Explore Relationships Between Numerical & Categorical Variables


```python
for num_var in CONTINUOUS_FEATURES:
    for cat_var in CATEGORICAL_FEATURES:
        grid = sns.displot(
            data=df_eda,
            x=num_var,
            col=cat_var,
            col_wrap=3,
            kind="hist",
            facet_kws={"sharex": False, "sharey": False},
            height=3,
            aspect=1.5,
        )
        for ax in grid.axes.flatten():
            for label in ax.get_xticklabels():
                label.set_rotation(90)
                label.set_ha("center")
        grid.figure.tight_layout()
        grid.figure.subplots_adjust(hspace=0.4, wspace=0.2)
        plt.show()
```


    
![png](README_files/DRP-MLP_74_0.png)
    



    
![png](README_files/DRP-MLP_74_1.png)
    



    
![png](README_files/DRP-MLP_74_2.png)
    



    
![png](README_files/DRP-MLP_74_3.png)
    



    
![png](README_files/DRP-MLP_74_4.png)
    



    
![png](README_files/DRP-MLP_74_5.png)
    



    
![png](README_files/DRP-MLP_74_6.png)
    



    
![png](README_files/DRP-MLP_74_7.png)
    



    
![png](README_files/DRP-MLP_74_8.png)
    



    
![png](README_files/DRP-MLP_74_9.png)
    



    
![png](README_files/DRP-MLP_74_10.png)
    



    
![png](README_files/DRP-MLP_74_11.png)
    



    
![png](README_files/DRP-MLP_74_12.png)
    



    
![png](README_files/DRP-MLP_74_13.png)
    



    
![png](README_files/DRP-MLP_74_14.png)
    



    
![png](README_files/DRP-MLP_74_15.png)
    



    
![png](README_files/DRP-MLP_74_16.png)
    



    
![png](README_files/DRP-MLP_74_17.png)
    



    
![png](README_files/DRP-MLP_74_18.png)
    



    
![png](README_files/DRP-MLP_74_19.png)
    



    
![png](README_files/DRP-MLP_74_20.png)
    



    
![png](README_files/DRP-MLP_74_21.png)
    



    
![png](README_files/DRP-MLP_74_22.png)
    



    
![png](README_files/DRP-MLP_74_23.png)
    



    
![png](README_files/DRP-MLP_74_24.png)
    



    
![png](README_files/DRP-MLP_74_25.png)
    



    
![png](README_files/DRP-MLP_74_26.png)
    



    
![png](README_files/DRP-MLP_74_27.png)
    



    
![png](README_files/DRP-MLP_74_28.png)
    



    
![png](README_files/DRP-MLP_74_29.png)
    



    
![png](README_files/DRP-MLP_74_30.png)
    



    
![png](README_files/DRP-MLP_74_31.png)
    



    
![png](README_files/DRP-MLP_74_32.png)
    



    
![png](README_files/DRP-MLP_74_33.png)
    



    
![png](README_files/DRP-MLP_74_34.png)
    



    
![png](README_files/DRP-MLP_74_35.png)
    



    
![png](README_files/DRP-MLP_74_36.png)
    



    
![png](README_files/DRP-MLP_74_37.png)
    



    
![png](README_files/DRP-MLP_74_38.png)
    



    
![png](README_files/DRP-MLP_74_39.png)
    



    
![png](README_files/DRP-MLP_74_40.png)
    



    
![png](README_files/DRP-MLP_74_41.png)
    



    
![png](README_files/DRP-MLP_74_42.png)
    



    
![png](README_files/DRP-MLP_74_43.png)
    



    
![png](README_files/DRP-MLP_74_44.png)
    



    
![png](README_files/DRP-MLP_74_45.png)
    



    
![png](README_files/DRP-MLP_74_46.png)
    



    
![png](README_files/DRP-MLP_74_47.png)
    



    
![png](README_files/DRP-MLP_74_48.png)
    



    
![png](README_files/DRP-MLP_74_49.png)
    



    
![png](README_files/DRP-MLP_74_50.png)
    



    
![png](README_files/DRP-MLP_74_51.png)
    



    
![png](README_files/DRP-MLP_74_52.png)
    



    
![png](README_files/DRP-MLP_74_53.png)
    



    
![png](README_files/DRP-MLP_74_54.png)
    



    
![png](README_files/DRP-MLP_74_55.png)
    



    
![png](README_files/DRP-MLP_74_56.png)
    



    
![png](README_files/DRP-MLP_74_57.png)
    



    
![png](README_files/DRP-MLP_74_58.png)
    



    
![png](README_files/DRP-MLP_74_59.png)
    



    
![png](README_files/DRP-MLP_74_60.png)
    



    
![png](README_files/DRP-MLP_74_61.png)
    



    
![png](README_files/DRP-MLP_74_62.png)
    



    
![png](README_files/DRP-MLP_74_63.png)
    



    
![png](README_files/DRP-MLP_74_64.png)
    


### Box Plots — Distribution of Each Numerical Variable by Categorical Variable, Faceted by the Respective Numerical Variable


```python
for num_var in CONTINUOUS_FEATURES:
    for cat_var in CATEGORICAL_FEATURES:
        n_categories = df_eda[cat_var].nunique()
        fig_width = max(8, n_categories * 0.6)
        grid = sns.catplot(
            data=df_eda,
            x=cat_var,
            y=num_var,
            kind="box",
            height=6,
            aspect=fig_width / 6,
        )
        for ax in grid.axes.flatten():
            for label in ax.get_xticklabels():
                label.set_rotation(90)
                label.set_ha("center")
        grid.figure.tight_layout()
        grid.figure.subplots_adjust(hspace=0.4, wspace=0.2)
        plt.show()
```


    
![png](README_files/DRP-MLP_76_0.png)
    



    
![png](README_files/DRP-MLP_76_1.png)
    



    
![png](README_files/DRP-MLP_76_2.png)
    



    
![png](README_files/DRP-MLP_76_3.png)
    



    
![png](README_files/DRP-MLP_76_4.png)
    



    
![png](README_files/DRP-MLP_76_5.png)
    



    
![png](README_files/DRP-MLP_76_6.png)
    



    
![png](README_files/DRP-MLP_76_7.png)
    



    
![png](README_files/DRP-MLP_76_8.png)
    



    
![png](README_files/DRP-MLP_76_9.png)
    



    
![png](README_files/DRP-MLP_76_10.png)
    



    
![png](README_files/DRP-MLP_76_11.png)
    



    
![png](README_files/DRP-MLP_76_12.png)
    



    
![png](README_files/DRP-MLP_76_13.png)
    



    
![png](README_files/DRP-MLP_76_14.png)
    



    
![png](README_files/DRP-MLP_76_15.png)
    



    
![png](README_files/DRP-MLP_76_16.png)
    



    
![png](README_files/DRP-MLP_76_17.png)
    



    
![png](README_files/DRP-MLP_76_18.png)
    



    
![png](README_files/DRP-MLP_76_19.png)
    



    
![png](README_files/DRP-MLP_76_20.png)
    



    
![png](README_files/DRP-MLP_76_21.png)
    



    
![png](README_files/DRP-MLP_76_22.png)
    



    
![png](README_files/DRP-MLP_76_23.png)
    



    
![png](README_files/DRP-MLP_76_24.png)
    



    
![png](README_files/DRP-MLP_76_25.png)
    



    
![png](README_files/DRP-MLP_76_26.png)
    



    
![png](README_files/DRP-MLP_76_27.png)
    



    
![png](README_files/DRP-MLP_76_28.png)
    



    
![png](README_files/DRP-MLP_76_29.png)
    



    
![png](README_files/DRP-MLP_76_30.png)
    



    
![png](README_files/DRP-MLP_76_31.png)
    



    
![png](README_files/DRP-MLP_76_32.png)
    



    
![png](README_files/DRP-MLP_76_33.png)
    



    
![png](README_files/DRP-MLP_76_34.png)
    



    
![png](README_files/DRP-MLP_76_35.png)
    



    
![png](README_files/DRP-MLP_76_36.png)
    



    
![png](README_files/DRP-MLP_76_37.png)
    



    
![png](README_files/DRP-MLP_76_38.png)
    



    
![png](README_files/DRP-MLP_76_39.png)
    



    
![png](README_files/DRP-MLP_76_40.png)
    



    
![png](README_files/DRP-MLP_76_41.png)
    



    
![png](README_files/DRP-MLP_76_42.png)
    



    
![png](README_files/DRP-MLP_76_43.png)
    



    
![png](README_files/DRP-MLP_76_44.png)
    



    
![png](README_files/DRP-MLP_76_45.png)
    



    
![png](README_files/DRP-MLP_76_46.png)
    



    
![png](README_files/DRP-MLP_76_47.png)
    



    
![png](README_files/DRP-MLP_76_48.png)
    



    
![png](README_files/DRP-MLP_76_49.png)
    



    
![png](README_files/DRP-MLP_76_50.png)
    



    
![png](README_files/DRP-MLP_76_51.png)
    



    
![png](README_files/DRP-MLP_76_52.png)
    



    
![png](README_files/DRP-MLP_76_53.png)
    



    
![png](README_files/DRP-MLP_76_54.png)
    



    
![png](README_files/DRP-MLP_76_55.png)
    



    
![png](README_files/DRP-MLP_76_56.png)
    



    
![png](README_files/DRP-MLP_76_57.png)
    



    
![png](README_files/DRP-MLP_76_58.png)
    



    
![png](README_files/DRP-MLP_76_59.png)
    



    
![png](README_files/DRP-MLP_76_60.png)
    



    
![png](README_files/DRP-MLP_76_61.png)
    



    
![png](README_files/DRP-MLP_76_62.png)
    



    
![png](README_files/DRP-MLP_76_63.png)
    



    
![png](README_files/DRP-MLP_76_64.png)
    


### Point Plots — Mean and Confidence Interval of Each Numerical Variable by Categorical Variable, Faceted by the Respective Numerical Variable


```python
for num_var in CONTINUOUS_FEATURES:
    for cat_var in CATEGORICAL_FEATURES:
        n_categories = df_eda[cat_var].nunique()
        fig_width = max(8, n_categories * 0.6)
        grid = sns.catplot(
            data=df_eda,
            x=cat_var,
            y=num_var,
            kind="point",
            linestyle="none",
            height=6,
            aspect=fig_width / 6,
        )
        for ax in grid.axes.flatten():
            for label in ax.get_xticklabels():
                label.set_rotation(90)
                label.set_ha("center")
        grid.figure.tight_layout()
        grid.figure.subplots_adjust(hspace=0.4, wspace=0.2)
        plt.show()
```


    
![png](README_files/DRP-MLP_78_0.png)
    



    
![png](README_files/DRP-MLP_78_1.png)
    



    
![png](README_files/DRP-MLP_78_2.png)
    



    
![png](README_files/DRP-MLP_78_3.png)
    



    
![png](README_files/DRP-MLP_78_4.png)
    



    
![png](README_files/DRP-MLP_78_5.png)
    



    
![png](README_files/DRP-MLP_78_6.png)
    



    
![png](README_files/DRP-MLP_78_7.png)
    



    
![png](README_files/DRP-MLP_78_8.png)
    



    
![png](README_files/DRP-MLP_78_9.png)
    



    
![png](README_files/DRP-MLP_78_10.png)
    



    
![png](README_files/DRP-MLP_78_11.png)
    



    
![png](README_files/DRP-MLP_78_12.png)
    



    
![png](README_files/DRP-MLP_78_13.png)
    



    
![png](README_files/DRP-MLP_78_14.png)
    



    
![png](README_files/DRP-MLP_78_15.png)
    



    
![png](README_files/DRP-MLP_78_16.png)
    



    
![png](README_files/DRP-MLP_78_17.png)
    



    
![png](README_files/DRP-MLP_78_18.png)
    



    
![png](README_files/DRP-MLP_78_19.png)
    



    
![png](README_files/DRP-MLP_78_20.png)
    



    
![png](README_files/DRP-MLP_78_21.png)
    



    
![png](README_files/DRP-MLP_78_22.png)
    



    
![png](README_files/DRP-MLP_78_23.png)
    



    
![png](README_files/DRP-MLP_78_24.png)
    



    
![png](README_files/DRP-MLP_78_25.png)
    



    
![png](README_files/DRP-MLP_78_26.png)
    



    
![png](README_files/DRP-MLP_78_27.png)
    



    
![png](README_files/DRP-MLP_78_28.png)
    



    
![png](README_files/DRP-MLP_78_29.png)
    



    
![png](README_files/DRP-MLP_78_30.png)
    



    
![png](README_files/DRP-MLP_78_31.png)
    



    
![png](README_files/DRP-MLP_78_32.png)
    



    
![png](README_files/DRP-MLP_78_33.png)
    



    
![png](README_files/DRP-MLP_78_34.png)
    



    
![png](README_files/DRP-MLP_78_35.png)
    



    
![png](README_files/DRP-MLP_78_36.png)
    



    
![png](README_files/DRP-MLP_78_37.png)
    



    
![png](README_files/DRP-MLP_78_38.png)
    



    
![png](README_files/DRP-MLP_78_39.png)
    



    
![png](README_files/DRP-MLP_78_40.png)
    



    
![png](README_files/DRP-MLP_78_41.png)
    



    
![png](README_files/DRP-MLP_78_42.png)
    



    
![png](README_files/DRP-MLP_78_43.png)
    



    
![png](README_files/DRP-MLP_78_44.png)
    



    
![png](README_files/DRP-MLP_78_45.png)
    



    
![png](README_files/DRP-MLP_78_46.png)
    



    
![png](README_files/DRP-MLP_78_47.png)
    



    
![png](README_files/DRP-MLP_78_48.png)
    



    
![png](README_files/DRP-MLP_78_49.png)
    



    
![png](README_files/DRP-MLP_78_50.png)
    



    
![png](README_files/DRP-MLP_78_51.png)
    



    
![png](README_files/DRP-MLP_78_52.png)
    



    
![png](README_files/DRP-MLP_78_53.png)
    



    
![png](README_files/DRP-MLP_78_54.png)
    



    
![png](README_files/DRP-MLP_78_55.png)
    



    
![png](README_files/DRP-MLP_78_56.png)
    



    
![png](README_files/DRP-MLP_78_57.png)
    



    
![png](README_files/DRP-MLP_78_58.png)
    



    
![png](README_files/DRP-MLP_78_59.png)
    



    
![png](README_files/DRP-MLP_78_60.png)
    



    
![png](README_files/DRP-MLP_78_61.png)
    



    
![png](README_files/DRP-MLP_78_62.png)
    



    
![png](README_files/DRP-MLP_78_63.png)
    



    
![png](README_files/DRP-MLP_78_64.png)
    


### Regression Plots — Linear Relationships Between Pairs of Numerical Variables, Colored by Categorical Variable


```python
def smart_downsample(
    df: pd.DataFrame,
    hue_col: str,
    num_cols: list[str],
    max_points: int = 10_000,
    outlier_keep: float = 1.0,
    random_state: int = SEED,
) -> pd.DataFrame:
    """Stratified downsample that preserves category proportions and keeps outliers.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    hue_col : str
        Categorical column used for stratification.
    num_cols : list[str]
        Numerical columns checked for outliers (IQR method).
    max_points : int
        Target number of rows when df exceeds this size.
    outlier_keep : float
        Fraction of detected outliers to always keep (1.0 = all).
    random_state : int
        Reproducibility seed.

    Returns
    -------
    pd.DataFrame
        Down-sampled dataframe (original returned unchanged if small enough).
    """
    if len(df) <= max_points:
        return df
    # --- 1. Identify IQR outliers across all numerical columns ---
    outlier_mask = pd.Series(False, index=df.index)
    for col in num_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        outlier_mask |= (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
    outliers = df[outlier_mask]
    if outlier_keep < 1.0:
        outliers = outliers.sample(frac=outlier_keep, random_state=random_state)
    # --- 2. Stratified sample from non-outlier rows ---
    non_outliers = df[~df.index.isin(outliers.index)]
    remaining_budget = max(0, max_points - len(outliers))
    if remaining_budget > 0 and len(non_outliers) > 0:
        frac = min(1.0, remaining_budget / len(non_outliers))
        sampled = non_outliers.groupby(hue_col, group_keys=False).apply(
            lambda g: g.sample(frac=frac, random_state=random_state)
        )
    else:
        sampled = non_outliers.iloc[:0]
    result = pd.concat([outliers, sampled]).drop_duplicates()
    print(
        f"smart_downsample: {len(df):,} → {len(result):,} rows "
        f"(outliers kept: {len(outliers):,}, sampled: {len(sampled):,})"
    )
    return result
```


```python
for x_var, y_var in combinations(CONTINUOUS_FEATURES, 2):
    for cat_var in CATEGORICAL_FEATURES:
        plot_df = smart_downsample(df_eda, hue_col=cat_var, num_cols=[x_var, y_var])
        sns.lmplot(
            data=plot_df,
            x=x_var,
            y=y_var,
            hue=cat_var,
            height=5,
            aspect=1.2,
        )
        plt.show()
```


    
![png](README_files/DRP-MLP_81_0.png)
    



    
![png](README_files/DRP-MLP_81_1.png)
    



    
![png](README_files/DRP-MLP_81_2.png)
    



    
![png](README_files/DRP-MLP_81_3.png)
    



    
![png](README_files/DRP-MLP_81_4.png)
    



    
![png](README_files/DRP-MLP_81_5.png)
    



    
![png](README_files/DRP-MLP_81_6.png)
    



    
![png](README_files/DRP-MLP_81_7.png)
    



    
![png](README_files/DRP-MLP_81_8.png)
    



    
![png](README_files/DRP-MLP_81_9.png)
    



    
![png](README_files/DRP-MLP_81_10.png)
    



    
![png](README_files/DRP-MLP_81_11.png)
    



    
![png](README_files/DRP-MLP_81_12.png)
    



    
![png](README_files/DRP-MLP_81_13.png)
    



    
![png](README_files/DRP-MLP_81_14.png)
    



    
![png](README_files/DRP-MLP_81_15.png)
    



    
![png](README_files/DRP-MLP_81_16.png)
    



    
![png](README_files/DRP-MLP_81_17.png)
    



    
![png](README_files/DRP-MLP_81_18.png)
    



    
![png](README_files/DRP-MLP_81_19.png)
    



    
![png](README_files/DRP-MLP_81_20.png)
    



    
![png](README_files/DRP-MLP_81_21.png)
    



    
![png](README_files/DRP-MLP_81_22.png)
    



    
![png](README_files/DRP-MLP_81_23.png)
    



    
![png](README_files/DRP-MLP_81_24.png)
    



    
![png](README_files/DRP-MLP_81_25.png)
    



    
![png](README_files/DRP-MLP_81_26.png)
    



    
![png](README_files/DRP-MLP_81_27.png)
    



    
![png](README_files/DRP-MLP_81_28.png)
    



    
![png](README_files/DRP-MLP_81_29.png)
    



    
![png](README_files/DRP-MLP_81_30.png)
    



    
![png](README_files/DRP-MLP_81_31.png)
    



    
![png](README_files/DRP-MLP_81_32.png)
    



    
![png](README_files/DRP-MLP_81_33.png)
    



    
![png](README_files/DRP-MLP_81_34.png)
    



    
![png](README_files/DRP-MLP_81_35.png)
    



    
![png](README_files/DRP-MLP_81_36.png)
    



    
![png](README_files/DRP-MLP_81_37.png)
    



    
![png](README_files/DRP-MLP_81_38.png)
    



    
![png](README_files/DRP-MLP_81_39.png)
    



    
![png](README_files/DRP-MLP_81_40.png)
    



    
![png](README_files/DRP-MLP_81_41.png)
    



    
![png](README_files/DRP-MLP_81_42.png)
    



    
![png](README_files/DRP-MLP_81_43.png)
    



    
![png](README_files/DRP-MLP_81_44.png)
    



    
![png](README_files/DRP-MLP_81_45.png)
    



    
![png](README_files/DRP-MLP_81_46.png)
    



    
![png](README_files/DRP-MLP_81_47.png)
    



    
![png](README_files/DRP-MLP_81_48.png)
    



    
![png](README_files/DRP-MLP_81_49.png)
    



    
![png](README_files/DRP-MLP_81_50.png)
    



    
![png](README_files/DRP-MLP_81_51.png)
    



    
![png](README_files/DRP-MLP_81_52.png)
    



    
![png](README_files/DRP-MLP_81_53.png)
    



    
![png](README_files/DRP-MLP_81_54.png)
    



    
![png](README_files/DRP-MLP_81_55.png)
    



    
![png](README_files/DRP-MLP_81_56.png)
    



    
![png](README_files/DRP-MLP_81_57.png)
    



    
![png](README_files/DRP-MLP_81_58.png)
    



    
![png](README_files/DRP-MLP_81_59.png)
    



    
![png](README_files/DRP-MLP_81_60.png)
    



    
![png](README_files/DRP-MLP_81_61.png)
    



    
![png](README_files/DRP-MLP_81_62.png)
    



    
![png](README_files/DRP-MLP_81_63.png)
    



    
![png](README_files/DRP-MLP_81_64.png)
    



    
![png](README_files/DRP-MLP_81_65.png)
    



    
![png](README_files/DRP-MLP_81_66.png)
    



    
![png](README_files/DRP-MLP_81_67.png)
    



    
![png](README_files/DRP-MLP_81_68.png)
    



    
![png](README_files/DRP-MLP_81_69.png)
    



    
![png](README_files/DRP-MLP_81_70.png)
    



    
![png](README_files/DRP-MLP_81_71.png)
    



    
![png](README_files/DRP-MLP_81_72.png)
    



    
![png](README_files/DRP-MLP_81_73.png)
    



    
![png](README_files/DRP-MLP_81_74.png)
    



    
![png](README_files/DRP-MLP_81_75.png)
    



    
![png](README_files/DRP-MLP_81_76.png)
    



    
![png](README_files/DRP-MLP_81_77.png)
    



    
![png](README_files/DRP-MLP_81_78.png)
    



    
![png](README_files/DRP-MLP_81_79.png)
    



    
![png](README_files/DRP-MLP_81_80.png)
    



    
![png](README_files/DRP-MLP_81_81.png)
    



    
![png](README_files/DRP-MLP_81_82.png)
    



    
![png](README_files/DRP-MLP_81_83.png)
    



    
![png](README_files/DRP-MLP_81_84.png)
    



    
![png](README_files/DRP-MLP_81_85.png)
    



    
![png](README_files/DRP-MLP_81_86.png)
    



    
![png](README_files/DRP-MLP_81_87.png)
    



    
![png](README_files/DRP-MLP_81_88.png)
    



    
![png](README_files/DRP-MLP_81_89.png)
    



    
![png](README_files/DRP-MLP_81_90.png)
    



    
![png](README_files/DRP-MLP_81_91.png)
    



    
![png](README_files/DRP-MLP_81_92.png)
    



    
![png](README_files/DRP-MLP_81_93.png)
    



    
![png](README_files/DRP-MLP_81_94.png)
    



    
![png](README_files/DRP-MLP_81_95.png)
    



    
![png](README_files/DRP-MLP_81_96.png)
    



    
![png](README_files/DRP-MLP_81_97.png)
    



    
![png](README_files/DRP-MLP_81_98.png)
    



    
![png](README_files/DRP-MLP_81_99.png)
    



    
![png](README_files/DRP-MLP_81_100.png)
    



    
![png](README_files/DRP-MLP_81_101.png)
    



    
![png](README_files/DRP-MLP_81_102.png)
    



    
![png](README_files/DRP-MLP_81_103.png)
    



    
![png](README_files/DRP-MLP_81_104.png)
    



    
![png](README_files/DRP-MLP_81_105.png)
    



    
![png](README_files/DRP-MLP_81_106.png)
    



    
![png](README_files/DRP-MLP_81_107.png)
    



    
![png](README_files/DRP-MLP_81_108.png)
    



    
![png](README_files/DRP-MLP_81_109.png)
    



    
![png](README_files/DRP-MLP_81_110.png)
    



    
![png](README_files/DRP-MLP_81_111.png)
    



    
![png](README_files/DRP-MLP_81_112.png)
    



    
![png](README_files/DRP-MLP_81_113.png)
    



    
![png](README_files/DRP-MLP_81_114.png)
    



    
![png](README_files/DRP-MLP_81_115.png)
    



    
![png](README_files/DRP-MLP_81_116.png)
    



    
![png](README_files/DRP-MLP_81_117.png)
    



    
![png](README_files/DRP-MLP_81_118.png)
    



    
![png](README_files/DRP-MLP_81_119.png)
    



    
![png](README_files/DRP-MLP_81_120.png)
    



    
![png](README_files/DRP-MLP_81_121.png)
    



    
![png](README_files/DRP-MLP_81_122.png)
    



    
![png](README_files/DRP-MLP_81_123.png)
    



    
![png](README_files/DRP-MLP_81_124.png)
    



    
![png](README_files/DRP-MLP_81_125.png)
    



    
![png](README_files/DRP-MLP_81_126.png)
    



    
![png](README_files/DRP-MLP_81_127.png)
    



    
![png](README_files/DRP-MLP_81_128.png)
    



    
![png](README_files/DRP-MLP_81_129.png)
    



    
![png](README_files/DRP-MLP_81_130.png)
    



    
![png](README_files/DRP-MLP_81_131.png)
    



    
![png](README_files/DRP-MLP_81_132.png)
    



    
![png](README_files/DRP-MLP_81_133.png)
    



    
![png](README_files/DRP-MLP_81_134.png)
    



    
![png](README_files/DRP-MLP_81_135.png)
    



    
![png](README_files/DRP-MLP_81_136.png)
    



    
![png](README_files/DRP-MLP_81_137.png)
    



    
![png](README_files/DRP-MLP_81_138.png)
    



    
![png](README_files/DRP-MLP_81_139.png)
    



    
![png](README_files/DRP-MLP_81_140.png)
    



    
![png](README_files/DRP-MLP_81_141.png)
    



    
![png](README_files/DRP-MLP_81_142.png)
    



    
![png](README_files/DRP-MLP_81_143.png)
    



    
![png](README_files/DRP-MLP_81_144.png)
    



    
![png](README_files/DRP-MLP_81_145.png)
    



    
![png](README_files/DRP-MLP_81_146.png)
    



    
![png](README_files/DRP-MLP_81_147.png)
    



    
![png](README_files/DRP-MLP_81_148.png)
    



    
![png](README_files/DRP-MLP_81_149.png)
    



    
![png](README_files/DRP-MLP_81_150.png)
    



    
![png](README_files/DRP-MLP_81_151.png)
    



    
![png](README_files/DRP-MLP_81_152.png)
    



    
![png](README_files/DRP-MLP_81_153.png)
    



    
![png](README_files/DRP-MLP_81_154.png)
    



    
![png](README_files/DRP-MLP_81_155.png)
    



    
![png](README_files/DRP-MLP_81_156.png)
    



    
![png](README_files/DRP-MLP_81_157.png)
    



    
![png](README_files/DRP-MLP_81_158.png)
    



    
![png](README_files/DRP-MLP_81_159.png)
    



    
![png](README_files/DRP-MLP_81_160.png)
    



    
![png](README_files/DRP-MLP_81_161.png)
    



    
![png](README_files/DRP-MLP_81_162.png)
    



    
![png](README_files/DRP-MLP_81_163.png)
    



    
![png](README_files/DRP-MLP_81_164.png)
    



    
![png](README_files/DRP-MLP_81_165.png)
    



    
![png](README_files/DRP-MLP_81_166.png)
    



    
![png](README_files/DRP-MLP_81_167.png)
    



    
![png](README_files/DRP-MLP_81_168.png)
    



    
![png](README_files/DRP-MLP_81_169.png)
    



    
![png](README_files/DRP-MLP_81_170.png)
    



    
![png](README_files/DRP-MLP_81_171.png)
    



    
![png](README_files/DRP-MLP_81_172.png)
    



    
![png](README_files/DRP-MLP_81_173.png)
    



    
![png](README_files/DRP-MLP_81_174.png)
    



    
![png](README_files/DRP-MLP_81_175.png)
    



    
![png](README_files/DRP-MLP_81_176.png)
    



    
![png](README_files/DRP-MLP_81_177.png)
    



    
![png](README_files/DRP-MLP_81_178.png)
    



    
![png](README_files/DRP-MLP_81_179.png)
    



    
![png](README_files/DRP-MLP_81_180.png)
    



    
![png](README_files/DRP-MLP_81_181.png)
    



    
![png](README_files/DRP-MLP_81_182.png)
    



    
![png](README_files/DRP-MLP_81_183.png)
    



    
![png](README_files/DRP-MLP_81_184.png)
    



    
![png](README_files/DRP-MLP_81_185.png)
    



    
![png](README_files/DRP-MLP_81_186.png)
    



    
![png](README_files/DRP-MLP_81_187.png)
    



    
![png](README_files/DRP-MLP_81_188.png)
    



    
![png](README_files/DRP-MLP_81_189.png)
    



    
![png](README_files/DRP-MLP_81_190.png)
    



    
![png](README_files/DRP-MLP_81_191.png)
    



    
![png](README_files/DRP-MLP_81_192.png)
    



    
![png](README_files/DRP-MLP_81_193.png)
    



    
![png](README_files/DRP-MLP_81_194.png)
    



    
![png](README_files/DRP-MLP_81_195.png)
    



    
![png](README_files/DRP-MLP_81_196.png)
    



    
![png](README_files/DRP-MLP_81_197.png)
    



    
![png](README_files/DRP-MLP_81_198.png)
    



    
![png](README_files/DRP-MLP_81_199.png)
    



    
![png](README_files/DRP-MLP_81_200.png)
    



    
![png](README_files/DRP-MLP_81_201.png)
    



    
![png](README_files/DRP-MLP_81_202.png)
    



    
![png](README_files/DRP-MLP_81_203.png)
    



    
![png](README_files/DRP-MLP_81_204.png)
    



    
![png](README_files/DRP-MLP_81_205.png)
    



    
![png](README_files/DRP-MLP_81_206.png)
    



    
![png](README_files/DRP-MLP_81_207.png)
    



    
![png](README_files/DRP-MLP_81_208.png)
    



    
![png](README_files/DRP-MLP_81_209.png)
    



    
![png](README_files/DRP-MLP_81_210.png)
    



    
![png](README_files/DRP-MLP_81_211.png)
    



    
![png](README_files/DRP-MLP_81_212.png)
    



    
![png](README_files/DRP-MLP_81_213.png)
    



    
![png](README_files/DRP-MLP_81_214.png)
    



    
![png](README_files/DRP-MLP_81_215.png)
    



    
![png](README_files/DRP-MLP_81_216.png)
    



    
![png](README_files/DRP-MLP_81_217.png)
    



    
![png](README_files/DRP-MLP_81_218.png)
    



    
![png](README_files/DRP-MLP_81_219.png)
    



    
![png](README_files/DRP-MLP_81_220.png)
    



    
![png](README_files/DRP-MLP_81_221.png)
    



    
![png](README_files/DRP-MLP_81_222.png)
    



    
![png](README_files/DRP-MLP_81_223.png)
    



    
![png](README_files/DRP-MLP_81_224.png)
    



    
![png](README_files/DRP-MLP_81_225.png)
    



    
![png](README_files/DRP-MLP_81_226.png)
    



    
![png](README_files/DRP-MLP_81_227.png)
    



    
![png](README_files/DRP-MLP_81_228.png)
    



    
![png](README_files/DRP-MLP_81_229.png)
    



    
![png](README_files/DRP-MLP_81_230.png)
    



    
![png](README_files/DRP-MLP_81_231.png)
    



    
![png](README_files/DRP-MLP_81_232.png)
    



    
![png](README_files/DRP-MLP_81_233.png)
    



    
![png](README_files/DRP-MLP_81_234.png)
    



    
![png](README_files/DRP-MLP_81_235.png)
    



    
![png](README_files/DRP-MLP_81_236.png)
    



    
![png](README_files/DRP-MLP_81_237.png)
    



    
![png](README_files/DRP-MLP_81_238.png)
    



    
![png](README_files/DRP-MLP_81_239.png)
    



    
![png](README_files/DRP-MLP_81_240.png)
    



    
![png](README_files/DRP-MLP_81_241.png)
    



    
![png](README_files/DRP-MLP_81_242.png)
    



    
![png](README_files/DRP-MLP_81_243.png)
    



    
![png](README_files/DRP-MLP_81_244.png)
    



    
![png](README_files/DRP-MLP_81_245.png)
    



    
![png](README_files/DRP-MLP_81_246.png)
    



    
![png](README_files/DRP-MLP_81_247.png)
    



    
![png](README_files/DRP-MLP_81_248.png)
    



    
![png](README_files/DRP-MLP_81_249.png)
    



    
![png](README_files/DRP-MLP_81_250.png)
    



    
![png](README_files/DRP-MLP_81_251.png)
    



    
![png](README_files/DRP-MLP_81_252.png)
    



    
![png](README_files/DRP-MLP_81_253.png)
    



    
![png](README_files/DRP-MLP_81_254.png)
    



    
![png](README_files/DRP-MLP_81_255.png)
    



    
![png](README_files/DRP-MLP_81_256.png)
    



    
![png](README_files/DRP-MLP_81_257.png)
    



    
![png](README_files/DRP-MLP_81_258.png)
    



    
![png](README_files/DRP-MLP_81_259.png)
    



    
![png](README_files/DRP-MLP_81_260.png)
    



    
![png](README_files/DRP-MLP_81_261.png)
    



    
![png](README_files/DRP-MLP_81_262.png)
    



    
![png](README_files/DRP-MLP_81_263.png)
    



    
![png](README_files/DRP-MLP_81_264.png)
    



    
![png](README_files/DRP-MLP_81_265.png)
    



    
![png](README_files/DRP-MLP_81_266.png)
    



    
![png](README_files/DRP-MLP_81_267.png)
    



    
![png](README_files/DRP-MLP_81_268.png)
    



    
![png](README_files/DRP-MLP_81_269.png)
    



    
![png](README_files/DRP-MLP_81_270.png)
    



    
![png](README_files/DRP-MLP_81_271.png)
    



    
![png](README_files/DRP-MLP_81_272.png)
    



    
![png](README_files/DRP-MLP_81_273.png)
    



    
![png](README_files/DRP-MLP_81_274.png)
    



    
![png](README_files/DRP-MLP_81_275.png)
    



    
![png](README_files/DRP-MLP_81_276.png)
    



    
![png](README_files/DRP-MLP_81_277.png)
    



    
![png](README_files/DRP-MLP_81_278.png)
    



    
![png](README_files/DRP-MLP_81_279.png)
    



    
![png](README_files/DRP-MLP_81_280.png)
    



    
![png](README_files/DRP-MLP_81_281.png)
    



    
![png](README_files/DRP-MLP_81_282.png)
    



    
![png](README_files/DRP-MLP_81_283.png)
    



    
![png](README_files/DRP-MLP_81_284.png)
    



    
![png](README_files/DRP-MLP_81_285.png)
    



    
![png](README_files/DRP-MLP_81_286.png)
    



    
![png](README_files/DRP-MLP_81_287.png)
    



    
![png](README_files/DRP-MLP_81_288.png)
    



    
![png](README_files/DRP-MLP_81_289.png)
    



    
![png](README_files/DRP-MLP_81_290.png)
    



    
![png](README_files/DRP-MLP_81_291.png)
    



    
![png](README_files/DRP-MLP_81_292.png)
    



    
![png](README_files/DRP-MLP_81_293.png)
    



    
![png](README_files/DRP-MLP_81_294.png)
    



    
![png](README_files/DRP-MLP_81_295.png)
    



    
![png](README_files/DRP-MLP_81_296.png)
    



    
![png](README_files/DRP-MLP_81_297.png)
    



    
![png](README_files/DRP-MLP_81_298.png)
    



    
![png](README_files/DRP-MLP_81_299.png)
    



    
![png](README_files/DRP-MLP_81_300.png)
    



    
![png](README_files/DRP-MLP_81_301.png)
    



    
![png](README_files/DRP-MLP_81_302.png)
    



    
![png](README_files/DRP-MLP_81_303.png)
    



    
![png](README_files/DRP-MLP_81_304.png)
    



    
![png](README_files/DRP-MLP_81_305.png)
    



    
![png](README_files/DRP-MLP_81_306.png)
    



    
![png](README_files/DRP-MLP_81_307.png)
    



    
![png](README_files/DRP-MLP_81_308.png)
    



    
![png](README_files/DRP-MLP_81_309.png)
    



    
![png](README_files/DRP-MLP_81_310.png)
    



    
![png](README_files/DRP-MLP_81_311.png)
    



    
![png](README_files/DRP-MLP_81_312.png)
    



    
![png](README_files/DRP-MLP_81_313.png)
    



    
![png](README_files/DRP-MLP_81_314.png)
    



    
![png](README_files/DRP-MLP_81_315.png)
    



    
![png](README_files/DRP-MLP_81_316.png)
    



    
![png](README_files/DRP-MLP_81_317.png)
    



    
![png](README_files/DRP-MLP_81_318.png)
    



    
![png](README_files/DRP-MLP_81_319.png)
    



    
![png](README_files/DRP-MLP_81_320.png)
    



    
![png](README_files/DRP-MLP_81_321.png)
    



    
![png](README_files/DRP-MLP_81_322.png)
    



    
![png](README_files/DRP-MLP_81_323.png)
    



    
![png](README_files/DRP-MLP_81_324.png)
    



    
![png](README_files/DRP-MLP_81_325.png)
    



    
![png](README_files/DRP-MLP_81_326.png)
    



    
![png](README_files/DRP-MLP_81_327.png)
    



    
![png](README_files/DRP-MLP_81_328.png)
    



    
![png](README_files/DRP-MLP_81_329.png)
    



    
![png](README_files/DRP-MLP_81_330.png)
    



    
![png](README_files/DRP-MLP_81_331.png)
    



    
![png](README_files/DRP-MLP_81_332.png)
    



    
![png](README_files/DRP-MLP_81_333.png)
    



    
![png](README_files/DRP-MLP_81_334.png)
    



    
![png](README_files/DRP-MLP_81_335.png)
    



    
![png](README_files/DRP-MLP_81_336.png)
    



    
![png](README_files/DRP-MLP_81_337.png)
    



    
![png](README_files/DRP-MLP_81_338.png)
    



    
![png](README_files/DRP-MLP_81_339.png)
    



    
![png](README_files/DRP-MLP_81_340.png)
    



    
![png](README_files/DRP-MLP_81_341.png)
    



    
![png](README_files/DRP-MLP_81_342.png)
    



    
![png](README_files/DRP-MLP_81_343.png)
    



    
![png](README_files/DRP-MLP_81_344.png)
    



    
![png](README_files/DRP-MLP_81_345.png)
    



    
![png](README_files/DRP-MLP_81_346.png)
    



    
![png](README_files/DRP-MLP_81_347.png)
    



    
![png](README_files/DRP-MLP_81_348.png)
    



    
![png](README_files/DRP-MLP_81_349.png)
    



    
![png](README_files/DRP-MLP_81_350.png)
    



    
![png](README_files/DRP-MLP_81_351.png)
    



    
![png](README_files/DRP-MLP_81_352.png)
    



    
![png](README_files/DRP-MLP_81_353.png)
    



    
![png](README_files/DRP-MLP_81_354.png)
    



    
![png](README_files/DRP-MLP_81_355.png)
    



    
![png](README_files/DRP-MLP_81_356.png)
    



    
![png](README_files/DRP-MLP_81_357.png)
    



    
![png](README_files/DRP-MLP_81_358.png)
    



    
![png](README_files/DRP-MLP_81_359.png)
    



    
![png](README_files/DRP-MLP_81_360.png)
    



    
![png](README_files/DRP-MLP_81_361.png)
    



    
![png](README_files/DRP-MLP_81_362.png)
    



    
![png](README_files/DRP-MLP_81_363.png)
    



    
![png](README_files/DRP-MLP_81_364.png)
    



    
![png](README_files/DRP-MLP_81_365.png)
    



    
![png](README_files/DRP-MLP_81_366.png)
    



    
![png](README_files/DRP-MLP_81_367.png)
    



    
![png](README_files/DRP-MLP_81_368.png)
    



    
![png](README_files/DRP-MLP_81_369.png)
    



    
![png](README_files/DRP-MLP_81_370.png)
    



    
![png](README_files/DRP-MLP_81_371.png)
    



    
![png](README_files/DRP-MLP_81_372.png)
    



    
![png](README_files/DRP-MLP_81_373.png)
    



    
![png](README_files/DRP-MLP_81_374.png)
    



    
![png](README_files/DRP-MLP_81_375.png)
    



    
![png](README_files/DRP-MLP_81_376.png)
    



    
![png](README_files/DRP-MLP_81_377.png)
    



    
![png](README_files/DRP-MLP_81_378.png)
    



    
![png](README_files/DRP-MLP_81_379.png)
    



    
![png](README_files/DRP-MLP_81_380.png)
    



    
![png](README_files/DRP-MLP_81_381.png)
    



    
![png](README_files/DRP-MLP_81_382.png)
    



    
![png](README_files/DRP-MLP_81_383.png)
    



    
![png](README_files/DRP-MLP_81_384.png)
    



    
![png](README_files/DRP-MLP_81_385.png)
    



    
![png](README_files/DRP-MLP_81_386.png)
    



    
![png](README_files/DRP-MLP_81_387.png)
    



    
![png](README_files/DRP-MLP_81_388.png)
    



    
![png](README_files/DRP-MLP_81_389.png)
    


### Encode Categorical Variables for Correlation, Feature Importance Analysis & Modeling


```python
df_eda.dtypes
```




    age                             int64
    gender                            str
    bmi                           float64
    blood_pressure                  int64
    fasting_glucose_level           int64
    insulin_level                 float64
    HbA1c_level                   float64
    cholesterol_level               int64
    triglycerides_level             int64
    physical_activity_level           str
    daily_calorie_intake            int64
    sugar_intake_grams_per_day    float64
    sleep_hours                   float64
    stress_level                    int64
    family_history_diabetes           str
    waist_circumference_cm        float64
    diabetes_risk_score             int64
    bmi_category                      str
    glucose_category                  str
    dtype: object



#### Encode `gender` as Binary Float


```python
df_eda["gender"].value_counts()
```




    gender
    Female    3051
    Male      2949
    Name: count, dtype: int64




```python
df_eda["gender"] = df_eda["gender"].map({"Female": 0, "Male": 1}).astype(float)
df_eda["gender"]
```




    0       0.0
    1       1.0
    2       1.0
    3       0.0
    4       1.0
           ... 
    5995    1.0
    5996    1.0
    5997    1.0
    5998    1.0
    5999    1.0
    Name: gender, Length: 6000, dtype: float64



#### Encode `physical_activity_level` as Ordinal Float


```python
df_eda["physical_activity_level"].value_counts()
```




    physical_activity_level
    Low         3181
    Moderate    1539
    High        1280
    Name: count, dtype: int64




```python
df_eda["physical_activity_level"] = (
    df_eda["physical_activity_level"]
    .map({"Low": 0, "Moderate": 1, "High": 2})
    .astype(float)
)
df_eda["physical_activity_level"]
```




    0       0.0
    1       2.0
    2       0.0
    3       0.0
    4       1.0
           ... 
    5995    2.0
    5996    0.0
    5997    1.0
    5998    2.0
    5999    0.0
    Name: physical_activity_level, Length: 6000, dtype: float64



#### Encode `family_history_diabetes` as Binary Float


```python
df_eda["family_history_diabetes"].value_counts()
```




    family_history_diabetes
    No     4838
    Yes    1162
    Name: count, dtype: int64




```python
df_eda["family_history_diabetes"] = (
    df_eda["family_history_diabetes"].map({"No": 0, "Yes": 1}).astype(float)
)
df_eda["family_history_diabetes"]
```




    0       0.0
    1       0.0
    2       1.0
    3       1.0
    4       1.0
           ... 
    5995    0.0
    5996    1.0
    5997    1.0
    5998    0.0
    5999    0.0
    Name: family_history_diabetes, Length: 6000, dtype: float64



#### Encode `bmi_category` as Ordinal Float


```python
df_eda["bmi_category"].value_counts()
```




    bmi_category
    obese          3731
    overweight     1240
    normal          961
    underweight      68
    Name: count, dtype: int64




```python
df_eda["bmi_category"] = (
    df_eda["bmi_category"]
    .map({"underweight": 0, "normal": 1, "overweight": 2, "obese": 3})
    .astype(float)
)
df_eda["bmi_category"]
```




    0       3.0
    1       1.0
    2       3.0
    3       3.0
    4       3.0
           ... 
    5995    1.0
    5996    3.0
    5997    3.0
    5998    2.0
    5999    3.0
    Name: bmi_category, Length: 6000, dtype: float64



#### Encode `glucose_category` as Ordinal Float


```python
df_eda["glucose_category"].value_counts()
```




    glucose_category
    normal         3518
    prediabetic    1482
    diabetic       1000
    Name: count, dtype: int64




```python
df_eda["glucose_category"] = (
    df_eda["glucose_category"]
    .map({"normal": 0, "prediabetic": 1, "diabetic": 2})
    .astype(float)
)
df_eda["glucose_category"]
```




    0       0.0
    1       0.0
    2       2.0
    3       2.0
    4       0.0
           ... 
    5995    0.0
    5996    1.0
    5997    0.0
    5998    0.0
    5999    2.0
    Name: glucose_category, Length: 6000, dtype: float64



### Verify Data Types Match PyTorch Requirements (float for `X`, int for `y`)


```python
df_eda.dtypes
```




    age                             int64
    gender                        float64
    bmi                           float64
    blood_pressure                  int64
    fasting_glucose_level           int64
    insulin_level                 float64
    HbA1c_level                   float64
    cholesterol_level               int64
    triglycerides_level             int64
    physical_activity_level       float64
    daily_calorie_intake            int64
    sugar_intake_grams_per_day    float64
    sleep_hours                   float64
    stress_level                    int64
    family_history_diabetes       float64
    waist_circumference_cm        float64
    diabetes_risk_score             int64
    bmi_category                  float64
    glucose_category              float64
    dtype: object



### Correlation Plots — Correlation Matrix of All Numerical Variables


```python
fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(
    data=df_eda.corr(),
    vmin=-1,
    vmax=1,
    center=0,
    cmap="coolwarm",
    cbar=True,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 10},
    ax=ax,
)
plt.tight_layout()
plt.show()
```


    
![png](README_files/DRP-MLP_102_0.png)
    


### Correlation Plots — Feature Correlation w/ Target Variable


```python
feature_vars = df_eda.columns.drop(TARGET_VAR)
corr_with_target = (
    df_eda[feature_vars].corrwith(df_eda[TARGET_VAR]).sort_values(ascending=False)
)
fig, ax = plt.subplots(figsize=(10, 8))
colors = ["tab:red" if x >= 0 else "tab:green" for x in corr_with_target]
corr_with_target.plot(kind="barh", ax=ax, color=colors)
ax.set_xlabel(f"Correlation with {TARGET_VAR}")
ax.set_title(f"Feature Correlation with Target ({TARGET_VAR})")
ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
plt.tight_layout()
plt.show()
```


    
![png](README_files/DRP-MLP_104_0.png)
    


### Create `features` Variable for Audit & Modeling


```python
features = [col for col in df_eda.columns if col != TARGET_VAR]
features
```




    ['age',
     'gender',
     'bmi',
     'blood_pressure',
     'fasting_glucose_level',
     'insulin_level',
     'HbA1c_level',
     'cholesterol_level',
     'triglycerides_level',
     'physical_activity_level',
     'daily_calorie_intake',
     'sugar_intake_grams_per_day',
     'sleep_hours',
     'stress_level',
     'family_history_diabetes',
     'waist_circumference_cm',
     'bmi_category',
     'glucose_category']



### Train/Test Split (Pre-Analysis)

Split **before** discriminative score analysis and feature audit to prevent data leakage. Test data must never influence feature selection decisions.


```python
# Stratified split BEFORE analysis to prevent data leakage
df_train, df_test = train_test_split(
    df_eda, test_size=TEST_SIZE, random_state=SEED, stratify=df_eda[TARGET_VAR]
)
print(f"Train: {len(df_train)}, Test: {len(df_test)}")
```

    Train: 4200, Test: 1800


## Discriminative Score Analysis

The **discriminative score** rates how useful a single feature is for separating two groups (label 0 vs label 1). It combines three normalized signals:

$$S = 0.4\,P + 0.4\,E + 0.2\,M$$

| Symbol | Name | Range | What it captures |
|---|---|---|---|
| $P$ | p-value strength | $[0, 1]$ | $\min\!\bigl(1,\;{-\log_{10}(p)}/{10}\bigr)$ — is the difference real? |
| $E$ | effect size | $[0, 1]$ | Cohen's $d$ (continuous) or Cramér's $V$ (categorical), capped at 1 |
| $M$ | mutual information | $[0, 1]$ | Normalized MI between feature and label |

### How to Interpret Each Value (Good vs Bad)

#### P — p-value strength
| P value | Raw p-value | Interpretation |
|---|---|---|
| < 0.05 | p > 0.3 | **Bad.** The difference could easily be random noise. |
| 0.05 - 0.20 | p ~ 0.01 - 0.1 | **Weak.** Some evidence, but not convincing. |
| 0.20 - 0.50 | p ~ 1e-2 to 1e-5 | **Good.** Strong statistical evidence the groups differ. |
| > 0.50 | p < 1e-5 | **Excellent.** Very strong evidence. |

#### E — effect size (Cohen's d or Cramer's V)
| E value | Interpretation |
|---|---|
| < 0.20 | **Negligible.** Groups overlap almost completely. |
| 0.20 - 0.50 | **Small.** Noticeable difference but lots of overlap. |
| 0.50 - 0.80 | **Medium.** Clear practical difference. |
| > 0.80 | **Large.** Groups are well separated. |

#### M — normalized mutual information
| M value | Interpretation |
|---|---|
| < 0.05 | **Bad.** Feature tells you almost nothing about the label. |
| 0.05 - 0.20 | **Weak.** Slight information gain. |
| 0.20 - 0.50 | **Good.** Feature meaningfully reduces uncertainty about the label. |
| > 0.50 | **Excellent.** Feature strongly predicts the label. |

#### S — final discriminative score
| S value | Interpretation |
|---|---|
| < 0.20 | **Drop it.** The feature is not useful for classification. |
| 0.20 - 0.40 | **Weak.** Might help slightly but do not rely on it alone. |
| 0.40 - 0.60 | **Moderate.** A decent feature worth including. |
| 0.60 - 0.80 | **Strong.** Reliable discriminator — prioritize this feature. |
| > 0.80 | **Excellent.** Top-tier feature with high significance, large effect, and high MI. |

### Continuous Features — Two-Sample t-Test

#### Sample Means

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$

#### Sample Variances (Bessel-corrected, ddof=1)

$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

#### Pooled Variance & t-Statistic

**Degrees of freedom:**

$$df = n_0 + n_1 - 2$$

**Pooled variance:**

$$s_p^2 = \frac{(n_0-1)\,s_0^2 + (n_1-1)\,s_1^2}{df}$$

**t-statistic:**

$$t = \frac{\bar{x}_1 - \bar{x}_0}{s_p\,\sqrt{\dfrac{1}{n_0} + \dfrac{1}{n_1}}}$$

#### Cohen's d → Component $E$

$$d = \frac{\bar{x}_1 - \bar{x}_0}{s_p}$$

$$E = \min(1,\;|d|)$$

### Categorical Features — Chi-Square Test

#### Contingency Table & Chi-Square Statistic

**Expected count:**

$$E_{ij} = \frac{(\text{row total}_i)(\text{col total}_j)}{n}$$

**Degrees of freedom:**

$$df = (r-1)(c-1)$$

**Chi-square statistic:**

$$\chi^2 = \sum_{i,j}\frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

#### Cramér's V → Component $E$

$$V = \sqrt{\frac{\chi^2}{n\,(\min(r,c) - 1)}}$$

$$E = \min(1,\;V)$$

### NMI → Component $M$

**Mutual Information:**

$$I(X;Y) = \sum_{x,y} p(x,y)\,\ln\!\frac{p(x,y)}{p(x)\,p(y)}$$

**Entropy:**

$$H(X) = -\sum_x p(x)\,\ln p(x)$$

**Normalized Mutual Information:**

$$\text{NMI} = \frac{I}{\sqrt{H(X) \cdot H(Y)}}$$

$$M = \text{NMI}$$

#### P Component & Final Score

$$P = \min\!\left(1,\;\frac{-\log_{10}(p)}{10}\right)$$

$$S = 0.4\,P + 0.4\,E + 0.2\,M$$

### Reusable Function: `compute_discriminative_score()`

Private helpers (prefixed `_`) are defined above the main function, in the exact order they are called.


```python
def _infer_feature_type(values):
    """
    Infer whether a pandas Series is continuous or categorical.

    Parameters:
        values (pd.Series): Feature column to evaluate.

    Returns:
        str: 'continuous' if numeric dtype, otherwise 'categorical'.
    """
    return "continuous" if pd.api.types.is_numeric_dtype(values) else "categorical"


def _continuous_stats(g0, g1, labels, values):
    """
    Compute p-value, effect size (Cohen's d), and NMI for a continuous feature.

    Parameters:
        g0 (np.ndarray): Feature values for label-0 samples.
        g1 (np.ndarray): Feature values for label-1 samples.
        labels (pd.Series): Full label column.
        values (pd.Series): Full feature column (used for median-split binning).

    Returns:
        tuple: (p_value, cohens_d, nmi).
    """
    n0, n1 = len(g0), len(g1)
    sp2 = ((n0 - 1) * g0.var(ddof=1) + (n1 - 1) * g1.var(ddof=1)) / (n0 + n1 - 2)
    sp = np.sqrt(sp2)
    _, p = stats.ttest_ind(g0, g1, equal_var=True)
    d = abs(g1.mean() - g0.mean()) / sp if sp > 0 else 0.0
    binned = (values > values.median()).astype(int)
    nmi = normalized_mutual_info_score(labels, binned)
    return p, d, nmi


def _categorical_stats(labels, values, n):
    """
    Compute p-value, effect size (Cramer's V), and NMI for a categorical feature.

    Parameters:
        labels (pd.Series): Binary label column.
        values (pd.Series): Categorical feature column.
        n (int): Total number of samples.

    Returns:
        tuple: (p_value, cramers_v, nmi).
    """
    ct = pd.crosstab(labels, values)
    chi2, p, _, _ = stats.chi2_contingency(ct, correction=False)
    k = min(ct.shape)
    v = np.sqrt(chi2 / (n * max(k - 1, 1)))
    nmi = normalized_mutual_info_score(labels, values)
    return p, v, nmi


def _compute_raw_stats(feature_type, labels, values, n):
    """
    Dispatch to the correct statistical test based on feature type.

    Parameters:
        feature_type (str): 'continuous' or 'categorical'.
        labels (pd.Series): Binary label column.
        values (pd.Series): Feature column.
        n (int): Total number of samples.

    Returns:
        tuple: (p_value, effect_size, nmi).
    """
    if feature_type == "continuous":
        g0 = values[labels == 0].dropna().values
        g1 = values[labels == 1].dropna().values
        return _continuous_stats(g0, g1, labels, values)
    return _categorical_stats(labels, values, n)


def _score_components(p_value, effect_size, nmi):
    """
    Normalize raw statistics into the P, E, M components and compute S.

    Parameters:
        p_value (float): Raw p-value from the statistical test.
        effect_size (float): Raw effect size (Cohen's d or Cramer's V).
        nmi (float): Normalized mutual information.

    Returns:
        dict: Keys: p_value, effect_size, nmi, P, E, M, S.
    """
    P = min(1.0, -np.log10(max(p_value, 1e-300)) / 10)
    E = min(1.0, effect_size)
    M = nmi
    S = 0.4 * P + 0.4 * E + 0.2 * M
    return {
        "p_value": p_value,
        "effect_size": effect_size,
        "nmi": nmi,
        "P": P,
        "E": E,
        "M": M,
        "S": S,
    }


def compute_discriminative_score(df, feature_col, label_col, feature_type="auto"):
    """
    Compute the discriminative score for a single feature.

    Parameters:
        df (pd.DataFrame): Input dataframe containing feature and label columns.
        feature_col (str): Name of the feature column to evaluate.
        label_col (str): Name of the binary label column (0 or 1).
        feature_type (str): 'continuous', 'categorical', or 'auto' (default: 'auto').

    Returns:
        dict: Keys: p_value, effect_size, nmi, P, E, M, S.
    """
    labels, values = df[label_col], df[feature_col]
    if feature_type == "auto":
        feature_type = _infer_feature_type(values)
    p, effect, nmi = _compute_raw_stats(feature_type, labels, values, len(df))
    return _score_components(p, effect, nmi)
```

### Apply Discriminative Score to Diabetes Features (Binary Classification)

Since this is a binary classification task (Low Risk vs High Risk), we compute discriminative scores directly against the binary target.


```python
# Define feature types for discriminative score computation
feature_types = {col: "continuous" for col in CONTINUOUS_FEATURES}
for cat in CATEGORICAL_FEATURES:
    feature_types[cat] = "categorical"

# Compute discriminative scores for each feature vs binary target
disc_results = []
for feat in features:
    score = compute_discriminative_score(
        df_train, feat, TARGET_VAR, feature_type=feature_types.get(feat, "auto")
    )
    disc_results.append({"Feature": feat, **score})
disc_df = pd.DataFrame(disc_results)
disc_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature</th>
      <th>p_value</th>
      <th>effect_size</th>
      <th>nmi</th>
      <th>P</th>
      <th>E</th>
      <th>M</th>
      <th>S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>age</td>
      <td>3.505463e-35</td>
      <td>0.385479</td>
      <td>0.018347</td>
      <td>1.000000</td>
      <td>0.385479</td>
      <td>0.018347</td>
      <td>0.557861</td>
    </tr>
    <tr>
      <th>1</th>
      <td>gender</td>
      <td>2.285568e-01</td>
      <td>0.018579</td>
      <td>0.000249</td>
      <td>0.064101</td>
      <td>0.018579</td>
      <td>0.000249</td>
      <td>0.033122</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bmi</td>
      <td>0.000000e+00</td>
      <td>2.470267</td>
      <td>0.509601</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.509601</td>
      <td>0.901920</td>
    </tr>
    <tr>
      <th>3</th>
      <td>blood_pressure</td>
      <td>0.000000e+00</td>
      <td>1.486729</td>
      <td>0.211505</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.211505</td>
      <td>0.842301</td>
    </tr>
    <tr>
      <th>4</th>
      <td>fasting_glucose_level</td>
      <td>0.000000e+00</td>
      <td>1.318728</td>
      <td>0.277136</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.277136</td>
      <td>0.855427</td>
    </tr>
    <tr>
      <th>5</th>
      <td>insulin_level</td>
      <td>0.000000e+00</td>
      <td>1.292912</td>
      <td>0.179430</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.179430</td>
      <td>0.835886</td>
    </tr>
    <tr>
      <th>6</th>
      <td>HbA1c_level</td>
      <td>0.000000e+00</td>
      <td>1.283916</td>
      <td>0.231579</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.231579</td>
      <td>0.846316</td>
    </tr>
    <tr>
      <th>7</th>
      <td>cholesterol_level</td>
      <td>0.000000e+00</td>
      <td>1.341652</td>
      <td>0.176183</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.176183</td>
      <td>0.835237</td>
    </tr>
    <tr>
      <th>8</th>
      <td>triglycerides_level</td>
      <td>0.000000e+00</td>
      <td>1.857744</td>
      <td>0.325686</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.325686</td>
      <td>0.865137</td>
    </tr>
    <tr>
      <th>9</th>
      <td>physical_activity_level</td>
      <td>0.000000e+00</td>
      <td>0.720229</td>
      <td>0.363297</td>
      <td>1.000000</td>
      <td>0.720229</td>
      <td>0.363297</td>
      <td>0.760751</td>
    </tr>
    <tr>
      <th>10</th>
      <td>daily_calorie_intake</td>
      <td>0.000000e+00</td>
      <td>1.762654</td>
      <td>0.337417</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.337417</td>
      <td>0.867483</td>
    </tr>
    <tr>
      <th>11</th>
      <td>sugar_intake_grams_per_day</td>
      <td>0.000000e+00</td>
      <td>1.394056</td>
      <td>0.218683</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.218683</td>
      <td>0.843737</td>
    </tr>
    <tr>
      <th>12</th>
      <td>sleep_hours</td>
      <td>8.927771e-109</td>
      <td>0.704311</td>
      <td>0.050860</td>
      <td>1.000000</td>
      <td>0.704311</td>
      <td>0.050860</td>
      <td>0.691897</td>
    </tr>
    <tr>
      <th>13</th>
      <td>stress_level</td>
      <td>1.141087e-166</td>
      <td>0.888973</td>
      <td>0.107488</td>
      <td>1.000000</td>
      <td>0.888973</td>
      <td>0.107488</td>
      <td>0.777087</td>
    </tr>
    <tr>
      <th>14</th>
      <td>family_history_diabetes</td>
      <td>7.517269e-07</td>
      <td>0.076342</td>
      <td>0.004944</td>
      <td>0.612394</td>
      <td>0.076342</td>
      <td>0.004944</td>
      <td>0.276483</td>
    </tr>
    <tr>
      <th>15</th>
      <td>waist_circumference_cm</td>
      <td>0.000000e+00</td>
      <td>2.487799</td>
      <td>0.478215</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.478215</td>
      <td>0.895643</td>
    </tr>
    <tr>
      <th>16</th>
      <td>bmi_category</td>
      <td>0.000000e+00</td>
      <td>0.720426</td>
      <td>0.371793</td>
      <td>1.000000</td>
      <td>0.720426</td>
      <td>0.371793</td>
      <td>0.762529</td>
    </tr>
    <tr>
      <th>17</th>
      <td>glucose_category</td>
      <td>0.000000e+00</td>
      <td>0.619778</td>
      <td>0.274772</td>
      <td>1.000000</td>
      <td>0.619778</td>
      <td>0.274772</td>
      <td>0.702866</td>
    </tr>
  </tbody>
</table>
</div>



### Discriminative Score Visualization — Feature Importance


```python
# Discriminative score per feature
avg_scores = disc_df.set_index("Feature")["S"].sort_values(ascending=False)

# Plot discriminative scores
fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#4C72B0" if s >= 0.4 else "#DD8452" for s in avg_scores]
avg_scores.plot(kind="barh", ax=ax, color=colors)
ax.set_xlabel("Discriminative Score (S)")
ax.set_title("Discriminative Score: Feature Importance (Binary Classification)")
ax.axvline(
    x=0.4, color="red", linestyle="--", linewidth=0.8, label="Moderate Threshold"
)
ax.legend()
plt.tight_layout()
plt.show()
```


    
![png](README_files/DRP-MLP_119_0.png)
    


### Discriminative Score Visualization — Component Breakdown


```python
# Component breakdown: P, E, M for each feature
fig, ax = plt.subplots(figsize=(14, 6))
disc_df.set_index("Feature")[["P", "E", "M"]].plot(kind="bar", ax=ax, ylim=(0, 1.15))
ax.set_title("Discriminative Score Components: P, E, M")
ax.set_ylabel("Value")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()
```


    
![png](README_files/DRP-MLP_121_0.png)
    


## Feature Selection Audit: Discriminative Score & Correlation


```python
def audit_reason(s):
    """
    Return a human-readable reason string based on discriminative score.

    Parameters:
        s (float): Average discriminative score.

    Returns:
        str: Interpretation of the score.
    """
    if s >= 0.8:
        return "Excellent — top-tier discriminator"
    if s >= 0.6:
        return "Strong — reliable discriminator"
    if s >= 0.4:
        return "Moderate — decent feature"
    if s >= 0.2:
        return "Weak — marginal value"
    return "Drop — not useful for classification"
```


```python
all_corrs = df_train[features].corrwith(df_train[TARGET_VAR])
results = []
for col in features:
    s = avg_scores[col]
    corr = all_corrs[col]
    recommendation = "✅ KEEP" if s >= 0.4 else "❌ DROP"
    reason = audit_reason(s)
    results.append(
        {
            "Feature": col,
            "Avg_S": round(s, 4),
            "Correlation": round(corr, 4),
            "Recommendation": recommendation,
            "Reason": reason,
        }
    )
audit_df = (
    pd.DataFrame(results).sort_values("Avg_S", ascending=False).reset_index(drop=True)
)
audit_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feature</th>
      <th>Avg_S</th>
      <th>Correlation</th>
      <th>Recommendation</th>
      <th>Reason</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bmi</td>
      <td>0.9019</td>
      <td>0.7773</td>
      <td>✅ KEEP</td>
      <td>Excellent — top-tier discriminator</td>
    </tr>
    <tr>
      <th>1</th>
      <td>waist_circumference_cm</td>
      <td>0.8956</td>
      <td>0.7794</td>
      <td>✅ KEEP</td>
      <td>Excellent — top-tier discriminator</td>
    </tr>
    <tr>
      <th>2</th>
      <td>daily_calorie_intake</td>
      <td>0.8675</td>
      <td>0.6613</td>
      <td>✅ KEEP</td>
      <td>Excellent — top-tier discriminator</td>
    </tr>
    <tr>
      <th>3</th>
      <td>triglycerides_level</td>
      <td>0.8651</td>
      <td>0.6807</td>
      <td>✅ KEEP</td>
      <td>Excellent — top-tier discriminator</td>
    </tr>
    <tr>
      <th>4</th>
      <td>fasting_glucose_level</td>
      <td>0.8554</td>
      <td>0.5506</td>
      <td>✅ KEEP</td>
      <td>Excellent — top-tier discriminator</td>
    </tr>
    <tr>
      <th>5</th>
      <td>HbA1c_level</td>
      <td>0.8463</td>
      <td>0.5403</td>
      <td>✅ KEEP</td>
      <td>Excellent — top-tier discriminator</td>
    </tr>
    <tr>
      <th>6</th>
      <td>sugar_intake_grams_per_day</td>
      <td>0.8437</td>
      <td>0.5719</td>
      <td>✅ KEEP</td>
      <td>Excellent — top-tier discriminator</td>
    </tr>
    <tr>
      <th>7</th>
      <td>blood_pressure</td>
      <td>0.8423</td>
      <td>0.5967</td>
      <td>✅ KEEP</td>
      <td>Excellent — top-tier discriminator</td>
    </tr>
    <tr>
      <th>8</th>
      <td>insulin_level</td>
      <td>0.8359</td>
      <td>0.5430</td>
      <td>✅ KEEP</td>
      <td>Excellent — top-tier discriminator</td>
    </tr>
    <tr>
      <th>9</th>
      <td>cholesterol_level</td>
      <td>0.8352</td>
      <td>0.5572</td>
      <td>✅ KEEP</td>
      <td>Excellent — top-tier discriminator</td>
    </tr>
    <tr>
      <th>10</th>
      <td>stress_level</td>
      <td>0.7771</td>
      <td>0.4063</td>
      <td>✅ KEEP</td>
      <td>Strong — reliable discriminator</td>
    </tr>
    <tr>
      <th>11</th>
      <td>bmi_category</td>
      <td>0.7625</td>
      <td>0.6623</td>
      <td>✅ KEEP</td>
      <td>Strong — reliable discriminator</td>
    </tr>
    <tr>
      <th>12</th>
      <td>physical_activity_level</td>
      <td>0.7608</td>
      <td>-0.7028</td>
      <td>✅ KEEP</td>
      <td>Strong — reliable discriminator</td>
    </tr>
    <tr>
      <th>13</th>
      <td>glucose_category</td>
      <td>0.7029</td>
      <td>0.6085</td>
      <td>✅ KEEP</td>
      <td>Strong — reliable discriminator</td>
    </tr>
    <tr>
      <th>14</th>
      <td>sleep_hours</td>
      <td>0.6919</td>
      <td>-0.3322</td>
      <td>✅ KEEP</td>
      <td>Strong — reliable discriminator</td>
    </tr>
    <tr>
      <th>15</th>
      <td>age</td>
      <td>0.5579</td>
      <td>0.1893</td>
      <td>✅ KEEP</td>
      <td>Moderate — decent feature</td>
    </tr>
    <tr>
      <th>16</th>
      <td>family_history_diabetes</td>
      <td>0.2765</td>
      <td>0.0763</td>
      <td>❌ DROP</td>
      <td>Weak — marginal value</td>
    </tr>
    <tr>
      <th>17</th>
      <td>gender</td>
      <td>0.0331</td>
      <td>-0.0186</td>
      <td>❌ DROP</td>
      <td>Drop — not useful for classification</td>
    </tr>
  </tbody>
</table>
</div>



## Preprocessing

### Create `df_preprocessed` Dataset


```python
df_preprocessed = df_eda.copy()
df_preprocessed.shape
```




    (6000, 19)



### Drop Low-Correlation Features

#### Based on Recommendations, Drop Features w/ Low-Correlation


```python
df_preprocessed.drop(columns=[
    "gender",
    "family_history_diabetes",
], inplace=True)
df_preprocessed
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>bmi</th>
      <th>blood_pressure</th>
      <th>fasting_glucose_level</th>
      <th>insulin_level</th>
      <th>HbA1c_level</th>
      <th>cholesterol_level</th>
      <th>triglycerides_level</th>
      <th>physical_activity_level</th>
      <th>daily_calorie_intake</th>
      <th>sugar_intake_grams_per_day</th>
      <th>sleep_hours</th>
      <th>stress_level</th>
      <th>waist_circumference_cm</th>
      <th>diabetes_risk_score</th>
      <th>bmi_category</th>
      <th>glucose_category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>77</td>
      <td>33.8</td>
      <td>154</td>
      <td>93</td>
      <td>12.1</td>
      <td>5.2</td>
      <td>242</td>
      <td>194</td>
      <td>0.0</td>
      <td>2169</td>
      <td>78.4</td>
      <td>8.1</td>
      <td>4</td>
      <td>101.1</td>
      <td>1</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>54</td>
      <td>19.2</td>
      <td>123</td>
      <td>94</td>
      <td>4.6</td>
      <td>5.4</td>
      <td>212</td>
      <td>76</td>
      <td>2.0</td>
      <td>1881</td>
      <td>16.5</td>
      <td>6.6</td>
      <td>3</td>
      <td>60.0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25</td>
      <td>33.7</td>
      <td>141</td>
      <td>150</td>
      <td>10.8</td>
      <td>6.9</td>
      <td>247</td>
      <td>221</td>
      <td>0.0</td>
      <td>2811</td>
      <td>147.9</td>
      <td>6.7</td>
      <td>10</td>
      <td>114.7</td>
      <td>1</td>
      <td>3.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23</td>
      <td>32.8</td>
      <td>140</td>
      <td>145</td>
      <td>11.6</td>
      <td>6.8</td>
      <td>195</td>
      <td>193</td>
      <td>0.0</td>
      <td>2826</td>
      <td>98.3</td>
      <td>4.4</td>
      <td>9</td>
      <td>96.6</td>
      <td>1</td>
      <td>3.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>70</td>
      <td>33.7</td>
      <td>165</td>
      <td>90</td>
      <td>18.3</td>
      <td>5.6</td>
      <td>217</td>
      <td>170</td>
      <td>1.0</td>
      <td>2610</td>
      <td>65.8</td>
      <td>9.1</td>
      <td>5</td>
      <td>107.4</td>
      <td>1</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5995</th>
      <td>58</td>
      <td>21.8</td>
      <td>158</td>
      <td>89</td>
      <td>6.3</td>
      <td>5.3</td>
      <td>198</td>
      <td>132</td>
      <td>2.0</td>
      <td>1995</td>
      <td>44.1</td>
      <td>7.6</td>
      <td>7</td>
      <td>82.7</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5996</th>
      <td>73</td>
      <td>47.3</td>
      <td>193</td>
      <td>109</td>
      <td>33.7</td>
      <td>6.1</td>
      <td>250</td>
      <td>209</td>
      <td>0.0</td>
      <td>3401</td>
      <td>61.4</td>
      <td>5.3</td>
      <td>10</td>
      <td>150.0</td>
      <td>1</td>
      <td>3.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5997</th>
      <td>35</td>
      <td>31.0</td>
      <td>139</td>
      <td>90</td>
      <td>15.1</td>
      <td>5.3</td>
      <td>190</td>
      <td>164</td>
      <td>1.0</td>
      <td>3022</td>
      <td>86.7</td>
      <td>6.8</td>
      <td>3</td>
      <td>102.7</td>
      <td>0</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5998</th>
      <td>58</td>
      <td>26.2</td>
      <td>136</td>
      <td>88</td>
      <td>6.3</td>
      <td>5.5</td>
      <td>223</td>
      <td>126</td>
      <td>2.0</td>
      <td>2311</td>
      <td>28.2</td>
      <td>8.4</td>
      <td>5</td>
      <td>90.6</td>
      <td>0</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5999</th>
      <td>54</td>
      <td>39.3</td>
      <td>158</td>
      <td>201</td>
      <td>10.7</td>
      <td>8.3</td>
      <td>240</td>
      <td>223</td>
      <td>0.0</td>
      <td>2442</td>
      <td>113.1</td>
      <td>5.7</td>
      <td>6</td>
      <td>123.8</td>
      <td>1</td>
      <td>3.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
<p>6000 rows × 17 columns</p>
</div>



### Save `df_preprocessed` Processed Dataset


```python
df_preprocessed.to_csv("drp_dataset_preprocessed.csv", index=False)
```

### Load `df_preprocessed` Processed Dataset


```python
df_preprocessed = pd.read_csv("drp_dataset_preprocessed.csv")
df_preprocessed
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>bmi</th>
      <th>blood_pressure</th>
      <th>fasting_glucose_level</th>
      <th>insulin_level</th>
      <th>HbA1c_level</th>
      <th>cholesterol_level</th>
      <th>triglycerides_level</th>
      <th>physical_activity_level</th>
      <th>daily_calorie_intake</th>
      <th>sugar_intake_grams_per_day</th>
      <th>sleep_hours</th>
      <th>stress_level</th>
      <th>waist_circumference_cm</th>
      <th>diabetes_risk_score</th>
      <th>bmi_category</th>
      <th>glucose_category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>77</td>
      <td>33.8</td>
      <td>154</td>
      <td>93</td>
      <td>12.1</td>
      <td>5.2</td>
      <td>242</td>
      <td>194</td>
      <td>0.0</td>
      <td>2169</td>
      <td>78.4</td>
      <td>8.1</td>
      <td>4</td>
      <td>101.1</td>
      <td>1</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>54</td>
      <td>19.2</td>
      <td>123</td>
      <td>94</td>
      <td>4.6</td>
      <td>5.4</td>
      <td>212</td>
      <td>76</td>
      <td>2.0</td>
      <td>1881</td>
      <td>16.5</td>
      <td>6.6</td>
      <td>3</td>
      <td>60.0</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25</td>
      <td>33.7</td>
      <td>141</td>
      <td>150</td>
      <td>10.8</td>
      <td>6.9</td>
      <td>247</td>
      <td>221</td>
      <td>0.0</td>
      <td>2811</td>
      <td>147.9</td>
      <td>6.7</td>
      <td>10</td>
      <td>114.7</td>
      <td>1</td>
      <td>3.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23</td>
      <td>32.8</td>
      <td>140</td>
      <td>145</td>
      <td>11.6</td>
      <td>6.8</td>
      <td>195</td>
      <td>193</td>
      <td>0.0</td>
      <td>2826</td>
      <td>98.3</td>
      <td>4.4</td>
      <td>9</td>
      <td>96.6</td>
      <td>1</td>
      <td>3.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>70</td>
      <td>33.7</td>
      <td>165</td>
      <td>90</td>
      <td>18.3</td>
      <td>5.6</td>
      <td>217</td>
      <td>170</td>
      <td>1.0</td>
      <td>2610</td>
      <td>65.8</td>
      <td>9.1</td>
      <td>5</td>
      <td>107.4</td>
      <td>1</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5995</th>
      <td>58</td>
      <td>21.8</td>
      <td>158</td>
      <td>89</td>
      <td>6.3</td>
      <td>5.3</td>
      <td>198</td>
      <td>132</td>
      <td>2.0</td>
      <td>1995</td>
      <td>44.1</td>
      <td>7.6</td>
      <td>7</td>
      <td>82.7</td>
      <td>0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5996</th>
      <td>73</td>
      <td>47.3</td>
      <td>193</td>
      <td>109</td>
      <td>33.7</td>
      <td>6.1</td>
      <td>250</td>
      <td>209</td>
      <td>0.0</td>
      <td>3401</td>
      <td>61.4</td>
      <td>5.3</td>
      <td>10</td>
      <td>150.0</td>
      <td>1</td>
      <td>3.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5997</th>
      <td>35</td>
      <td>31.0</td>
      <td>139</td>
      <td>90</td>
      <td>15.1</td>
      <td>5.3</td>
      <td>190</td>
      <td>164</td>
      <td>1.0</td>
      <td>3022</td>
      <td>86.7</td>
      <td>6.8</td>
      <td>3</td>
      <td>102.7</td>
      <td>0</td>
      <td>3.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5998</th>
      <td>58</td>
      <td>26.2</td>
      <td>136</td>
      <td>88</td>
      <td>6.3</td>
      <td>5.5</td>
      <td>223</td>
      <td>126</td>
      <td>2.0</td>
      <td>2311</td>
      <td>28.2</td>
      <td>8.4</td>
      <td>5</td>
      <td>90.6</td>
      <td>0</td>
      <td>2.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5999</th>
      <td>54</td>
      <td>39.3</td>
      <td>158</td>
      <td>201</td>
      <td>10.7</td>
      <td>8.3</td>
      <td>240</td>
      <td>223</td>
      <td>0.0</td>
      <td>2442</td>
      <td>113.1</td>
      <td>5.7</td>
      <td>6</td>
      <td>123.8</td>
      <td>1</td>
      <td>3.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
<p>6000 rows × 17 columns</p>
</div>



## Reusable ML Pipeline

### Model Architecture


```python
class DiabetesMLP(nn.Module):
    """A feedforward neural network with two hidden layers and optional dropout."""

    def __init__(self, in_features, h1=32, h2=16, out_features=2, dropout=0.3):
        """
        Initialize the neural network layers.

        Parameters:
            in_features (int): Number of input features.
            h1 (int): Neurons in hidden layer 1.
            h2 (int): Neurons in hidden layer 2.
            out_features (int): Number of output classes.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Define the forward pass.

        Parameters:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output logits.
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.out(x)
```

### Dataset Class


```python
class NumpyDataset(torch.utils.data.Dataset):
    """Memory-efficient dataset converting numpy arrays to tensors on demand."""

    def __init__(self, X, y, device):
        """
        Initialize with numpy arrays.

        Parameters:
            X (np.ndarray): Feature array.
            y (np.ndarray): Label array.
            device (torch.device): Target device.
        """
        self.X = X
        self.y = y
        self.device = device

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """
        Return a single sample as tensors on the specified device.

        Parameters:
            idx (int): Sample index.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Feature and label tensors.
        """
        X_tensor = torch.tensor(self.X[idx]).float().to(self.device)
        y_tensor = torch.tensor(self.y[idx]).long().to(self.device)
        return X_tensor, y_tensor
```

### Data Preparation Functions


```python
def _extract_arrays(df, feature_cols, target_col):
    """
    Extract feature and target arrays from a DataFrame.

    Parameters:
        df (pd.DataFrame): Source dataframe.
        feature_cols (list): Feature column names.
        target_col (str): Target column name.

    Returns:
        tuple: (X, y) as numpy arrays.
    """
    X = df[feature_cols].values.astype("float32")
    y = df[target_col].values.astype("int64")
    return X, y


def _fit_scaler(X_train, feature_cols):
    """
    Fit StandardScaler on training features.

    Parameters:
        X_train (np.ndarray): Training feature array.
        feature_cols (list): Feature column names.

    Returns:
        tuple: (scaled_DataFrame, fitted_scaler).
    """
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
    return X_scaled, scaler


def _scale_array(X, scaler, feature_cols):
    """
    Transform a feature array with a fitted scaler.

    Parameters:
        X (np.ndarray): Feature array to transform.
        scaler (StandardScaler): Fitted scaler.
        feature_cols (list): Feature column names.

    Returns:
        pd.DataFrame: Scaled features.
    """
    return pd.DataFrame(scaler.transform(X), columns=feature_cols)


def _split_indices(n, size, seed, stratify):
    """
    Return stratified train/validation index arrays.

    Parameters:
        n (int): Total number of samples.
        size (float): Fraction for the validation set.
        seed (int): Random seed.
        stratify (np.ndarray): Labels for stratification.

    Returns:
        tuple: (train_indices, val_indices).
    """
    return train_test_split(
        np.arange(n), test_size=size, random_state=seed, stratify=stratify
    )


def prepare_data(df_train, df_test, feature_cols, target_col, val_size, seed):
    """
    Sub-split train into train/val, scale all three sets, return arrays + scaler.

    Parameters:
        df_train (pd.DataFrame): Training dataframe (from early split).
        df_test (pd.DataFrame): Test dataframe (from early split).
        feature_cols (list): Feature column names.
        target_col (str): Target column name.
        val_size (float): Fraction of training data for validation.
        seed (int): Random seed.

    Returns:
        tuple: (X_tr, X_val, X_te, y_tr, y_val, y_te, scaler).
    """
    X_all, y_all = _extract_arrays(df_train, feature_cols, target_col)
    X_te_raw, y_te = _extract_arrays(df_test, feature_cols, target_col)
    tr_idx, val_idx = _split_indices(len(X_all), val_size, seed, y_all)
    X_tr, scaler = _fit_scaler(X_all[tr_idx], feature_cols)
    X_val = _scale_array(X_all[val_idx], scaler, feature_cols)
    X_te = _scale_array(X_te_raw, scaler, feature_cols)
    return X_tr, X_val, X_te, y_all[tr_idx], y_all[val_idx], y_te, scaler


def _compute_class_weights(y_train, device):
    """
    Compute inverse-frequency class weights for imbalanced datasets.

    Parameters:
        y_train (np.ndarray): Training labels.
        device (torch.device): Target device for the weight tensor.

    Returns:
        torch.Tensor: Class weight tensor (balanced datasets get ~equal weights).
    """
    counts = np.bincount(y_train)
    weights = len(y_train) / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32).to(device)


def _make_loader(X, y, device, batch_size, shuffle):
    """
    Create a DataLoader from a feature DataFrame and label array.

    Parameters:
        X (pd.DataFrame): Scaled feature DataFrame.
        y (np.ndarray): Label array.
        device (torch.device): Target device.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle.

    Returns:
        DataLoader: Configured data loader.
    """
    return DataLoader(
        NumpyDataset(X.values, y, device), batch_size=batch_size, shuffle=shuffle
    )


def create_loaders(X_tr, X_val, X_te, y_tr, y_val, y_te, device, batch_size):
    """
    Create train, validation, and test DataLoaders.

    Parameters:
        X_tr (pd.DataFrame): Scaled training features.
        X_val (pd.DataFrame): Scaled validation features.
        X_te (pd.DataFrame): Scaled test features.
        y_tr (np.ndarray): Training labels.
        y_val (np.ndarray): Validation labels.
        y_te (np.ndarray): Test labels.
        device (torch.device): Target device.
        batch_size (int): Batch size.

    Returns:
        tuple: (train_loader, val_loader, test_loader).
    """
    train_ldr = _make_loader(X_tr, y_tr, device, batch_size, shuffle=True)
    val_ldr = _make_loader(X_val, y_val, device, batch_size, shuffle=False)
    test_ldr = _make_loader(X_te, y_te, device, batch_size, shuffle=False)
    return train_ldr, val_ldr, test_ldr
```

### Training Functions


```python
def _train_batch(model, X_batch, y_batch, criterion, optimizer):
    """
    Process one training batch and return batch loss.

    Parameters:
        model (nn.Module): Model being trained.
        X_batch (torch.Tensor): Input batch.
        y_batch (torch.Tensor): Label batch.
        criterion: Loss function.
        optimizer: Optimizer.

    Returns:
        float: Batch loss value.
    """
    optimizer.zero_grad()
    loss = criterion(model(X_batch), y_batch)
    loss.backward()
    optimizer.step()
    return loss.item()


def _run_train_epoch(model, loader, criterion, optimizer):
    """
    Run one full training epoch and return average loss.

    Parameters:
        model (nn.Module): Model being trained.
        loader (DataLoader): Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        total_loss += _train_batch(model, X_batch, y_batch, criterion, optimizer)
    return total_loss / len(loader)


def _run_val_epoch(model, loader, criterion):
    """
    Run one full validation epoch and return average loss.

    Parameters:
        model (nn.Module): Model being evaluated.
        loader (DataLoader): Validation data loader.
        criterion: Loss function.

    Returns:
        float: Average validation loss for the epoch.
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            total_loss += criterion(model(X_batch), y_batch).item()
    return total_loss / len(loader)


def _save_if_improved(model, val_loss, best_val_loss, path, counter):
    """
    Save model if val loss improved; return (best_loss, patience_counter).

    Parameters:
        model (nn.Module): Model to potentially save.
        val_loss (float): Current validation loss.
        best_val_loss (float): Best validation loss so far.
        path (str): File path for saving weights.
        counter (int): Current patience counter.

    Returns:
        tuple: (updated_best_loss, updated_patience_counter).
    """
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), path)
        return val_loss, 0
    return best_val_loss, counter + 1


def _check_early_stop(counter, patience, epoch):
    """
    Print message and return True if patience is exhausted.

    Parameters:
        counter (int): Epochs without improvement.
        patience (int): Maximum allowed patience.
        epoch (int): Current epoch number.

    Returns:
        bool: True if training should stop.
    """
    if counter >= patience:
        print(f"Early stopping at epoch {epoch} (patience={patience})")
        return True
    return False


def _log_progress(epoch, epochs, train_loss, val_loss, interval):
    """
    Print epoch progress at specified intervals.

    Parameters:
        epoch (int): Current epoch number.
        epochs (int): Total number of epochs.
        train_loss (float): Training loss for this epoch.
        val_loss (float): Validation loss for this epoch.
        interval (int): Logging interval.
    """
    if epoch % interval == 0:
        print(f"Epoch {epoch}/{epochs} - Train: {train_loss:.4f} - Val: {val_loss:.4f}")


def train_model(model, train_loader, val_loader, criterion, optimizer, config):
    """
    Full training loop with validation, checkpointing, early stopping, and logging.

    Parameters:
        model (nn.Module): Model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        config (dict): Keys: epochs, log_interval, save_path, patience.

    Returns:
        tuple: (train_losses, val_losses, best_val_loss).
    """
    train_losses, val_losses, best, ctr = [], [], float("inf"), 0
    for epoch in range(config["epochs"]):
        train_losses.append(_run_train_epoch(model, train_loader, criterion, optimizer))
        val_losses.append(_run_val_epoch(model, val_loader, criterion))
        best, ctr = _save_if_improved(
            model, val_losses[-1], best, config["save_path"], ctr
        )
        _log_progress(
            epoch,
            config["epochs"],
            train_losses[-1],
            val_losses[-1],
            config["log_interval"],
        )
        if _check_early_stop(ctr, config["patience"], epoch):
            break
    return train_losses, val_losses, best
```

### Evaluation Functions


```python
def _accumulate_batch(y_true, y_pred, y_probs, logits, y_batch):
    """
    Accumulate predictions and probabilities from a single batch.

    Parameters:
        y_true (list): Accumulator for true labels.
        y_pred (list): Accumulator for predicted labels.
        y_probs (list): Accumulator for class probabilities.
        logits (torch.Tensor): Raw model output.
        y_batch (torch.Tensor): True label batch.
    """
    y_true.extend(y_batch.cpu().numpy())
    y_pred.extend(logits.argmax(dim=1).cpu().numpy())
    y_probs.extend(F.softmax(logits, dim=1).cpu().numpy())


def collect_predictions(model, loader):
    """
    Collect all predictions, true labels, and probabilities from a loader.

    Parameters:
        model (nn.Module): Trained model.
        loader (DataLoader): Data loader to iterate over.

    Returns:
        tuple: (y_true, y_pred, y_probs) as numpy arrays.
    """
    y_true, y_pred, y_probs = [], [], []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            _accumulate_batch(y_true, y_pred, y_probs, model(X_batch), y_batch)
    return np.array(y_true), np.array(y_pred), np.array(y_probs)
```

### Visualization Functions


```python
def _draw_pie(ax, accuracy, title):
    """
    Draw a single accuracy pie chart on the given axis.

    Parameters:
        ax (matplotlib.axes.Axes): Target axis.
        accuracy (float): Accuracy value (0 to 1).
        title (str): Chart title.
    """
    colors = ["#00FF00", "#FF0000"]
    ax.pie(
        [accuracy, 1 - accuracy],
        labels=["Correct", "Incorrect"],
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        explode=(0.1, 0),
    )
    ax.set_title(title)


def plot_accuracy_pies(train_acc, test_acc):
    """
    Display training and test accuracy as side-by-side pie charts.

    Parameters:
        train_acc (float): Training accuracy.
        test_acc (float): Test accuracy.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    _draw_pie(ax1, train_acc, "Training Accuracy")
    _draw_pie(ax2, test_acc, "Test Accuracy")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Display confusion matrix as a heatmap.

    Parameters:
        y_true (array): True labels.
        y_pred (array): Predicted labels.
        class_names (list): Human-readable class names.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Confusion Matrix")
    plt.show()


def plot_loss_curves(train_losses, val_losses):
    """
    Plot training and validation loss curves over epochs.

    Parameters:
        train_losses (list): Training loss per epoch.
        val_losses (list): Validation loss per epoch.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(train_losses)), train_losses, label="Training Loss")
    plt.plot(range(len(val_losses)), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.show()
```

### ROC & Metrics Functions


```python
def plot_roc_curve(y_true, y_probs):
    """
    Plot binary ROC curve with AUC.

    Parameters:
        y_true (array): True labels.
        y_probs (array): Predicted probabilities (n_samples, 2).
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr, tpr, color="blue", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.50)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Binary ROC Curve (AUC = {roc_auc:.4f})")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def _compute_all_metrics(y_true, y_pred, y_probs):
    """
    Compute all classification metrics and return as dict.

    Parameters:
        y_true (array): True labels.
        y_pred (array): Predicted labels.
        y_probs (array): Class probabilities.

    Returns:
        dict: Metric names mapped to scores.
    """
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    f2 = fbeta_score(y_true, y_pred, beta=2.0)
    auc_val = roc_auc_score(y_true, y_probs[:, 1])
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "F2-Score": f2,
        "ROC AUC": auc_val,
    }


def _metric_descriptions():
    """
    Return descriptions for each classification metric.

    Returns:
        dict: Metric names mapped to description strings.
    """
    return {
        "Accuracy": "Overall correctness (caution: misleading if imbalanced)",
        "Precision": "When predicting positive, how often correct?",
        "Recall": "Of all actual positives, how many found?",
        "F1-Score": "Harmonic mean of Precision & Recall",
        "F2-Score": "Recall-weighted F-beta (beta=2)",
        "ROC AUC": "Class separability (1.0 = perfect)",
    }


def print_summary_metrics(y_true, y_pred, y_probs):
    """
    Print formatted classification metrics table.

    Parameters:
        y_true (array): True labels.
        y_pred (array): Predicted labels.
        y_probs (array): Class probabilities.
    """
    metrics = _compute_all_metrics(y_true, y_pred, y_probs)
    descs = _metric_descriptions()
    print("-" * 90)
    print(f"{'METRIC':<15} {'SCORE':<10} {'DESCRIPTION'}")
    print("-" * 90)
    for name, score in metrics.items():
        print(f"{name:<15} {score:.4f}     {descs[name]}")
    print("-" * 90)
```

### SHAP Functions


```python
def _create_predict_fn(model, device):
    """
    Create a SHAP-compatible prediction function from a PyTorch model.

    Parameters:
        model (nn.Module): Trained PyTorch model.
        device (torch.device): Computing device.

    Returns:
        callable: Function mapping numpy arrays to softmax probabilities.
    """

    def predict_fn(data_numpy):
        """Convert numpy input to softmax probabilities via the model."""
        tensor = torch.tensor(data_numpy).float().to(device)
        model.eval()
        with torch.no_grad():
            return F.softmax(model(tensor), dim=1).cpu().numpy()

    return predict_fn


def _build_shap_explainer(model, X_train_scaled, device, n_background=100):
    """
    Build SHAP KernelExplainer with summarized training background.

    Parameters:
        model (nn.Module): Trained model.
        X_train_scaled (pd.DataFrame): Scaled training features.
        device (torch.device): Computing device.
        n_background (int): Number of background samples for KernelExplainer.

    Returns:
        shap.KernelExplainer: Configured SHAP explainer.
    """
    predict_fn = _create_predict_fn(model, device)
    n_unique = len(np.unique(X_train_scaled.values, axis=0))
    background = shap.kmeans(X_train_scaled.values, min(n_background, n_unique))
    return shap.KernelExplainer(predict_fn, background)


def run_shap_analysis(model, X_train_scaled, X_test_scaled, class_names, device):
    """
    Run SHAP analysis and display global feature importance bar plot.

    Parameters:
        model (nn.Module): Trained model.
        X_train_scaled (pd.DataFrame): Scaled training features.
        X_test_scaled (pd.DataFrame): Scaled test features.
        class_names (list): Human-readable class names.
        device (torch.device): Computing device.
    """
    explainer = _build_shap_explainer(model, X_train_scaled, device)
    shap_values = explainer.shap_values(X_test_scaled.values)
    plt.figure(figsize=(10, 5))
    shap.summary_plot(
        shap_values, X_test_scaled, class_names=class_names, plot_type="bar", show=False
    )
    plt.xlabel("Average Absolute SHAP Value (Feature Importance)")
    plt.tight_layout()
    plt.show()
```

### Inference Functions


```python
def _prepare_input_tensor(features, device):
    """
    Prepare input features as a tensor on the specified device.

    Parameters:
        features (list[float]): Feature values.
        device (torch.device): Target device.

    Returns:
        torch.Tensor: Input tensor with shape (1, n_features).
    """
    X_new = torch.tensor(features).float().to(device)
    if X_new.dim() == 1:
        X_new = X_new.unsqueeze(0)
    return X_new


def _get_prediction_details(logits):
    """
    Extract predicted class and confidence from logits.

    Parameters:
        logits (torch.Tensor): Raw model output.

    Returns:
        tuple: (predicted_class_index, confidence_score).
    """
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = logits.argmax(dim=1).item()
    confidence = probabilities[0][predicted_class].item()
    return predicted_class, confidence


def predict(model, features, class_map, device):
    """
    Make a prediction on new data.

    Parameters:
        model (nn.Module): Trained model.
        features (list[float]): Feature values (pre-scaled).
        class_map (dict): Mapping of class index to name.
        device (torch.device): Target device.

    Returns:
        tuple: (class_name, confidence_score).
    """
    model.eval()
    with torch.no_grad():
        X_new = _prepare_input_tensor(features, device)
        logits = model(X_new)
        pred_class, confidence = _get_prediction_details(logits)
        return class_map[pred_class], confidence
```

## Full-Feature Model (18 Features)

### Data Preparation & Training


```python
# Prepare data using all features (train/val/test from early split)
X_tr_full, X_val_full, X_te_full, y_tr_full, y_val_full, y_te_full, scaler_full = (
    prepare_data(df_train, df_test, features, TARGET_VAR, VAL_SIZE, SEED)
)
train_loader_full, val_loader_full, test_loader_full = create_loaders(
    X_tr_full,
    X_val_full,
    X_te_full,
    y_tr_full,
    y_val_full,
    y_te_full,
    DEVICE,
    BATCH_SIZE,
)
print(f"Features ({len(features)}): {features}")
print(f"Train: {len(y_tr_full)}, Val: {len(y_val_full)}, Test: {len(y_te_full)}")
```

    Features (18): ['age', 'gender', 'bmi', 'blood_pressure', 'fasting_glucose_level', 'insulin_level', 'HbA1c_level', 'cholesterol_level', 'triglycerides_level', 'physical_activity_level', 'daily_calorie_intake', 'sugar_intake_grams_per_day', 'sleep_hours', 'stress_level', 'family_history_diabetes', 'waist_circumference_cm', 'bmi_category', 'glucose_category']
    Train: 3570, Val: 630, Test: 1800



```python
# Create and train the full-feature model
model_full = DiabetesMLP(
    in_features=len(features), h1=H1, h2=H2, out_features=OUT_FEATURES, dropout=DROPOUT
).to(DEVICE)
class_weights_full = _compute_class_weights(y_tr_full, DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights_full)
optimizer = torch.optim.AdamW(model_full.parameters(), lr=LR)
full_config = {
    "epochs": EPOCHS,
    "log_interval": LOG_INTERVAL,
    "save_path": "drp_model_full.pth",
    "patience": PATIENCE,
}
train_losses_full, val_losses_full, best_full = train_model(
    model_full, train_loader_full, val_loader_full, criterion, optimizer, full_config
)
print(f"\nBest Validation Loss: {best_full:.4f}")
```

    Epoch 0/1000 - Train: 0.5737 - Val: 0.4138
    Epoch 10/1000 - Train: 0.0940 - Val: 0.0891
    Epoch 20/1000 - Train: 0.0680 - Val: 0.0815
    Epoch 30/1000 - Train: 0.0600 - Val: 0.0803
    Epoch 40/1000 - Train: 0.0548 - Val: 0.0827
    Epoch 50/1000 - Train: 0.0504 - Val: 0.0791
    Epoch 60/1000 - Train: 0.0501 - Val: 0.0766
    Epoch 70/1000 - Train: 0.0426 - Val: 0.0835
    Epoch 80/1000 - Train: 0.0422 - Val: 0.0806
    Epoch 90/1000 - Train: 0.0432 - Val: 0.0830
    Early stopping at epoch 91 (patience=50)
    
    Best Validation Loss: 0.0761


### Evaluate Full-Feature Model


```python
# Collect predictions and compute accuracy
y_true_full, y_pred_full, y_probs_full = collect_predictions(
    model_full, test_loader_full
)
train_acc_full = accuracy_score(*collect_predictions(model_full, train_loader_full)[:2])
val_acc_full = accuracy_score(*collect_predictions(model_full, val_loader_full)[:2])
test_acc_full = accuracy_score(y_true_full, y_pred_full)
print(f"Train Accuracy: {train_acc_full:.4f}")
print(f"Val Accuracy:   {val_acc_full:.4f}")
print(f"Test Accuracy:  {test_acc_full:.4f}")
```

    Train Accuracy: 0.9891
    Val Accuracy:   0.9683
    Test Accuracy:  0.9744



```python
plot_accuracy_pies(train_acc_full, test_acc_full)
```


    
![png](README_files/DRP-MLP_160_0.png)
    



```python
plot_confusion_matrix(y_true_full, y_pred_full, CLASS_NAMES)
```


    
![png](README_files/DRP-MLP_161_0.png)
    



```python
plot_loss_curves(train_losses_full, val_losses_full)
```


    
![png](README_files/DRP-MLP_162_0.png)
    



```python
print(classification_report(y_true_full, y_pred_full, target_names=CLASS_NAMES))
```

                  precision    recall  f1-score   support
    
        Low Risk       0.97      0.98      0.97       900
       High Risk       0.98      0.97      0.97       900
    
        accuracy                           0.97      1800
       macro avg       0.97      0.97      0.97      1800
    weighted avg       0.97      0.97      0.97      1800
    



```python
plot_roc_curve(y_true_full, y_probs_full)
```


    
![png](README_files/DRP-MLP_164_0.png)
    



```python
print_summary_metrics(y_true_full, y_pred_full, y_probs_full)
```

    ------------------------------------------------------------------------------------------
    METRIC          SCORE      DESCRIPTION
    ------------------------------------------------------------------------------------------
    Accuracy        0.9744     Overall correctness (caution: misleading if imbalanced)
    Precision       0.9766     When predicting positive, how often correct?
    Recall          0.9722     Of all actual positives, how many found?
    F1-Score        0.9744     Harmonic mean of Precision & Recall
    F2-Score        0.9731     Recall-weighted F-beta (beta=2)
    ROC AUC         0.9979     Class separability (1.0 = perfect)
    ------------------------------------------------------------------------------------------


### SHAP Analysis — Full-Feature Model


```python
run_shap_analysis(model_full, X_tr_full, X_te_full, CLASS_NAMES, DEVICE)
```


      0%|          | 0/1800 [00:00<?, ?it/s]



    <Figure size 1000x500 with 0 Axes>



    
![png](README_files/DRP-MLP_167_2.png)
    


## Lean-Feature Model (9 Features)

### Data Preparation & Training


```python
# Select lean feature set based on discriminative score + domain knowledge
features_lean = [
    "HbA1c_level",
    "fasting_glucose_level",
    "bmi",
    "insulin_level",
    "age",
    "blood_pressure",
    "waist_circumference_cm",
    "family_history_diabetes",
    "glucose_category",
]
X_tr_lean, X_val_lean, X_te_lean, y_tr_lean, y_val_lean, y_te_lean, scaler_lean = (
    prepare_data(df_train, df_test, features_lean, TARGET_VAR, VAL_SIZE, SEED)
)
train_loader_lean, val_loader_lean, test_loader_lean = create_loaders(
    X_tr_lean,
    X_val_lean,
    X_te_lean,
    y_tr_lean,
    y_val_lean,
    y_te_lean,
    DEVICE,
    BATCH_SIZE,
)
print(f"Features ({len(features_lean)}): {features_lean}")
print(f"Train: {len(y_tr_lean)}, Val: {len(y_val_lean)}, Test: {len(y_te_lean)}")
```

    Features (9): ['HbA1c_level', 'fasting_glucose_level', 'bmi', 'insulin_level', 'age', 'blood_pressure', 'waist_circumference_cm', 'family_history_diabetes', 'glucose_category']
    Train: 3570, Val: 630, Test: 1800



```python
# Create and train the lean-feature model
model_lean = DiabetesMLP(
    in_features=len(features_lean),
    h1=H1,
    h2=H2,
    out_features=OUT_FEATURES,
    dropout=DROPOUT,
).to(DEVICE)
class_weights_lean = _compute_class_weights(y_tr_lean, DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights_lean)
optimizer = torch.optim.AdamW(model_lean.parameters(), lr=LR)
lean_config = {
    "epochs": EPOCHS,
    "log_interval": LOG_INTERVAL,
    "save_path": "drp_model_lean.pth",
    "patience": PATIENCE,
}
train_losses_lean, val_losses_lean, best_lean = train_model(
    model_lean, train_loader_lean, val_loader_lean, criterion, optimizer, lean_config
)
print(f"\nBest Validation Loss: {best_lean:.4f}")
```

    Epoch 0/1000 - Train: 0.5410 - Val: 0.3906
    Epoch 10/1000 - Train: 0.1096 - Val: 0.0972
    Epoch 20/1000 - Train: 0.0936 - Val: 0.0999
    Epoch 30/1000 - Train: 0.0844 - Val: 0.1039
    Epoch 40/1000 - Train: 0.0767 - Val: 0.1028
    Epoch 50/1000 - Train: 0.0788 - Val: 0.1079
    Early stopping at epoch 58 (patience=50)
    
    Best Validation Loss: 0.0949


### Evaluate Lean-Feature Model


```python
# Collect predictions and compute accuracy
y_true_lean, y_pred_lean, y_probs_lean = collect_predictions(
    model_lean, test_loader_lean
)
train_acc_lean = accuracy_score(*collect_predictions(model_lean, train_loader_lean)[:2])
val_acc_lean = accuracy_score(*collect_predictions(model_lean, val_loader_lean)[:2])
test_acc_lean = accuracy_score(y_true_lean, y_pred_lean)
print(f"Train Accuracy: {train_acc_lean:.4f}")
print(f"Val Accuracy:   {val_acc_lean:.4f}")
print(f"Test Accuracy:  {test_acc_lean:.4f}")
```

    Train Accuracy: 0.9706
    Val Accuracy:   0.9556
    Test Accuracy:  0.9594



```python
plot_accuracy_pies(train_acc_lean, test_acc_lean)
```


    
![png](README_files/DRP-MLP_174_0.png)
    



```python
plot_confusion_matrix(y_true_lean, y_pred_lean, CLASS_NAMES)
```


    
![png](README_files/DRP-MLP_175_0.png)
    



```python
plot_loss_curves(train_losses_lean, val_losses_lean)
```


    
![png](README_files/DRP-MLP_176_0.png)
    



```python
print(classification_report(y_true_lean, y_pred_lean, target_names=CLASS_NAMES))
```

                  precision    recall  f1-score   support
    
        Low Risk       0.96      0.95      0.96       900
       High Risk       0.95      0.96      0.96       900
    
        accuracy                           0.96      1800
       macro avg       0.96      0.96      0.96      1800
    weighted avg       0.96      0.96      0.96      1800
    



```python
plot_roc_curve(y_true_lean, y_probs_lean)
```


    
![png](README_files/DRP-MLP_178_0.png)
    



```python
print_summary_metrics(y_true_lean, y_pred_lean, y_probs_lean)
```

    ------------------------------------------------------------------------------------------
    METRIC          SCORE      DESCRIPTION
    ------------------------------------------------------------------------------------------
    Accuracy        0.9594     Overall correctness (caution: misleading if imbalanced)
    Precision       0.9549     When predicting positive, how often correct?
    Recall          0.9644     Of all actual positives, how many found?
    F1-Score        0.9596     Harmonic mean of Precision & Recall
    F2-Score        0.9625     Recall-weighted F-beta (beta=2)
    ROC AUC         0.9951     Class separability (1.0 = perfect)
    ------------------------------------------------------------------------------------------


### SHAP Analysis — Lean-Feature Model


```python
run_shap_analysis(model_lean, X_tr_lean, X_te_lean, CLASS_NAMES, DEVICE)
```


      0%|          | 0/1800 [00:00<?, ?it/s]



    <Figure size 1000x500 with 0 Axes>



    
![png](README_files/DRP-MLP_181_2.png)
    


### Save Lean Model Artifacts


```python
# Save lean model scaler for production inference
joblib.dump(scaler_lean, "drp_scaler.pkl")
print("Scaler Saved: 'drp_scaler.pkl'")
```

    Scaler Saved: 'drp_scaler.pkl'


## Inference

### Load Model


```python
loaded_model = DiabetesMLP(
    in_features=len(features_lean), h1=H1, h2=H2, out_features=OUT_FEATURES
).to(DEVICE)
loaded_model.load_state_dict(
    torch.load("drp_model_lean.pth", map_location=DEVICE, weights_only=True)
)
loaded_model.eval()
print("Model Loaded: 'drp_model_lean.pth'")
```

    Model Loaded: 'drp_model_lean.pth'


### Load Scaler


```python
scaler = joblib.load("drp_scaler.pkl")
print("Scaler Loaded: 'drp_scaler.pkl'")
```

    Scaler Loaded: 'drp_scaler.pkl'


### Example: New Patient Data


```python
# 1. Define Raw Data (matching features_lean order)
HbA1c_level = 6.5  # elevated glycated hemoglobin
fasting_glucose_level = 120.0  # borderline high fasting glucose
bmi = 31.5  # obese BMI range
insulin_level = 15.0  # moderate insulin
age = 45.0  # middle-aged patient
blood_pressure = 140.0  # stage 1 hypertension
waist_circumference_cm = 95.0  # elevated waist circumference
family_history_diabetes = 1.0  # has family history (encoded)
glucose_category = 1.0  # prediabetic (encoded)
raw_patient = [
    HbA1c_level,
    fasting_glucose_level,
    bmi,
    insulin_level,
    age,
    blood_pressure,
    waist_circumference_cm,
    family_history_diabetes,
    glucose_category,
]

# 2. CRITICAL: Scale using the fitted scaler from training
scaled_patient = scaler.transform([raw_patient])

# 3. Predict using SCALED data
risk_class, confidence = predict(loaded_model, scaled_patient[0], CLASS_MAP, DEVICE)
print(f"Raw Input:          {raw_patient}")
print(f"Predicted Risk:     {risk_class}")
print(f"Confidence:         {confidence:.2%}")
```

    Raw Input:          [6.5, 120.0, 31.5, 15.0, 45.0, 140.0, 95.0, 1.0, 1.0]
    Predicted Risk:     High Risk
    Confidence:         96.01%

