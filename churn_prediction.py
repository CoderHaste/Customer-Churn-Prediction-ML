import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score

df = pd.read_csv("Telco-Customer-Churn.csv")
print("Class Distribution: \n")
print(df['Churn'].value_counts())
print("\nSample Data:\n",df.head())

# Handle Missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.fillna({'TotalCharges': df['TotalCharges'].median()}, inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
for col in df.select_dtypes(include=['str']).columns:
    if col != 'Churn':
        df[col] = label_encoder.fit_transform(df[col])

# Encode target variable
df['Churn'] = label_encoder.fit_transform(df['Churn'])

# Scale the numerical features
scaler = StandardScaler()
numerical_feaures = ['tenure','MonthlyCharges','TotalCharges']
df[numerical_feaures] = scaler.fit_transform(df[numerical_feaures])

# Features and Target of the dataset
X = df.drop(columns=['Churn'])
y = df['Churn']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train,y_train)

# Display class distribution after SMOTE
print("\nClass Distribution after SMOTE:\n")
print(pd.Series(y_train_resampled).value_counts())

# Train random forest on resampled data
rf_model_smote = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf_model_smote.fit(X_train_resampled, y_train_resampled) 

# Predict and evaluate
y_pred_smote = rf_model_smote.predict(X_test)
print(f"RF Classification report: \n",classification_report(y_test,y_pred_smote))

# XGBoost
xgb_model = XGBClassifier(eval_metric='logloss', random_state = 42, n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8)
xgb_model.fit(X_train_resampled,y_train_resampled)
y_pred_xgb = xgb_model.predict(X_test)
print(f"Classification report(XGBoost): \n",classification_report(y_test,y_pred_xgb))

# Train LightGBM
LGBM_model = LGBMClassifier(random_state = 42, n_estimators=200, learning_rate=0.05, max_depth=-1,)
LGBM_model.fit(X_train_resampled,y_train_resampled)
y_pred_LGBM = LGBM_model.predict(X_test)
print(f"Classification report(LightGBM): \n",classification_report(y_test,y_pred_LGBM))

# Model Comparison summary
print("\nModel comparison summary:")
print(f"RF ROC-AUC: {roc_auc_score(y_test,rf_model_smote.predict_proba(X_test)[:,1]):.4f}")
print(f"XGB ROC-AUC: {roc_auc_score(y_test,xgb_model.predict_proba(X_test)[:,1]):.4f}")
print(f"LGBM ROC-AUC: {roc_auc_score(y_test,LGBM_model.predict_proba(X_test)[:,1]):.4f}")
