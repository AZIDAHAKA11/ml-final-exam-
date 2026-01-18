import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score 
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from sklearn.impute import SimpleImputer 
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV  
import gradio as gr
import pickle

#Load dataset
df = pd.read_csv("vgsales.csv") 
print("Shape:", df.shape) 
df.head()
# Inspect and basic cleaning
print(df.info())
print("\nMissing values per column:\n", df.isna().sum())

# Missing rows dropped 
df = df.dropna(subset=['Global_Sales']).copy()

# imputing floating values with median
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# Replacing years that are heavy
df.loc[(df['Year'] < 1970) | (df['Year'] > 2050), 'Year'] = np.nan

# Filling unknown pulisher
df['Publisher'] = df['Publisher'].fillna('Unknown')

#regional sales
df['Total_Regional_Sales'] = df[['NA_Sales','EU_Sales','JP_Sales','Other_Sales']].sum(axis=1)

# Sales ratio features
df['NA_ratio'] = df['NA_Sales'] / (df['Global_Sales'] + 1e-6)
df['EU_ratio'] = df['EU_Sales'] / (df['Global_Sales'] + 1e-6)
df['JP_ratio'] = df['JP_Sales'] / (df['Global_Sales'] + 1e-6)
df['Other_ratio'] = df['Other_Sales'] / (df['Global_Sales'] + 1e-6)

# 3) Outlier clipping 
for col in ['NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales','Total_Regional_Sales']:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = df[col].clip(lower, upper)

print("After preprocessing:")
print(df.describe(include='all').transpose().head(20))
feature_cols = [
    'Platform','Year','Genre','Publisher',
    'NA_Sales','EU_Sales','JP_Sales','Other_Sales',
    'Total_Regional_Sales','NA_ratio','EU_ratio','JP_ratio','Other_ratio'
]

X = df[feature_cols].copy()
y = df['Global_Sales'].copy()

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()
#Creating Pipeline

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

model = RandomForestRegressor(n_estimators=300, random_state=42)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('clf', model)])
#Trinning and splitting

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)
# Cross-Validation and Hyperparameter tunning using RandomizedSearchCV

param_dist = {
    'clf__n_estimators': [100, 300, 500, 800],
    'clf__max_depth': [None, 10, 20, 30],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4],
    'clf__max_features': ['sqrt', 'log2', None]
}

random_search_cv = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=5,   
    cv=3,       
    scoring='neg_root_mean_squared_error',
    random_state=42,
    n_jobs=-1
)

random_search_cv.fit(X_train, y_train)

print("Best parameters:", random_search_cv.best_params_)
print(f"Best CV RMSE: {-random_search_cv.best_score_:.4f}")

best_model = random_search_cv.best_estimator_


with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
