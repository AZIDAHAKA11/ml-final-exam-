import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle


# Load dataset

df = pd.read_csv("vgsales.csv")
print("Shape:", df.shape)


# Basic cleaning 

df = df.dropna(subset=['Global_Sales']).copy()

df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df.loc[(df['Year'] < 1970) | (df['Year'] > 2050), 'Year'] = np.nan

df['Publisher'] = df['Publisher'].fillna('Unknown')


# Outlier clipping 

lower = df['Global_Sales'].quantile(0.01)
upper = df['Global_Sales'].quantile(0.99)
df['Global_Sales'] = df['Global_Sales'].clip(lower, upper)


# FEATURES 

feature_cols = [
    'Platform',
    'Year',
    'Genre',
    'Publisher'
]

X = df[feature_cols].copy()
y = df['Global_Sales'].copy()


# Feature types

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()


# Pipelines 

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

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('clf', model)
])


# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

pipeline.fit(X_train, y_train)


# Hyperparameter tuning (unchanged)

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


# Test Set Evaluation

y_test_pred = best_model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae = mean_absolute_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("Test Set Evaluation Results:")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")



# Save model

with open("model.pkl", "wb") as file:
    pickle.dump(best_model, file)


