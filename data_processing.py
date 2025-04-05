

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def load_data(train_file, test_file, sample_submission_file):
    """Load the datasets from the provided file paths."""
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    sample_submission = pd.read_csv(sample_submission_file)
    return train_df, test_df, sample_submission

def preprocess_data(train_df, test_df):
    """Preprocess the data: handle missing values, one-hot encoding, scaling."""
    # Сохраняем целевую переменную и идентификаторы
    target = train_df['final_math_score']
    train_ids = train_df['student_id']
    test_ids = test_df['student_id']

    train_df = train_df.drop(columns=['final_math_score'])

    # Заполнение пропусков
    train_df = train_df.fillna(train_df.mean(numeric_only=True))
    test_df = test_df.fillna(test_df.mean(numeric_only=True))

    # One-hot encoding
    train_df = pd.get_dummies(train_df, drop_first=True)
    test_df = pd.get_dummies(test_df, drop_first=True)

    # Выравнивание колонок
    test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

    # Масштабирование
    scaler = StandardScaler()
    train_df = pd.DataFrame(scaler.fit_transform(train_df), columns=train_df.columns)
    test_df = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)

    # Добавим обратно ID и цель
    train_df['student_id'] = train_ids.values
    train_df['final_math_score'] = target.values
    test_df['student_id'] = test_ids.values

    return train_df, test_df, scaler

def train_model(train_df):
    """Train a Random Forest model using the training data."""
    X = train_df.drop(['final_math_score', 'student_id'], axis=1)
    y = train_df['final_math_score']

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_valid_pred = model.predict(X_valid)
    mse = mean_squared_error(y_valid, y_valid_pred)
    print(f"Mean Squared Error on validation set: {mse:.2f}")

    return model

def make_predictions(model, test_df):
    """Use the trained model to make predictions on the test set."""
    X_test = test_df.drop(['student_id'], axis=1)
    return model.predict(X_test)

def prepare_submission(test_df, y_test_pred, output_file):
    """Prepare the submission file."""
    submission = pd.DataFrame({
        'student_id': test_df['student_id'],
        'predicted_math_score': np.clip(y_test_pred, 0, 100)  # Обрезаем от 0 до 100
    })
    submission.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")

# --- data_analysis.py ---

from data_processing import load_data, preprocess_data, train_model, make_predictions, prepare_submission

# Step 1: Load data
train_df, test_df, sample_submission = load_data('student_performance_train.csv',
                                                 'student_performance_test.csv',
                                                 'sample_submission.csv')

# Step 2: Preprocess data
train_df, test_df, scaler = preprocess_data(train_df, test_df)

# Step 3: Train the model
model = train_model(train_df)

# Step 4: Make predictions on the test data
y_test_pred = make_predictions(model, test_df)

# Step 5: Prepare and save the final submission
prepare_submission(test_df, y_test_pred, 'submission.csv')