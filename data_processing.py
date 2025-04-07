import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression

# Настройки отображения
sns.set_style("whitegrid")
plt.style.use('ggplot')
pd.set_option('display.max_columns', 100)
np.random.seed(42)


def load_data():
    """Загрузка и объединение данных"""
    train = pd.read_csv('student_performance_train.csv')
    test = pd.read_csv('student_performance_test.csv')
    return train, test


def preprocess_data(train, test):
    """Улучшенная предобработка данных"""
    y_train = train['final_math_score']
    train_ids = train['student_id']
    test_ids = test['student_id']

    combined = pd.concat([train.drop('final_math_score', axis=1), test])

    # Расширенный feature engineering
    combined['efficiency_ratio'] = combined['previous_scores'] / (combined['study_hours'] + 1e-6)
    combined['attendance_impact'] = combined['previous_scores'] * (combined['attendance_rate'] / 100)
    combined['parent_edu_encoded'] = combined['parental_education'].map(
        {'High School': 1, 'Bachelor’s': 2, 'Master’s': 3})
    combined['study_attendance_interaction'] = combined['study_hours'] * combined['attendance_rate']
    combined['score_trend'] = combined.groupby('parental_education')['previous_scores'].transform('mean')

    # Обработка выбросов
    combined['study_hours'] = np.clip(combined['study_hours'], 1, 20)

    # Обработка пропущенных значений
    num_cols = combined.select_dtypes(include=np.number).columns
    cat_cols = ['gender', 'parental_education', 'school_type']

    combined[num_cols] = SimpleImputer(strategy='median').fit_transform(combined[num_cols])
    combined[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(combined[cat_cols])

    # Выбор фичей
    num_features = ['age', 'study_hours', 'attendance_rate', 'previous_scores',
                    'extracurricular', 'efficiency_ratio', 'attendance_impact',
                    'parent_edu_encoded', 'study_attendance_interaction', 'score_trend']
    cat_features = ['gender', 'school_type', 'parental_education']

    # Пайплайн предобработки
    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(f_regression, k=8))
        ]), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

    X_train = combined.iloc[:len(train)]
    X_test = combined.iloc[len(train):]

    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, y_train, X_test_processed, test_ids, preprocessor


def train_model(X_train, y_train):
    """Улучшенное обучение модели"""
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'n_estimators': [500, 700],
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9]
    }

    model = XGBRegressor(random_state=42, tree_method='hist')
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(model, param_grid, cv=kfold,
                               scoring='neg_mean_squared_error',
                               n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

    return best_model


def create_submission(model, X_test, test_ids, filename='improved_submission.csv'):
    predictions = model.predict(X_test)
    predictions = np.clip(predictions, 0, 100)

    submission = pd.DataFrame({
        'student_id': test_ids,
        'predicted_math_score': np.round(predictions, 1)
    })

    plt.figure(figsize=(10, 6))
    sns.histplot(submission['predicted_math_score'], bins=30, kde=True)
    plt.title('Improved Distribution of Predicted Math Scores')
    plt.savefig('prediction_distribution.png', bbox_inches='tight')

    submission.to_csv(filename, index=False)
    return submission


def main():
    train, test = load_data()
    X_train, y_train, X_test, test_ids, _ = preprocess_data(train, test)

    # Анализ целевой переменной
    plt.figure(figsize=(10, 6))
    sns.histplot(y_train, bins=30, kde=True)
    plt.title('Distribution of Target Variable (final_math_score)')
    plt.savefig('target_distribution.png', bbox_inches='tight')

    model = train_model(X_train, y_train)
    submission = create_submission(model, X_test, test_ids)
    print("\nSubmission summary:")
    print(submission.describe())


if __name__ == "__main__":
    main()

