import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

def load_data(file_path: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    从 Excel 文件加载训练和测试数据。
    """
    try:
        train_df = pd.read_excel(file_path, sheet_name=0, header=None)
        test_df = pd.read_excel(file_path, sheet_name=1, header=None)
        
        # 分离特征和目标变量
        # 特征：第 0 到 30 列（包含 30，所以切片到 31）
        # 目标：第 31 列
        X_train = train_df.iloc[:, :31]
        y_train = train_df.iloc[:, 31]
        
        X_test = test_df.iloc[:, :31]
        y_test = test_df.iloc[:, 31]
        
        return X_train, y_train, X_test, y_test
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def create_pipeline() -> Pipeline:
    """
    创建一个包含预处理和模型的 scikit-learn 管道。
    """
    # 第 0-29 列是数值型（索引 0 到 29）
    # 第 30 列是分类型（索引 30）
    
    numeric_features = list(range(30))
    categorical_features = [30]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(random_state=42))
    ])
    
    return pipeline

def train_and_tune(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """
    使用 GridSearchCV 训练模型并进行超参数调优。
    """
    pipeline = create_pipeline()
    
    param_grid = {
        'regressor__n_estimators': [50, 100, 200, 300],
        'regressor__learning_rate': [0.005, 0.01, 0.05, 0.1],
        'regressor__max_depth': [2, 3, 4, 5],
        'regressor__min_samples_split': [2, 5, 10]
    }
    
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    print("Starting hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score (MSE): {-grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    在测试集上评估模型并报告平方相对误差统计信息。
    """
    y_pred = model.predict(X_test)
    
    # 计算平方相对误差 (SRE)
    # SRE_i = ((y_true - y_pred) / y_true)^2
    
    # 避免除以零
    epsilon = 1e-10
    sre = ((y_test - y_pred) / (y_test + epsilon)) ** 2
    
    mean_sre = np.mean(sre)
    var_sre = np.var(sre)
    
    print("\n" + "="*40)
    print("Evaluation Report on Test Set (Sheet 2)")
    print("="*40)
    print(f"Mean Squared Relative Error: {mean_sre:.6f}")
    print(f"Variance of Squared Relative Error: {var_sre:.6f}")
    
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print("="*40)

if __name__ == "__main__":
    file_path = "回归预测.xlsx"
    
    try:
        print("Loading data...")
        X_train, y_train, X_test, y_test = load_data(file_path)
        
        best_model = train_and_tune(X_train, y_train)
        
        evaluate_model(best_model, X_test, y_test)
        
    except Exception as e:
        print(f"An error occurred: {e}")
