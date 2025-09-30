import os
os.environ["OMP_NUM_THREADS"] = "1"

from config import DATA_DIR, SAVE_DIR
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
import numpy as np
import pandas as pd
import joblib
import os
from data_loader import load_dataset
from feature_extractor import Au20FeatureExtractor
from model_utils import evaluate_model, plot_feature_importance, find_lowest_energy_structure
from task2 import task2
from task3 import task3
# 在文件开头的导入部分添加
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.model_selection import KFold, cross_validate

def main():
    # 设置数据路径
    data_dir = "C:/Users/DKE/Desktop/cyber c/data/Au20_OPT_1000"
    
    # 加载数据
    print("正在加载数据和提取特征...")
    features_df, energies = load_dataset(data_dir)
    
    print(f"特征矩阵形状: {features_df.shape}")
    print(f"能量数组形状: {energies.shape}")
    
    # 数据预处理
    features_df = features_df.fillna(features_df.mean())
    
    # 特征和标签
    X = features_df.values
    y = energies
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # -----------------------------
    # 交叉验证
    # -----------------------------
    from sklearn.model_selection import KFold, cross_val_score
    import numpy as np

    print("\n使用5折交叉验证评估模型性能...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    xgb_model_cv = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=0.5,
        min_child_weight=1,
        random_state=42,
        n_jobs=-1
    )

    from sklearn.metrics import make_scorer, mean_absolute_error, r2_score

    scoring = {
        'rmse': make_scorer(lambda y_true, y_pred: np.sqrt(np.mean((y_true - y_pred)**2))),
        'mae': make_scorer(mean_absolute_error),
        'r2': make_scorer(r2_score)
    }

    cv_results = cross_validate(xgb_model_cv, X_scaled, y, cv=kf, scoring=scoring)

    for metric in scoring.keys():
        values = cv_results[f'test_{metric}']
        print(f"{metric.upper()}:", values)
        print(f"{metric.upper()} 平均: {values.mean():.4f} ± {values.std():.4f}\n")

    
    # -----------------------------
    # 使用全量数据训练最终模型
    # -----------------------------
    print("\n使用全量数据训练最终模型用于 Task2/Task3...")
    xgb_model_final = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=0.5,
        min_child_weight=1,
        random_state=42,
        n_jobs=-1
    )
    xgb_model_final.fit(X_scaled, y)
    
    # -----------------------------
    # 特征重要性分析
    # -----------------------------
    print("\n特征重要性分析:")
    plot_feature_importance(xgb_model_final, X_scaled, features_df.columns.tolist())
    
    # -----------------------------
    # 保存模型和预处理器
    # -----------------------------
    results = {
        'feature_importance': dict(zip(features_df.columns, xgb_model_final.feature_importances_))
    }
    
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "best_au20_model.pkl")
    joblib.dump(xgb_model_final, model_path)
    
    scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    
    feature_names_path = os.path.join(model_dir, "feature_names.pkl")
    joblib.dump(features_df.columns.tolist(), feature_names_path)
    
    extractor_path = os.path.join(model_dir, "feature_extractor.pkl")
    joblib.dump(Au20FeatureExtractor(), extractor_path)
    
    print(f"\n模型和预处理器已保存到 '{model_dir}' 目录:")
    print(f"- 模型: {model_path}")
    print(f"- 特征标准化器: {scaler_path}")
    print(f"- 特征名称: {feature_names_path}")
    print(f"- 特征提取器: {extractor_path}")
    
    # -----------------------------
    # 执行 Task 3
    # -----------------------------
    print("\n执行 Task 3: 扰动灵敏度分析...")
    best_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    extractor = joblib.load(extractor_path)
    feature_names = joblib.load(feature_names_path)
    
    atoms, lowest_coords = find_lowest_energy_structure(data_dir, extractor, energies)
    lowest_energy = np.min(energies)
    
    print(f"最低能量: {lowest_energy:.4f}")
    print(f"最低能量结构坐标形状: {lowest_coords.shape}")
    
    stability_df = task3(lowest_coords, lowest_energy, best_model, scaler, extractor, feature_names)
    print("\n扰动灵敏度分析结果:")
    print(stability_df)
    
    return {
        **results,
        "energies": energies
    }

if __name__ == "__main__":
    results = main()
    task2("C:/Users/DKE/Desktop/cyber c/data/Au20_OPT_1000", results["energies"])
