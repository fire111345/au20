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
from sklearn.model_selection import KFold, cross_val_score


def main():
    # 设置数据路径
    data_dir = "C:/Users/DKE/Desktop/cyber c/data/Au20_OPT_1000"
    
    # 加载数据
    print("正在加载数据和提取特征...")
    features_df, energies = load_dataset(data_dir)
    
    print(f"特征矩阵形状: {features_df.shape}")
    print(f"能量数组形状: {energies.shape}")
    
    # 数据预处理
    # 处理缺失值
    features_df = features_df.fillna(features_df.mean())
    
    # 特征和标签
    X = features_df.values
    y = energies
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"PCA降维后特征数: {X_train_pca.shape[1]}")
    
    # 训练XGBoost模型
    print("\n训练XGBoost模型...")
    
    # 使用原始特征
   # 使用调优找到的最佳参数
    xgb_model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,           # 使用调优找到的4而不是6
        subsample=0.7,         # 使用调优找到的0.7
        colsample_bytree=0.7,  # 使用调优找到的0.7而不是0.8
        reg_alpha=0.5,         # 添加调优找到的reg_alpha
        reg_lambda=0.5,        # 添加调优找到的reg_lambda  
        min_child_weight=1,    # 添加调优找到的min_child_weight
        random_state=42,
        n_jobs=-1
    )
    
    # 正确的fit方法调用
    xgb_model.fit(X_train, y_train,
                 eval_set=[(X_test, y_test)],
                 verbose=False)
    
    # 评估模型
    print("\n模型性能评估:")
    mae, rmse, r2, y_pred = evaluate_model(xgb_model, X_test, y_test)
    
    # 特征重要性分析
    print("\n特征重要性分析:")
    plot_feature_importance(xgb_model, X_test, features_df.columns.tolist())
    
    # 使用PCA特征训练另一个模型比较
    print("\n使用PCA特征训练模型...")
    xgb_model_pca = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    # 同样修正这里的fit调用
    xgb_model_pca.fit(X_train_pca, y_train,
                     eval_set=[(X_test_pca, y_test)],
                     verbose=False)
    
    print("PCA特征模型性能:")
    mae_pca, rmse_pca, r2_pca, _ = evaluate_model(xgb_model_pca, X_test_pca, y_test)
    
    # 结果可视化
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('True Energy')
    plt.ylabel('Predicted Energy')
    plt.title('True vs Predicted Energy')
    plt.show()
    
    # 保存调优结果
    results  = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAE_PCA': mae_pca,
        'RMSE_PCA': rmse_pca,
        'R2_PCA': r2_pca,
        'feature_importance': dict(zip(features_df.columns, xgb_model.feature_importances_))
    }
     # 创建模型保存目录
    model_dir = "saved_models"
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存最佳模型
    model_path = os.path.join(model_dir, "best_au20_model.pkl")
    joblib.dump(xgb_model, model_path)
    
    # 保存特征标准化器
    scaler_path = os.path.join(model_dir, "feature_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    
    # 保存特征名称
    feature_names_path = os.path.join(model_dir, "feature_names.pkl")
    joblib.dump(features_df.columns.tolist(), feature_names_path)
    
    # 保存PCA转换器（可选）
    pca_path = os.path.join(model_dir, "pca_transformer.pkl")
    joblib.dump(pca, pca_path)
    
    # 保存特征提取器（如果需要）
    extractor_path = os.path.join(model_dir, "feature_extractor.pkl")
    joblib.dump(Au20FeatureExtractor(), extractor_path)
    
    print(f"\n模型和预处理器已保存到 '{model_dir}' 目录:")
    print(f"- 模型: {model_path}")
    print(f"- 特征标准化器: {scaler_path}")
    print(f"- 特征名称: {feature_names_path}")
    print(f"- PCA转换器: {pca_path}")
    print(f"- 特征提取器: {extractor_path}")
    

     # 执行 Task 3
    print("\n执行 Task 3: 扰动灵敏度分析...")
    
    # 重新加载必要的组件（确保一致性）
    best_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    extractor = joblib.load(extractor_path)
    feature_names = joblib.load(feature_names_path)
    
    # 找到最低能量结构
    atoms, lowest_coords = find_lowest_energy_structure(data_dir, extractor, energies)
    lowest_energy = np.min(energies)
    
    print(f"最低能量: {lowest_energy:.4f}")
    print(f"最低能量结构坐标形状: {lowest_coords.shape}")
    
    # 运行 Task 3（传入特征名称）
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
