
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 从 config 导入参数
from config import PERTURB_N_SAMPLES, PERTURB_MAGNITUDES, PERTURB_RATIO, PERTURB_SEED


def apply_local_perturbations(coords,
                             n_samples=PERTURB_N_SAMPLES,
                             magnitudes=PERTURB_MAGNITUDES,
                             perturb_ratio=PERTURB_RATIO,
                             seed=PERTURB_SEED):
    """
    对原始坐标应用局部随机扰动
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_atoms = coords.shape[0]
    perturbed_structures = {}
    
    for mag in magnitudes:
        perturbed_list = []
        for _ in range(n_samples):
            perturbation = np.zeros_like(coords)
            
            if perturb_ratio < 1.0:
                n_perturb = int(n_atoms * perturb_ratio)
                perturb_indices = np.random.choice(n_atoms, size=n_perturb, replace=False)
                perturbation[perturb_indices] = np.random.normal(
                    scale=mag, 
                    size=(n_perturb, 3)
                )
            else:
                perturbation = np.random.normal(scale=mag, size=coords.shape)
            
            new_coords = coords + perturbation
            perturbed_list.append(new_coords)
        
        perturbed_structures[mag] = perturbed_list
    
    return perturbed_structures


def rmsd(a, b):
    """计算两结构的均方根偏差 (RMSD)"""
    return np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))


def kabsch_rmsd(P, Q):
    """
    Kabsch 对齐后的 RMSD
    P: (N,3) 参考结构坐标
    Q: (N,3) 扰动后结构坐标
    """
    P = np.array(P)
    Q = np.array(Q)

    # 去中心化
    P_cent = P - np.mean(P, axis=0)
    Q_cent = Q - np.mean(Q, axis=0)

    # Kabsch 旋转
    C = np.dot(Q_cent.T, P_cent)
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(np.dot(V, Wt)))
    D = np.diag([1, 1, d])
    U = np.dot(V, np.dot(D, Wt))

    Q_rot = np.dot(Q_cent, U)

    diff = P_cent - Q_rot
    return np.sqrt((diff * diff).sum() / P.shape[0])


def task3(lowest_coords, lowest_energy, model, scaler, extractor, feature_names):
    """
    Task 3: 扰动灵敏度分析（使用模型预测的最低能量结构）
    """
    print("\n=== Task 3: 扰动灵敏度分析（基于预测最低能量结构） ===")

    perturbed_structures = apply_local_perturbations(lowest_coords,
                                                     n_samples=PERTURB_N_SAMPLES,
                                                     magnitudes=PERTURB_MAGNITUDES,
                                                     perturb_ratio=PERTURB_RATIO,
                                                     seed=PERTURB_SEED)
    stability_results = []

    for mag, struct_list in perturbed_structures.items():
        true_energies, pred_energies, deltas, rmsd_values, stability_indices = [], [], [], [], []

        for coords in struct_list:
            try:
                feature_dict = extractor.extract_all_features(coords)
                X_df = pd.DataFrame([feature_dict])
                for col in feature_names:
                    if col not in X_df.columns:
                        X_df[col] = 0.0
                X_df = X_df[feature_names]
                X_scaled = scaler.transform(X_df.values)
                pred_energy = model.predict(X_scaled)[0]
                delta = pred_energy - lowest_energy
                rmsd_val = kabsch_rmsd(lowest_coords, coords)
                stability_var = delta / rmsd_val if rmsd_val > 1e-8 else np.nan

                true_energies.append(lowest_energy)
                pred_energies.append(pred_energy)
                deltas.append(delta)
                rmsd_values.append(rmsd_val)
                stability_indices.append(stability_var)
            except:
                continue

        if not pred_energies:
            continue

        mae = mean_absolute_error(true_energies, pred_energies)
        rmse_val = np.sqrt(mean_squared_error(true_energies, pred_energies))

        print(f"\n扰动幅度 {mag} Å:")
        print(f"- 平均能量变化 (预测 ΔE): {np.mean(deltas):.4f} ± {np.std(deltas):.4f}")
        print(f"- 平均 RMSD: {np.mean(rmsd_values):.4f} ± {np.std(rmsd_values):.4f}")
        print(f"- 平均稳定性指数 (ΔE / RMSD): {np.nanmean(stability_indices):.4f} ± {np.nanstd(stability_indices):.4f}")
        print(f"- MAE: {mae:.4f}, RMSE: {rmse_val:.4f}")

        stability_results.append({
            "magnitude": mag,
            "mean_pred_delta": np.mean(deltas),
            "std_pred_delta": np.std(deltas),
            "mean_rmsd": np.mean(rmsd_values),
            "std_rmsd": np.std(rmsd_values),
            "mean_stability_index": np.nanmean(stability_indices),
            "std_stability_index": np.nanstd(stability_indices),
            "mae": mae,
            "rmse": rmse_val
        })

    # 可视化 ΔE vs RMSD
    plt.figure(figsize=(8, 6))
    for mag, struct_list in perturbed_structures.items():
        deltas, rmsd_vals = [], []
        for coords in struct_list:
            try:
                feature_dict = extractor.extract_all_features(coords)
                X_df = pd.DataFrame([feature_dict])
                for col in feature_names:
                    if col not in X_df.columns:
                        X_df[col] = 0.0
                X_df = X_df[feature_names]
                X_scaled = scaler.transform(X_df.values)
                pred_energy = model.predict(X_scaled)[0]
                deltas.append(pred_energy - lowest_energy)
                rmsd_vals.append(kabsch_rmsd(lowest_coords, coords))
            except:
                continue
        if deltas:
            plt.scatter(rmsd_vals, deltas, label=f"mag={mag} Å", alpha=0.6)

    plt.xlabel("RMSD (Å)")
    plt.ylabel("Predicted ΔE")
    plt.title("Energy Change vs Structural Perturbation")
    plt.legend()
    plt.show()

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

# 绘制折线图

    if stability_results:
        df_results = pd.DataFrame(stability_results)
        
        # 图1: 平均预测能量变化
        plt.figure(figsize=(10, 6))
        plt.plot(df_results["magnitude"], df_results["mean_pred_delta"], marker='o', color='blue', linewidth=2)
        plt.xlabel("Perturbation Magnitude (Å)")
        plt.ylabel("Mean ΔE (Predicted)")
        plt.title("Perturbation Magnitude vs Mean Predicted Energy Change")
        plt.grid(True)
        plt.show()
        
        # 图2: 平均RMSD
        plt.figure(figsize=(10, 6))
        plt.plot(df_results["magnitude"], df_results["mean_rmsd"], marker='s', color='red', linewidth=2)
        plt.xlabel("Perturbation Magnitude (Å)")
        plt.ylabel("Mean RMSD")
        plt.title("Perturbation Magnitude vs Mean RMSD")
        plt.grid(True)
        plt.show()
        
        # 图3: 平均稳定性指数
        plt.figure(figsize=(10, 6))
        plt.plot(df_results["magnitude"], df_results["mean_stability_index"], marker='^', color='green', linewidth=2)
        plt.xlabel("Perturbation Magnitude (Å)")
        plt.ylabel("Mean Stability Index (ΔE/RMSD)")
        plt.title("Perturbation Magnitude vs Mean Stability Index")
        plt.grid(True)
        plt.show()
    return pd.DataFrame(stability_results)
