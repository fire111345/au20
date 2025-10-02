import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from ase import Atoms
import numpy as np
from scipy.stats import skew
import shap
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, X_test, y_test):
    """评估模型性能"""
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

    return mae, rmse, r2, y_pred

def plot_feature_importance(model, X, feature_names, top_n=20, save_path=None):
    print("计算 SHAP 值中...")
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    mean_shap = np.abs(shap_values.values).mean(axis=0)
    indices = np.argsort(mean_shap)[::-1]

    plt.figure(figsize=(12, 8))
    plt.bar(range(min(top_n, len(mean_shap))), mean_shap[indices][:top_n])
    plt.xticks(range(min(top_n, len(mean_shap))),
               [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
    plt.title("Feature Importance (SHAP)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"SHAP特征重要性图已保存: {save_path}")
    plt.show()


def analyze_energy_distribution(energies, save_path=None):
    import seaborn as sns
    mean_energy = np.mean(energies)
    var_energy = np.var(energies)
    skew_energy = skew(energies)

    plt.figure(figsize=(8, 5))
    sns.histplot(energies, kde=True, bins=30, color="gold", alpha=0.7)
    plt.axvline(mean_energy, color="red", linestyle="--", label="Mean")
    plt.xlabel("Total Energy")
    plt.ylabel("Frequency")
    plt.title("Energy Distribution of Au20 Clusters")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        print(f"能量分布图已保存: {save_path}")
    plt.show()
    return mean_energy, var_energy, skew_energy


def find_lowest_energy_structure(data_dir, extractor, energies):
    """找到最低能量结构并返回坐标"""
    data_dir = Path(data_dir)
    xyz_files = list(data_dir.glob("*.xyz"))

    min_idx = np.argmin(energies)
    lowest_file = xyz_files[min_idx]

    coords, energy, symbols = extractor.parse_xyz_file(lowest_file)
    print(f"\n最低能量结构文件: {lowest_file.name}")
    print(f"能量值: {energy:.4f}")

    atoms = Atoms(symbols=symbols, positions=coords)
    return atoms, coords

def describe_geometry(coords):
    """总结几何特征（键长、分布）"""
    dist_matrix = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=2)
    triu_idx = np.triu_indices_from(dist_matrix, k=1)
    distances = dist_matrix[triu_idx]

    mean_bond = np.mean(distances)
    min_bond = np.min(distances)
    max_bond = np.max(distances)

    print("\n=== 低能量结构几何特征 ===")
    print(f"平均键长: {mean_bond:.4f} Å")
    print(f"最短键长: {min_bond:.4f} Å")
    print(f"最长键长: {max_bond:.4f} Å")

    plt.figure(figsize=(8, 5))
    sns.histplot(distances, bins=30, kde=True, color="blue", alpha=0.6)
    plt.xlabel("Bond Length (Å)")
    plt.ylabel("Count")
    plt.title("Bond Length Distribution of Lowest Energy Structure")
    plt.show()
