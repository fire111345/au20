from ase.visualize import view
from feature_extractor import Au20FeatureExtractor
from model_utils import analyze_energy_distribution, find_lowest_energy_structure, describe_geometry


def task2(data_dir, energies):
    """执行Task 2：能量分布 + 最稳定结构分析"""
    extractor = Au20FeatureExtractor()

    # 统计分析
    analyze_energy_distribution(energies)

    # 找到最低能量结构
    atoms, coords = find_lowest_energy_structure(data_dir, extractor, energies)

    # 可视化
    print("\n正在可视化最低能量结构...")
    view(atoms)  # 弹出 ASE 窗口（在Jupyter或Linux下也可以转化成PNG）

    # 描述几何特征
    describe_geometry(coords)


