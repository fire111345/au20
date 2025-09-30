import numpy as np
import pandas as pd
from pathlib import Path
from feature_extractor import Au20FeatureExtractor


def load_dataset(data_dir):
    """加载数据集"""
    data_dir = Path(data_dir)
    xyz_files = list(data_dir.glob("*.xyz"))
    
    features_list = []
    energies = []
    extractor = Au20FeatureExtractor()
    
    print(f"找到 {len(xyz_files)} 个XYZ文件")
    
    for i, file_path in enumerate(xyz_files):
        if i % 100 == 0:
            print(f"处理第 {i} 个文件...")
        
        try:
            coordinates, energy, symbols = extractor.parse_xyz_file(file_path)
            
            # 检查数据有效性
            if coordinates is None or len(coordinates) != 20:
                print(f"警告: 文件 {file_path.name} 原子数不是20，跳过")
                continue
                
            if np.isnan(energy) or not np.isfinite(energy):
                print(f"警告: 文件 {file_path.name} 能量值无效，跳过")
                continue
                
            features = extractor.extract_all_features(coordinates)
            features_list.append(features)
            energies.append(energy)
            
        except Exception as e:
            print(f"处理文件 {file_path.name} 时出错: {e}")
            continue
    
    # 创建DataFrame
    if features_list:
        features_df = pd.DataFrame(features_list)
        energies = np.array(energies)
        print(f"成功加载 {len(features_list)} 个有效样本")
    else:
        features_df = pd.DataFrame()
        energies = np.array([])
        print("警告: 没有成功加载任何有效样本！")
    
    return features_df, energies
