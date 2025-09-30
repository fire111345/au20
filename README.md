## 项目结构

- `config.py` – 数据集路径和部分模型参数。  
- `data_loader.py` – 加载 XYZ 文件并准备数据集的函数。  
- `feature_extractor.py` – 从 Au$_{20}$ 团簇提取几何、电子和拓扑特征。  
- `model_utils.py` – 模型训练、评估和交叉验证函数（XGBoost）。  
- `task2.py` – 统计分析和最低能量结构表征。  
- `task3.py` – 稳定性局部扰动分析。  
- `main.py` – 主运行脚本，运行。

## 依赖环境

- Python >= 3.8  

- numpy
- pandas
- scikit-learn
- xgboost
- scikit-optimize
- ase
- dscribe
- scipy
- networkx
- matplotlib
- seaborn
- shap
