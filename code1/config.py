DATA_DIR = "C:/Users/DKE/Desktop/cyber c/data/Au20_OPT_1000"
SAVE_DIR = "saved_models"

# 扰动参数配置
PERTURB_N_SAMPLES = 200    
PERTURB_MAGNITUDES = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.015, 0.02]
PERTURB_RATIO = 1  #被扰动的原子数量占总原子数的比例。
PERTURB_SEED = 42



#自动调参结果:
#最佳参数: OrderedDict([('colsample_bytree', 0.9524879318330092), ('learning_rate', 0.02056255691138276), ('max_depth', 3), ('min_child_weight', 1), ('n_estimators', 1967), ('reg_alpha', 3.0746072569950025e-07), ('reg_lambda', 0.19273134791406898), ('subsample', 0.5391211138124519)])
#最佳模型: XGBRegressor(base_score=None, booster=None, callbacks=None,
             #colsample_bylevel=None, colsample_bynode=None,
             #colsample_bytree=0.9524879318330092, device=None,
             #early_stopping_rounds=None, enable_categorical=False,
             #eval_metric=None, feature_types=None, feature_weights=None,
             #gamma=None, grow_policy=None, importance_type=None,
             #interaction_constraints=None, learning_rate=0.02056255691138276,
             #max_bin=None, max_cat_threshold=None, max_cat_to_onehot=None,
             #max_delta_step=None, max_depth=3, max_leaves=None,
             #min_child_weight=1, missing=nan, monotone_constraints=None,
             #multi_strategy=None, n_estimators=1967, n_jobs=-1,
             #num_parallel_tree=None, ...)


#XGBRegressor(
    #colsample_bytree=0.9524,
    #learning_rate=0.0206,
    #max_depth=3,
    #min_child_weight=1,
    #n_estimators=1967,
    #reg_alpha=3.07e-07,
    #reg_lambda=0.1927,
    #subsample=0.5391,
    #...
#)


