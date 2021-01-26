## 特征选择脚本

对exSeek进行特征选择模块进行重构，主要改进是ReliefF方法的加速，只要十分之一时间。


### 1) 主函数说明

/BioII/lulab_b/liuxiaofan/project/pico_feature_select/bin/pipeline.py

该脚本为主函数，其中

--matrix 数据矩阵
/BioII/lulab_b/liuxiaofan/project/pico_feature_select/pico_0530/count/output_discovery_1/CRC_20/featurecounts.CRC.up.txt' 

--sample-classes 样本标签
/BioII/lulab_b/liuxiaofan/project/pico_feature_select/row_data_2/sample_classes.txt

--config 参数配置文件 
/BioII/lulab_b/liuxiaofan/project/pico_feature_select/pico_0530/count/config/config_20.yaml

--positive-class 'CRC' 

--negative-class 'NC' 

--output-dir 输出文件夹
/BioII/lulab_b/liuxiaofan/project/pico_feature_select/pico_0530/count/output_discovery_1/CRC_20/cv_rf_1/

--group 进行离散化处理


### 2) 涉及函数说明

1. 特征筛选+模型估计

/BioII/lulab_b/liuxiaofan/project/pico_feature_select/bin/estimators2.py

2.relief

/BioII/lulab_b/liuxiaofan/project/pico_feature_select/bin/relief_class.py

3.差异表达ttest

/BioII/lulab_b/liuxiaofan/project/pico_feature_select/bin/Difference_analysis.py

4.基本函数

/BioII/lulab_b/liuxiaofan/project/pico_feature_select/bin/utils.py


### 3) 配置文件说明

```python
#预处理步骤：scaler，log，特征选择
preprocess_steps:
- scale_features:
    enabled: False
    name: StandardScaler
    params:
      with_mean: true
    type: scaler
- log_transform:
    enabled: False
    name: LogTransform
    params:
      base: 2
      pseudo_count: 1
    type: transformer

#ReliefF方法
#- feature_selection:
#    enabled: true
#    name: ReliefF
#    params:
#      n_jobs: 1
#      n_neighbors: 50
#    type: selector

#差异表达方法
#- feature_selection:
#    enabled: true
#    name: Diff
#    params:
#      n_jobs: 1
#      n_neighbors: 50
#    type: selector

#MaxFeatures_RandomForestClassifier方法,可参考/BioII/lulab_b/liuxiaofan/exSeek/exSeek-dev/config/machine_learning.yaml中selectors

- feature_selection:
    enabled: true
    type: selector
    name: MaxFeatures
    params:
      classifier: RandomForestClassifier
      grid_search: true
      grid_search_params:
        cv: 5
        iid: false
        param_grid:
          max_depth:
          - 3
          - 4
          - 5
          n_estimators:
          - 25
          - 50
          - 75
        scoring: roc_auc

#特征选择个数
n_features_to_select: 5
#交叉验证参数
cv_params:
  n_repeats: 1
  n_splits: 5
  scoring: roc_auc
  shuffle: true
  splitter: RepeatedStratifiedKFold
  test_size: 0.2
 
#最后的分类器    
classifier: RandomForestClassifier
classifier_params: {} #参数
grid_search: true #是否进行参数寻优
grid_search_params:
  cv:
    n_splits: 5
    splitter: StratifiedShuffleSplit
    test_size: 0.1
  iid: false
  param_grid:
    max_depth:
    - 3
    - 4
    - 5
    n_estimators:
    - 25
    - 50
    - 75
  scoring: roc_auc

sample_weight: balanced
transpose: true
#给出特征list
features: null

```



