########################################
############特征选择流程################
#######################################
import argparse, sys, os, errno
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('machine_learning')
import json
from tqdm import tqdm


def read_data_matrix(matrix, sample_classes,group=False ,features=None, transpose=True, positive_class=None, negative_class=None):

    import pandas as pd
    import numpy as np
    # read data matrix
    logger.info('read data matrix: ' + matrix)
    X = pd.read_table(matrix, index_col=0, sep='\t')
    if group==False:
        logger.info('group: {}'.format(group))
        X = X
    elif group=='70_90':
        logger.info('group: {}'.format(group))
        for i in range(len(X.T)):
            X_ = X.iloc[:, i]
            X_ = np.array(X_)
            X_cut_d = np.percentile(X_, 70)
            X_cut_g = np.percentile(X_, 90)
            X.iloc[:, i] = X.iloc[:, i].apply(lambda x: 1 if x <= X_cut_d else 2 if x <= X_cut_g else 3)
    elif group=='0_1':
        logger.info('group: {}'.format(group))
        # for i in range(len(X.T)):
        #     X_ = X.iloc[:, i]
        #     X_ = np.array(X_)
        #     X_cut_d = 0
        #     X_cut_g = 0.5
        #     X.iloc[:, i] = X.iloc[:, i].apply(lambda x: 1 if x <= X_cut_d else 2 if x <= X_cut_g else 3)
        for i in range(len(X.T)):
            X_ = X.iloc[:, i]
            X_ = np.array(X_)
            X_cut_d = 0
            X_cut_g = 0.5
            X.iloc[:, i] = X.iloc[:, i].apply(lambda x: 0 if x <= 0 else 1)

    # transpose
    if transpose:
        logger.info('transpose feature matrix')
        X = X.T
    if features is not None:
        logger.info('read subset of feature names from: ' + features)
        features = pd.read_table(features, header=None).iloc[:, 0].values
        logger.info('select {} features'.format(len(features)))
        X = X.reindex(columns=features)
        ###给定特征中全是空值#####
        is_na_features = X.isna().any(axis=0)
        na_features = X.columns.values[is_na_features]
        if na_features.shape[0] > 0:
            logger.warning('missing features found in matrix file: {}'.format(na_features[:10].tolist()))
            # X = X.loc[:, ~is_na_features]
            # raise ValueError('some given features are not found in matrix file: {}'.format(na_features[:10].tolist()))
    logger.info('number of features: {}'.format(X.shape[1]))

    # read sample classes
    logger.info('read sample classes: ' + sample_classes)
    sample_classes = pd.read_table(sample_classes, index_col=0, sep='\t')
    sample_classes = sample_classes.iloc[:, 0]
    sample_classes = sample_classes.loc[X.index.values]
    logger.info('sample_classes: {}'.format(sample_classes.shape[0]))

    # get positive and negative classes
    if (positive_class is not None) and (negative_class is not None):
        pass
    else:
        unique_classes = np.unique(sample_classes.values)
        if len(unique_classes) != 2:
            raise ValueError('expect 2 classes but {} classes found'.format(len(unique_classes)))
        positive_class, negative_class = unique_classes
    positive_class = np.atleast_1d(positive_class)
    negative_class = np.atleast_1d(negative_class)
    # select positive samples and negative samples
    logger.info('positive class: {}'.format(positive_class))
    logger.info('negative class: {}'.format(negative_class))
    X_pos = X.loc[sample_classes[sample_classes.isin(positive_class)].index.values]
    X_neg = X.loc[sample_classes[sample_classes.isin(negative_class)].index.values]
    logger.info('number of positive samples: {}, negative samples: {}, class ratio: {}'.format(
        X_pos.shape[0], X_neg.shape[0], float(X_pos.shape[0]) / X_neg.shape[0]))
    X_new = pd.concat([X_pos, X_neg], axis=0)
    # set negative class to 0 and positive class to 1
    y = np.zeros(X_new.shape[0], dtype=np.int32)
    y[:X_pos.shape[0]] = 1
    del X_pos
    del X_neg
    #n_samples, n_features = X_new.shape
    sample_ids = X_new.index.values
    feature_names = X_new.columns.values
    X_new = X_new.values

    return X_new, y, sample_ids, feature_names



def run_pipeline(args):
    import sys
    sys.path.append('/BioII/lulab_b/liuxiaofan/project/pico_feature_select/bin/')
    from estimators2 import cross_validation as _cross_validation
    from estimators2 import search_dict,CollectMetrics, CollectPredictions, CollectTrainIndex, FeatureSelectionMatrix,\
        CombinedEstimator
    from sklearn.utils.class_weight import compute_sample_weight
    import pandas as pd
    import numpy as np
    import h5py
    import pickle
    import yaml
    #from sklearn.externals import joblib
    ##########################
    #######输入部分###########
    ##########################
    ### 参数读取
    logger.info('read configuration file: ' + args.config)
    with open(args.config, 'r') as f:
        config = yaml.load(f)
    if args.matrix is not None:
        config['matrix'] = args.matrix
    if args.sample_classes is not None:
        config['sample_classes'] = args.sample_classes
    if args.positive_class is not None:
        config['positive_class'] = args.positive_class.split(',')
    if args.negative_class is not None:
        config['negative_class'] = args.negative_class.split(',')
    if args.features is not None:
        config['features'] = args.features
    if args.group is not None:
        config['group'] = args.group
    else:
        config['group']=False

    ### 输入数据读入
    X, y, sample_ids, feature_names = read_data_matrix(
        config['matrix'], config['sample_classes'],config['group'],
        **search_dict(config, ('features', 'transpose', 'positive_class', 'negative_class')))

    # fill missing features
    X = np.nan_to_num(X)
    if X.shape[0] < 20:
        raise ValueError('too few samples for machine learning')

    # get feature weight
    if config.get('sample_weight') is not None:
        if config['sample_weight'] == 'balanced':
            logger.info('compute sample weight from class ratio: balanced')
            sample_weight = 'balanced'
        else:
            logger.info('read sample weight from file: ' + config['sample_weight'])
            sample_weight = pd.read_table(
                config['sample_weight'], header=None, index_col=0).iloc[:, 0]
    else:
        sample_weight = None

    ##########################
    #######计算部分###########
    ##########################

    ### 处理config参数，确定数据处理步骤
    preprocess_steps = {}
    for step_dict in config['preprocess_steps']:
        step_tuple = tuple(step_dict.items())[0]
        preprocess_steps[step_tuple[0]] = step_tuple[1]
        preprocess_steps[step_tuple[0]]['params'] = step_tuple[1].get('params', {})

    logger.info('get cross-validation fucsion')

    ####交叉验证选择特征函数######
    estimator = CombinedEstimator(config)
    ####输出结果函数#######
    collect_metrics = CollectMetrics()
    collect_predictions = CollectPredictions()
    collect_train_index = CollectTrainIndex()
    cv_callbacks = [collect_metrics, collect_predictions, collect_train_index]
    has_selector = False
    for step in preprocess_steps.values():
        if step['type'] == 'selector':
            has_selector = True
    if has_selector:
        logger.info('add cross-validation callback: FeatureSelectionMatrix')
        feature_selection_matrix = FeatureSelectionMatrix()
        cv_callbacks.append(feature_selection_matrix)

    logger.info('start cross-validation')
    _cross_validation(estimator, X, y, sample_weight=sample_weight,params=config['cv_params'], callbacks=cv_callbacks)

    ### 输出数据概况
    if not os.path.isdir(args.output_dir):
        logger.info('create output directory: ' + args.output_dir)
        os.makedirs(args.output_dir)
    # read other input files
    logger.info('save class labels to: ' + os.path.join(args.output_dir, 'classes.txt'))
    pd.Series(y).to_csv(os.path.join(args.output_dir, 'classes.txt'), header=False, index=False)
    logger.info('save feature names to: ' + os.path.join(args.output_dir, 'feature_names.txt'))
    pd.Series(feature_names).to_csv(os.path.join(args.output_dir, 'feature_names.txt'), header=False, index=False)
    logger.info('save sample ids to: ' + os.path.join(args.output_dir, 'samples.txt'))
    pd.Series(sample_ids).to_csv(os.path.join(args.output_dir, 'samples.txt'), header=False, index=False)

    metrics = collect_metrics.get_metrics()
    for name in ('train', 'test'):
        logger.info('save metrics to: ' + os.path.join(args.output_dir, 'metrics_{}.txt'.format(name)))
        # if there are missing features, set metrics to NA
        pd.DataFrame(metrics[name]).to_csv(os.path.join(args.output_dir, 'metrics.{}.txt'.format(name)), header=True, index=True,na_rep='NA', sep='\t')

    logger.info('save cross-validation details to: ' + os.path.join(args.output_dir, 'cross_validation.h5'))
    with h5py.File(os.path.join(args.output_dir, 'cross_validation.h5'), 'w') as f:
        f.create_dataset('labels', data=y)
        f.create_dataset('predicted_labels', data=collect_predictions.get_pred_labels())
        f.create_dataset('predictions', data=collect_predictions.get_pred_probs())
        f.create_dataset('train_index', data=collect_train_index.get_train_index())
        # print(feature_selection_matrix.get_matrix())
        if has_selector:
            f.create_dataset('feature_selection', data=feature_selection_matrix.get_matrix())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Machine learning module')
    parser.add_argument('--matrix', '-i', type=str, metavar='FILE', required=True,
        help='input feature matrix (rows are samples and columns are features',dest='matrix')
    parser.add_argument('--sample-classes', type=str, metavar='FILE', required=True,
        help='input file containing sample classes with 2 columns: sample_id, sample_class',dest='sample_classes')
    parser.add_argument('--positive-class', type=str,
        help='comma-separated list of sample classes to use as positive class',dest='positive_class')
    parser.add_argument('--negative-class', type=str,
        help='comma-separates list of sample classes to use as negative class',dest='negative_class')
    parser.add_argument('--features', type=str, metavar='FILE',
        help='input file containing subset of feature names',dest='features')
    parser.add_argument('--config', '-c', type=str, metavar='FILE', required=True,
        help='configuration file of parameters in YAML format',dest='config')
    parser.add_argument('--output-dir', '-o', type=str, metavar='DIR',
        required=True, help='output directory',dest='output_dir')
    parser.add_argument('--group', type=str,
        help='group',dest='group')


    args = parser.parse_args()
    run_pipeline(args)
    # args = argparse.Namespace(matrix='/BioII/lulab_b/liuxiaofan/project/pico_feature_select/pico_0530/count/output_discovery_1/CRC_20/featurecounts.CRC.up.txt',
    #                           sample_classes='/BioII/lulab_b/liuxiaofan/project/pico_feature_select/row_data_2/sample_classes.txt',
    #                           config = '/BioII/lulab_b/liuxiaofan/project/pico_feature_select/pico_0530/count/config/config_20.yaml',
    #                           positive_class='CRC',
    #                           negative_class='NC',
    #                           output_dir='/BioII/lulab_b/liuxiaofan/project/pico_feature_select/pico_0530/count/output_discovery_1/CRC_20/cv_rf_1/',
    #                           features=None,
    #                           group=False)


