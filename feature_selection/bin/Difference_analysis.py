import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore") #不显示警告
import statsmodels.stats.multitest as multitest
from numba import jit
from sklearn import preprocessing

def BH(p_vulae):
    p_vulae = np.array(p_vulae)
    p_BH = multitest.multipletests(p_vulae,alpha = 0.05,method = 'fdr_bh',is_sorted = False,returnsorted = False)[1]

    return p_BH

@jit(nopython=False)
def wilcoxontest_feature(X_0,X_1):
    p_vulae=[]
    for i in range(len(X_0.T)):
        m1 = X_0[:,i]
        m2 = X_1[:,i]
        if m1.mean() == 0: m1 = m1 + 0.00001
        if m2.mean() == 0: m2 = m2 + 0.00001
        if (len(set(m1)) == 1) & (len(set(m2)) == 1):
            p = np.nan
        elif ((len(set(m1)) == 1) | (len(set(m2)) == 1))&(len(set(m1))+len(set(m2))>2):
            t, p = stats.ranksums(m1, m2)
        else:
            t, p = stats.mannwhitneyu(m1, m2)
        p_vulae.append(p)
    return np.array(p_vulae)

# @jit(nopython=False)
# def FC_test(X_0,X_1):
#     FC=[]
#     for i in range(len(X_0.T)):
#         m1 = X_0[:,i]
#         m2 = X_1[:,i]
#         if m1.mean() == 0: m1_mean = m1.mean()+0.00001
#         else: m1_mean = m1.mean()
#         if m2.mean() == 0: m2_mean = m2.mean()+0.00001
#         else: m2_mean = m2.mean()
#         FC_ = m2_mean/m1_mean
#         FC.append(FC_)
#     return np.array(FC)

@jit(nopython=False)
def FC_test(X_0,X_1):
    FC=[]
    for i in range(len(X_0.T)):
        m1 = X_0[:,i]
        m2 = X_1[:,i]
        if (m1.mean() == 0)|(m2.mean() == 0):
            FC_=0
        else:
            m2_mean = m2.mean()
            m1_mean = m1.mean()
            FC_ = m2_mean/m1_mean
        FC.append(FC_)
    return np.array(FC)

@jit(nopython=False)
def get_gini(X):
    sorted = np.sort(X)
    height, area = 0, 0
    for i in range(0, len(sorted)):
        height += sorted[i]
        area += height - sorted[i] / 2.
    fair_area = height * len(sorted) / 2.
    if fair_area == 0:
        gini = np.nan
    else:
        gini = (fair_area - area) / fair_area
    return gini

@jit(nopython=False)
def gini_index(X_0,X_1):
    gini_NC =[]
    gini_cancer =[]
    for i in range(len(X_0.T)):
        m1 = X_0[:,i]
        m2 = X_1[:,i]
        gini_NC_ = get_gini(m1)
        gini_cancer_ = get_gini(m2)
        gini_NC.append(gini_NC_)
        gini_cancer.append(gini_cancer_)
    return np.array(gini_NC),np.array(gini_cancer)


class Difference_test(object):


    def __init__(self,n_features_to_keep=10):

        self.n_features_to_keep = n_features_to_keep

    def fit(self, X, y):
        """Computes the feature importance scores from the training data.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training labels
        }

        Returns
        -------
        None

        """
        if (X.min()<0)|(X.max()>1):
            min_max_scaler = preprocessing.MinMaxScaler()
            self.X_ = min_max_scaler.fit_transform(X.T)
            self.X_ = self.X_.T
        else:
            self.X_ = X
        X_0 = self.X_[y==0,:]
        X_1 = self.X_[y==1,:]
        p_vulae = wilcoxontest_feature(X_0,X_1)
        FC = FC_test(X_0,X_1)
        gini_NC,gini_cancer = gini_index(X_0,X_1)
        p_vulae_BH = BH(p_vulae)
        feature_index = np.where(p_vulae_BH<0.05)[0]
        if len(feature_index)==0:
            feature_index = np.where(p_vulae<0.05)[0]
            if len(feature_index)==0:
                feature_index = np.where(p_vulae < 0.1)[0]
                if len(feature_index)==0:
                    feature_index = np.where(p_vulae < 2)[0]
        if len(feature_index)==0:
            feature_index = np.array([i for i in range(len(p_vulae))])
        self.feature_rank = pd.DataFrame(columns=['feature', 'FC','gini_NC','gini_cancer'])
        self.feature_rank['feature'] = feature_index
        self.feature_rank['FC'] = list(FC[feature_index])
        self.feature_rank['gini_NC'] = list(gini_NC[feature_index])
        self.feature_rank['gini_cancer'] = list(gini_cancer[feature_index])
        # self.feature_rank['FC_new'] = self.feature_rank['FC'].map(lambda x:1/x if x<1 else x)
        self.feature_rank['FC_new'] = self.feature_rank['FC'].map(lambda x: 1 / x if 0<x < 1 else x)

        self.feature_rank['feature_FC_rank'] = self.feature_rank['FC_new'].rank(ascending=False)
        self.feature_rank['gini_NC_rank'] = self.feature_rank['FC_new'].rank(ascending=True)
        self.feature_rank['gini_cancer_rank'] = self.feature_rank['FC_new'].rank(ascending=True)
        self.feature_rank['rank_sum'] = self.feature_rank['feature_FC_rank'] +self.feature_rank['gini_cancer_rank']
        # feature_rank['rank'] = feature_rank['rank_sum'].rank()
        self.feature_rank['rank'] = self.feature_rank['feature_FC_rank']
        self.top_features = np.array(self.feature_rank.loc[self.feature_rank['rank']<=self.n_features_to_keep,'feature'])
        if len(self.top_features)==0:
            self.top_features = np.array([i for i in range(self.n_features_to_keep)])
        return self

    def transform(self, X):
        """Reduces the feature set down to the top `n_features_to_keep` features.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Feature matrix to perform feature selection on

        Returns
        -------
        X_reduced: array-like {n_samples, n_features_to_keep}
            Reduced feature matrix

        """
        return X[:, self.top_features]

    def fit_transform(self, X, y):
        """Computes the feature importance scores from the training data, then
        reduces the feature set down to the top `n_features_to_keep` features.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training labels

        Returns
        -------
        X_reduced: array-like {n_samples, n_features_to_keep}
            Reduced feature matrix

        """
        self.fit(X, y)
        return self.transform(X)

# X = np.array([[1.61311827, 0.19955703,0],
#                 [-0.21997067, 0.86474714,1],
#                 [-0.58658846, -1.46341823,0],
#                 [-1.31982404, -0.79822813,0],
#                 [0.5132649, 1.19734219,1],
#               [-0.5132649, -0.19734219,1],
#               [1.5132649, 0.9734219,1]])
#
# y = np.array([1, 1, 1, 0, 0,0,0])
# fs = Difference_test(n_features_to_keep=2)
# fs.fit(X, y)
# fs.transform(X)
# features_ = np.arange(X.shape[1]).reshape((1, -1))
# features_ = fs.transform(features_)