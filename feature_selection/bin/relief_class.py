from __future__ import print_function
import numpy as np
import pandas as pd
import time
import warnings
import sys
from sklearn.base import BaseEstimator
from sklearn.externals.joblib import Parallel, delayed


def get_row_missing(xc, xd, cdiffs, index, cindices, dindices):
    """ Calculate distance between index instance and all other instances. """
    row = np.empty(0, dtype=np.double)  # initialize empty row
    cinst1 = xc[index]  # continuous-valued features for index instance
    dinst1 = xd[index]  # discrete-valued features for index instance
    # Boolean mask locating missing values for continuous features for index instance
    can = cindices[index]
    # Boolean mask locating missing values for discrete features for index instance
    dan = dindices[index]
    tf = len(cinst1) + len(dinst1)  # total number of features.

    # Progressively compare current instance to all others. Excludes comparison with self indexed instance. (Building the distance matrix triangle).
    for j in range(index):
        dist = 0
        dinst2 = xd[j]  # discrete-valued features for compared instance
        cinst2 = xc[j]  # continuous-valued features for compared instance

        # Manage missing values in discrete features
        # Boolean mask locating missing values for discrete features for compared instance
        dbn = dindices[j]
        # indexes where there is at least one missing value in the feature between an instance pair.
        idx = np.unique(np.append(dan, dbn))
        # Number of features excluded from distance calculation due to one or two missing values within instance pair. Used to normalize distance values for comparison.
        dmc = len(idx)
        d1 = np.delete(dinst1, idx)  # delete unique missing features from index instance
        d2 = np.delete(dinst2, idx)  # delete unique missing features from compared instance

        # Manage missing values in continuous features
        # Boolean mask locating missing values for continuous features for compared instance
        cbn = cindices[j]
        # indexes where there is at least one missing value in the feature between an instance pair.
        idx = np.unique(np.append(can, cbn))
        # Number of features excluded from distance calculation due to one or two missing values within instance pair. Used to normalize distance values for comparison.
        cmc = len(idx)
        c1 = np.delete(cinst1, idx)  # delete unique missing features from index instance
        c2 = np.delete(cinst2, idx)  # delete unique missing features from compared instance
        # delete unique missing features from continuous value difference scores
        cdf = np.delete(cdiffs, idx)

        # Add discrete feature distance contributions (missing values excluded) - Hamming distance
        dist += len(d1[d1 != d2])

        # Add continuous feature distance contributions (missing values excluded) - Manhattan distance (Note that 0-1 continuous value normalization is included ~ subtraction of minimums cancel out)
        dist += np.sum(np.absolute(np.subtract(c1, c2)) / cdf)

        # Normalize distance calculation based on total number of missing values bypassed in either discrete or continuous features.
        tnmc = tf - dmc - cmc  # Total number of unique missing counted
        # Distance normalized by number of features included in distance sum (this seeks to handle missing values neutrally in distance calculation)
        dist = dist/float(tnmc)

        row = np.append(row, dist)

    return row



class ReliefF(BaseEstimator):
    """Feature selection using data-mined expert knowledge.
    Based on the ReliefF algorithm as introduced in:
    Igor et al. Overcoming the myopia of inductive learning
    algorithms with RELIEFF (1997), Applied Intelligence, 7(1), p39-55"""

    """Note that ReliefF class establishes core functionality that is inherited by all other Relief-based algorithms.
    Assumes: * There are no missing values in the label/outcome/dependent variable.
             * For ReliefF, the setting of k is <= to the number of instances that have the least frequent class label
             (binary and multiclass endpoint data. """

    def __init__(self, n_features_to_select=10, n_neighbors=100, discrete_threshold=10, verbose=False, n_jobs=1):
        """Sets up ReliefF to perform feature selection. Note that an approximation of the original 'Relief'
        algorithm may be run by setting 'n_features_to_select' to 1. Also note that the original Relief parameter 'm'
        is not included in this software. 'm' specifies the number of random training instances out of 'n' (total
        training instances) used to update feature scores. Since scores are most representative when m=n, all
        available training instances are utilized in all Relief-based algorithm score updates here. If the user
        wishes to utilize a smaller 'm' in Relief-based scoring, simply pass any of these algorithms a subset of the
        original training dataset samples.

        Parameters
        ----------
        n_features_to_select: int (default: 10)
            the number of top features (according to the relieff score) to
            retain after feature selection is applied.
        n_neighbors: int or float (default: 100)
            The number of neighbors to consider when assigning feature
            importance scores. If a float number is provided, that percentage of
            training samples is used as the number of neighbors.
            More neighbors results in more accurate scores, but takes longer.
        discrete_threshold: int (default: 10)
            Value used to determine if a feature is discrete or continuous.
            If the number of unique levels in a feature is > discrete_threshold, then it is
            considered continuous, or discrete otherwise.
        verbose: bool (default: False)
            If True, output timing of distance array and scoring
        n_jobs: int (default: 1)
            The number of cores to dedicate to computing the scores with joblib.
            Assigning this parameter to -1 will dedicate as many cores as are available on your system.
            We recommend setting this parameter to -1 to speed up the algorithm as much as possible.

        """
        self.n_features_to_select = n_features_to_select
        self.n_neighbors = n_neighbors
        self.discrete_threshold = discrete_threshold
        self.verbose = verbose
        self.n_jobs = n_jobs

    # =========================================================================#
    def fit(self, X, y):
        """Scikit-learn required: Computes the feature importance scores from the training data.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training labels

        Returns
        -------
        Copy of the ReliefF instance

        """
        self._X = X  # matrix of predictive variables ('independent variables')
        self._y = y  # vector of values for outcome variable ('dependent variable')

        # Set up the properties for ReliefF -------------------------------------------------------------------------------------
        self._datalen = len(self._X)  # Number of training instances ('n')

        """"Below: Handles special case where user requests that a proportion of training instances be neighbors for
        ReliefF rather than a specified 'k' number of neighbors.  Note that if k is specified, then k 'hits' and k
        'misses' will be used to update feature scores.  Thus total number of neighbors is 2k. If instead a proportion
        is specified (say 0.1 out of 1000 instances) this represents the total number of neighbors (e.g. 100). In this
        case, k would be set to 50 (i.e. 50 hits and 50 misses). """
        if hasattr(self, 'n_neighbors') and type(self.n_neighbors) is float:
            # Halve the number of neighbors because ReliefF uses n_neighbors matches and n_neighbors misses
            self.n_neighbors = int(self.n_neighbors * self._datalen * 0.5)

        # Number of unique outcome (label) values (used to determine outcome variable type)
        self._label_list = list(set(self._y))
        # Determine if label is discrete
        discrete_label = (len(self._label_list) <= self.discrete_threshold)

        # Identify label type (binary, multiclass, or continuous)
        if discrete_label:
            if len(self._label_list) == 2:
                self._class_type = 'binary'
                self.mcmap = 0
            elif len(self._label_list) > 2:
                self._class_type = 'multiclass'
                self.mcmap = self._getMultiClassMap()
            else:
                raise ValueError('All labels are of the same class.')

        else:
            self._class_type = 'continuous'
            self.mcmap = 0

        # Training labels standard deviation -- only used if the training labels are continuous
        self._labels_std = 0.
        if len(self._label_list) > self.discrete_threshold:
            self._labels_std = np.std(self._y, ddof=1)

        self._num_attributes = len(self._X[0])  # Number of features in training data

        # Number of missing data values in predictor variable matrix.
        self._missing_data_count = np.isnan(self._X).sum()

        """Assign internal headers for the features (scikit-learn does not accept external headers from dataset):
        The pre_normalize() function relies on the headers being ordered, e.g., X01, X02, etc.
        If this is changed, then the sort in the pre_normalize() function needs to be adapted as well. """
        xlen = len(self._X[0])
        mxlen = len(str(xlen + 1))
        self._headers = ['X{}'.format(str(i).zfill(mxlen)) for i in range(1, xlen + 1)]

        start = time.time()  # Runtime tracking

        # Determine data types for all features/attributes in training data (i.e. discrete or continuous)
        C = D = False
        # Examines each feature and applies discrete_threshold to determine variable type.
        self.attr = self._get_attribute_info()
        for key in self.attr.keys():
            if self.attr[key][0] == 'discrete':
                D = True
            if self.attr[key][0] == 'continuous':
                C = True

        # For downstream computational efficiency, determine if dataset is comprised of all discrete, all continuous, or a mix of discrete/continuous features.
        if C and D:
            self.data_type = 'mixed'
        elif D and not C:
            self.data_type = 'discrete'
        elif C and not D:
            self.data_type = 'continuous'
        else:
            raise ValueError('Invalid data type in data set.')
        # --------------------------------------------------------------------------------------------------------------------

        # Compute the distance array between all data points ----------------------------------------------------------------
        # For downstream efficiency, separate features in dataset by type (i.e. discrete/continuous)
        diffs, cidx, didx = self._dtype_array()
        cdiffs = diffs[cidx]  # max/min continuous value difference for continuous features.

        xc = self._X[:, cidx]  # Subset of continuous-valued feature data
        xd = self._X[:, didx]  # Subset of discrete-valued feature data

        """ For efficiency, the distance array is computed more efficiently for data with no missing values.
        This distance array will only be used to identify nearest neighbors. """
        if self._missing_data_count > 0:
            self._distance_array = self._distarray_missing(xc, xd, cdiffs)
        else:
            self._distance_array = self._distarray_no_missing(xc, xd)

        if self.verbose:
            elapsed = time.time() - start
            print('Created distance array in {} seconds.'.format(elapsed))
            print('Feature scoring under way ...')

        start = time.time()
        # --------------------------------------------------------------------------------------------------------------------

        # Run remainder of algorithm (i.e. identification of 'neighbors' for each instance, and feature scoring).------------
        # Stores feature importance scores for ReliefF or respective Relief-based algorithm.
        self.feature_importances_ = self._run_algorithm()

        # Delete the internal distance array because it is no longer needed
        del self._distance_array

        if self.verbose:
            elapsed = time.time() - start
            print('Completed scoring in {} seconds.'.format(elapsed))

        # Compute indices of top features
        self.top_features_ = np.argsort(self.feature_importances_)[::-1]

        # return self.top_features_,self.feature_importances_
        return self
    # =========================================================================#
    def transform(self, X):
        """Scikit-learn required: Reduces the feature set down to the top `n_features_to_select` features.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Feature matrix to perform feature selection on

        Returns
        -------
        X_reduced: array-like {n_samples, n_features_to_select}
            Reduced feature matrix

        """
        if self._num_attributes < self.n_features_to_select:
            raise ValueError('Number of features to select is larger than the number of features in the dataset.')

        return X[:, self.top_features_[:self.n_features_to_select]]

    # =========================================================================#
    def fit_transform(self, X, y):
        """Scikit-learn required: Computes the feature importance scores from the training data, then reduces the feature set down to the top `n_features_to_select` features.

        Parameters
        ----------
        X: array-like {n_samples, n_features}
            Training instances to compute the feature importance scores from
        y: array-like {n_samples}
            Training labels

        Returns
        -------
        X_reduced: array-like {n_samples, n_features_to_select}
            Reduced feature matrix

        """
        self.fit(X, y)

        return self.transform(X)

    ######################### SUPPORTING FUNCTIONS ###########################
    def _getMultiClassMap(self):
        """ Relief algorithms handle the scoring updates a little differently for data with multiclass outcomes. In ReBATE we implement multiclass scoring in line with
        the strategy described by Kononenko 1994 within the RELIEF-F variant which was suggested to outperform the RELIEF-E multiclass variant. This strategy weights
        score updates derived from misses of different classes by the class frequency observed in the training data. 'The idea is that the algorithm should estimate the
        ability of attributes to separate each pair of classes regardless of which two classes are closest to each other'.  In this method we prepare for this normalization
        by creating a class dictionary, and storing respective class frequencies. This is needed for ReliefF multiclass score update normalizations. """
        mcmap = dict()

        for i in range(self._datalen):
            if (self._y[i] not in mcmap):
                mcmap[self._y[i]] = 0
            else:
                mcmap[self._y[i]] += 1

        for each in self._label_list:
            mcmap[each] = mcmap[each] / float(self._datalen)

        return mcmap

    def _get_attribute_info(self):
        """ Preprocess the training dataset to identify which features/attributes are discrete vs. continuous valued. Ignores missing values in this determination."""
        attr = dict()
        d = 0
        limit = self.discrete_threshold
        w = self._X.transpose()

        for idx in range(len(w)):
            h = self._headers[idx]
            z = w[idx]
            if self._missing_data_count > 0:
                z = z[np.logical_not(np.isnan(z))]  # Exclude any missing values from consideration
            zlen = len(np.unique(z))
            if zlen <= limit:
                attr[h] = ('discrete', 0, 0, 0, 0)
                d += 1
            else:
                mx = np.max(z)
                mn = np.min(z)
                sd = np.std(z)
                attr[h] = ('continuous', mx, mn, mx - mn, sd)
        # For each feature/attribute we store (type, max value, min value, max min difference, average, standard deviation) - the latter three values are set to zero if feature is discrete.
        return attr

    def _distarray_no_missing(self, xc, xd):
        """Distance array calculation for data with no missing values. The 'pdist() function outputs a condense distance array, and squareform() converts this vector-form
        distance vector to a square-form, redundant distance matrix.
        *This could be a target for saving memory in the future, by not needing to expand to the redundant square-form matrix. """
        from scipy.spatial.distance import pdist, squareform

        # ------------------------------------------#
        def pre_normalize(x):
            """Normalizes continuous features so they are in the same range (0 to 1)"""
            idx = 0
            x = np.array(x, dtype=float)
            # goes through all named features (doesn really need to) this method is only applied to continuous features
            for i in sorted(self.attr.keys()):
                if self.attr[i][0] == 'discrete':
                    continue
                cmin = self.attr[i][2]
                diff = self.attr[i][3]
                x[:, idx] -= cmin
                x[:, idx] /= float(diff)
                idx += 1
            return x

        # ------------------------------------------#

        if self.data_type == 'discrete':  # discrete features only
            return squareform(pdist(self._X, metric='hamming'))
        elif self.data_type == 'mixed':  # mix of discrete and continuous features
            d_dist = squareform(pdist(xd, metric='hamming'))
            # Cityblock is also known as Manhattan distance
            c_dist = squareform(pdist(pre_normalize(xc), metric='cityblock'))
            return np.add(d_dist, c_dist) / self._num_attributes

        else:  # continuous features only
            # xc = pre_normalize(xc)
            return squareform(pdist(pre_normalize(xc), metric='cityblock'))

    # ==================================================================#
    def _dtype_array(self):
        """Return mask for discrete(0)/continuous(1) attributes and their indices. Return array of max/min diffs of attributes."""
        attrtype = []
        attrdiff = []

        for key in self._headers:
            if self.attr[key][0] == 'continuous':
                attrtype.append(1)
            else:
                attrtype.append(0)
            attrdiff.append(self.attr[key][3])

        attrtype = np.array(attrtype)
        cidx = np.where(attrtype == 1)[0]
        didx = np.where(attrtype == 0)[0]

        attrdiff = np.array(attrdiff)

        return attrdiff, cidx, didx

    # ==================================================================#

    def _distarray_missing(self, xc, xd, cdiffs):
        """Distance array calculation for data with missing values"""
        cindices = []
        dindices = []
        # Get Boolean mask locating missing values for continuous and discrete features separately. These correspond to xc and xd respectively.
        for i in range(self._datalen):
            cindices.append(np.where(np.isnan(xc[i]))[0])
            dindices.append(np.where(np.isnan(xd[i]))[0])

        if self.n_jobs != 1:
            dist_array = Parallel(n_jobs=self.n_jobs)(delayed(get_row_missing)(
                xc, xd, cdiffs, index, cindices, dindices) for index in range(self._datalen))
        else:
            # For each instance calculate distance from all other instances (in non-redundant manner) (i.e. computes triangle, and puts zeros in for rest to form square).
            dist_array = [get_row_missing(xc, xd, cdiffs, index, cindices, dindices)
                          for index in range(self._datalen)]

        return np.array(dist_array)

    # ==================================================================#

    ############################# ReliefF ############################################

    # def find_neighbors(self):
    #
    #     NN = np.zeros([self._datalen, self.n_neighbors])
    #     for inst in range(self._datalen):
    #         _distance = self._distance_array[inst, :]
    #         _distance[inst] = sys.maxsize
    #         nn_list = _distance.argsort()
    #         NN[inst, :] = nn_list
    #     return NN

    def _find_neighbors(self,inst):
        """ Identify k nearest hits and k nearest misses for given instance. This is accomplished differently based on the type of endpoint (i.e. binary, multiclass, and continuous). """
        # Make a vector of distances between target instance (inst) and all others

        _distance = self._distance_array[inst, :]
        _distance[inst] = sys.maxsize

        # Identify neighbors-------------------------------------------------------
        """ NN for Binary Endpoints: """
        if self._class_type == 'binary':
            nn_list = []
            match_count = 0
            miss_count = 0
            for nn_index in np.argsort(_distance):
                if self._y[inst] == self._y[nn_index]:  # Hit neighbor identified
                    if match_count >= self.n_neighbors:
                        continue
                    nn_list.append(nn_index)
                    match_count += 1
                else:  # Miss neighbor identified
                    if miss_count >= self.n_neighbors:
                        continue
                    nn_list.append(nn_index)
                    miss_count += 1

                if match_count >= self.n_neighbors and miss_count >= self.n_neighbors:
                    break

        return np.array(nn_list)

    def _run_algorithm(self):
        # NN = self.find_neighbors()
        NN_ = self._find_neighbors(0)
        NN = np.zeros([self._datalen,len(NN_)])
        for instance_num in range(self._datalen):
            NN_ = self._find_neighbors(instance_num)
            NN[instance_num] = NN_
        n_neighbors_new = len(NN.T)

        attr_np = np.array(pd.DataFrame(self.attr).T)
        ftype = attr_np[:self._num_attributes, 0]
        mmdiff = attr_np[:self._num_attributes, 3]
        standDev = attr_np[:self._num_attributes, 4]

        y_NN = np.zeros(NN.shape)
        for i in range(n_neighbors_new):
            y_NN[:, i] = self._y[NN[:, i].astype('int')]

        _y_NN = np.tile(self._y, (n_neighbors_new, 1)).T
        count = np.equal(y_NN, _y_NN)
        count_hit = count.sum(axis=1)
        count_miss = n_neighbors_new - count_hit

        diff_sum = np.zeros([self._datalen, self._num_attributes])
        for i in range(n_neighbors_new):
            hit_miss = np.ones([self._datalen, self._num_attributes])
            X_NN = self._X[NN[:, i].astype('int'), :]
            rawfd = abs(X_NN - self._X)
            diff = abs(X_NN - self._X) / (mmdiff.reshape(1, self._num_attributes) + 0.000001)
            if self.data_type == 'mixed':
                diff[rawfd > standDev.reshape(1, self._num_attributes)] = 1
            diff[(X_NN == self._X) & (ftype == 'discrete')] = 0
            diff[(X_NN != self._X) & (ftype == 'discrete')] = 1
            hit_miss[np.equal(y_NN[:, i], self._y), :] = np.tile(-1 / count_hit[np.equal(y_NN[:, i], self._y)],
                                                            (self._num_attributes, 1)).T
            hit_miss[y_NN[:, i] != self._y, :] = np.tile(1 / count_miss[y_NN[:, i] != self._y], (self._num_attributes, 1)).T
            diff = diff * hit_miss
            diff_sum = diff_sum + diff
        diff_sum = diff_sum / self._datalen

        scores_feature = np.sum(diff_sum, axis=0)

        return scores_feature


# X = np.array([[1.61311827, 0.19955703,0],
#                 [-0.21997067, 0.86474714,1],
#                 [-0.58658846, -1.46341823,0],
#                 [-1.31982404, -0.79822813,0],
#                 [0.5132649, 1.19734219,1],
#               [-0.5132649, -0.19734219,1],
#               [1.5132649, 0.9734219,1]])
#
# y = np.array([1, 1, 1, 0, 0,0,0])
# fs = ReliefF(n_features_to_select=2,n_neighbors=4,discrete_threshold=3)
# fs.fit(X, y)
# fs.transform(X)
# features_ = np.arange(X.shape[1]).reshape((1, -1))
# features_ = fs.transform(features_)

