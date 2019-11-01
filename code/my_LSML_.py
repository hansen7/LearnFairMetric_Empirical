import numpy as np, scipy.linalg as LA, pandas as pd
from numpy.linalg import LinAlgError
import warnings, random, itertools


class NonPSDError(LinAlgError):

    def __init__(self):
        err_msg = "Matrix is not positive semidefinite (PSD)."
        super(LinAlgError, self).__init__(err_msg)


class LSML:

    def __init__(self, tol=1e-5, max_iter=1000, sigma=0.01):
        self.tol = tol
        self.max_iter = max_iter
        self.metric = None
        self.sigma = sigma

    def _comparison_loss(self, metric, comparisons):
        loss = 0.
        for xa, xb, xc, xd in comparisons:
            vab = xa - xb
            vcd = xc - xd
            dab = np.dot(vab.T, np.dot(metric, vab))
            dcd = np.dot(vcd.T, np.dot(metric, vcd))
            if dab > dcd:
                loss += (np.sqrt(dab) - np.sqrt(dcd)) ** 2
        return loss

    def _regularization_loss(self, prior_inv):
        return self.sigma * (np.trace(np.dot(self.metric, prior_inv))
                             - np.log(LA.det(self.metric + np.diag(np.ones(len(prior_inv))*1e-12))))

    def _total_loss(self, prior_inv, new_metric, comparisons):
        return self._comparison_loss(new_metric, comparisons) + \
               self._regularization_loss(prior_inv)

    def _gradient(self, prior_inv, comparisons):
        dMetric = self.sigma * (prior_inv - LA.inv(self.metric))

        for xa, xb, xc, xd in comparisons:
            vab = xa - xb
            vcd = xc - xd
            dab = np.dot(vab.T, np.dot(self.metric, vab))
            dcd = np.dot(vcd.T, np.dot(self.metric, vcd))
            if dab <= dcd:
                continue  # comparison already satisfied.
            if dab == 0 or dcd == 0:
                continue  # this is the setting for COMPAS
            dMetric += (1 - np.sqrt(dcd / dab)) * np.outer(vab, vab) + \
                       (1 - np.sqrt(dab / dcd)) * np.outer(vcd, vcd)
        return dMetric

    def score(self, comparisons, h, dmetric=None):
        count = 0
        if dmetric is None:
            metric = self.metric
        else:
            metric = dmetric
        for xa, xb, xc, xd in comparisons:
            vab, vcd = xa - xb, xc - xd
            dab = np.dot(vab.T, np.dot(metric, vab))
            dcd = np.dot(vcd.T, np.dot(metric, vcd))
            if dab >= dcd + h:
                count += 1

        return count/len(comparisons)

    def _inv_cov_prior(self, X):
        """the inverse covariance matrix, which is the prior for the fitting process"""
        return LA.pinvh(np.cov(X, rowvar=False))

    def fit(self, X, comparisons, prior_mat=None):
        """Fitting with LSML"""
        # pass the inverse of covariance matrix for the input
        if prior_mat is not None:
            prior = self._inv_cov_prior(X)
        else:
            prior = prior_mat

        prior_inv = LA.inv(prior)
        self.metric = prior.copy()
        # print(self.metric)

        it, cur_s = 0, 100
        s_best = self._total_loss(prior_inv, self.metric, comparisons)
        # print('initial loss', s_best)
        while True:
            if it > self.max_iter:
                break
            grad = self._gradient(prior_inv, comparisons)
            grad_norm = LA.norm(grad)

            # print('gradient norm', grad_norm)

            # initial the step size as 0, which means take no actions with no gains
            l_best = 0.

            for step_size in np.logspace(-10, -1, 20):
                # actually here it is using grid search to select the best step size
                # print(step_size)
                step_size /= grad_norm
                new_metric = self.metric - step_size * grad
                # spectral decomposition
                w, v = LA.eigh(new_metric)
                # if np.any(w < 0):
                #    print 'for step_size', step_size, 'some eigs neg'
                new_metric = np.dot(v, np.dot(np.diag(np.maximum(w, 0)), v.T))
                # assert new_metric.ndim == 2
                cur_s = self._total_loss(prior_inv, new_metric, comparisons)
                # print(cur_s)
                if cur_s < s_best:
                    # print(cur_s, s_best)
                    l_best = step_size
                    s_best = cur_s
                    # self.metric = new_metric
            step_size = l_best
            metric = self.metric - step_size * grad
            w, v = LA.eigh(metric)
            metric = np.dot(v, np.dot(np.diag(np.maximum(w, 0)), v.T))

            # print(prev_s, cur_s, s_best)
            if LA.norm(metric - self.metric) < self.tol:
                break
            self.metric = metric
            it += 1
            # print('iter', it, 'best cost', s_best, 'best step size', l_best * grad_norm)
            # self.metric = M_best

        return None

    def idx2comparisions(self, X, index):
        """transform the index of positive negative pairs(a,b,c,d) into the format for (xa, xb, xc. xd),
        which is used for the downstream fitting method"""
        comparisons = []
        for a, b, c, d in index:
            comparisons.append([X[a], X[b], X[c], X[d]])

        return np.array(comparisons)

    # def positive_negative_pairs(self, num_constraints, same_length=False,
    #                             random_state=np.random.RandomState(777)):
    #     a, b = self._pairs(num_constraints, same_label=True,
    #                        random_state=random_state)
    #     c, d = self._pairs(num_constraints, same_label=False,
    #                        random_state=random_state)
    #     if same_length and len(a) != len(c):
    #         n = min(len(a), len(c))
    #         return a[:n], b[:n], c[:n], d[:n]
    #     return a, b, c, d

    # def _pairs(self, num_constraints, y, same_label=True, max_iter=10,
    #            random_state=np.random):
    #
    #     known_labels = np.unique(y)
    #     num_labels = len(known_labels)
    #     ab, it = set(), 0
    #
    #     while it < max_iter and len(ab) < num_constraints:
    #         nc = num_constraints - len(ab)
    #         for aidx in random_state.randint(num_labels, size=nc):
    #             if same_label:
    #                 mask = known_labels[aidx] == known_labels
    #                 mask[aidx] = False  # avoid identity pairs
    #             else:
    #                 mask = known_labels[aidx] != known_labels
    #             b_choices, = np.where(mask)
    #             if len(b_choices) > 0:
    #                 ab.add((aidx, random_state.choice(b_choices)))
    #         it += 1
    #     if len(ab) < num_constraints:
    #         warnings.warn("Only generated %d %s constraints (requested %d)" % (
    #             len(ab), 'positive' if same_label else 'negative', num_constraints))
    #     ab = np.array(list(ab)[:num_constraints], dtype=int)
    #     return known_label_idx[ab.T]

    def get_mahalanobis_matrix(self):
        """Gets the Mahalanobis Matrix, which should be a symmetric semidefinite matrix"""
        return self.metric

    def get_invcov_matrix(self, X):
        """Gets the inverse covariance matrix, which is the prior for the fitting process"""
        return LA.pinvh(np.cov(X, rowvar=False))

    def transform(self, X):
        return self._components_from_metric().dot(X)

    def _check_sdp_from_eigen(self, w, tol=None):
        """Checks if some of the eigenvalues given are negative, up to a tolerance
        level, with a default value of the tolerance depending on the eigenvalues.
        It also returns whether the matrix is positive definite, up to the above
        tolerance.
        """
        if tol is None:
            tol = np.abs(w).max() * len(w) * np.finfo(w.dtype).eps
        if tol < 0:
            raise ValueError("tol should be positive.")
        if any(w < - tol):
            raise NonPSDError()
        if any(abs(w) < tol):
            return False
        return True

    def _components_from_metric(self, tol=None):
        "Returns the transformation matrix from the Mahalanobis matrix"
        if not np.allclose(self.metric, self.metric.T):
            raise ValueError("The input metric should be symmetric.")
        # If M is diagonal, we will just return the elementwise square root:
        if np.array_equal(self.metric, np.diag(np.diag(self.metric))):
            self._check_sdp_from_eigen(np.diag(self.metric), tol)
            return np.diag(np.sqrt(np.maximum(0, np.diag(self.metric))))
            # there might be circumstances while the eigenvalue is negative but super close to zero
        else:
            try:
                # if `M` is positive semi-definite, it will admit a Cholesky
                # decomposition: L = cholesky(M).T
                return np.linalg.cholesky(self.metric).T
            except LinAlgError:
                # However, currently np.linalg.cholesky does not support indefinite
                # matrices. So if the latter does not work we will return L = V.T w^(
                # -1/2), with M = V*w*V.T being the eigenvector decomposition of M with
                # the eigenvalues in the diagonal matrix w and the columns of V being the
                # eigenvectors.
                w, V = np.linalg.eigh(self.metric)
                self._check_sdp_from_eigen(w, tol)
                return V.T * np.sqrt(np.maximum(0, w[:, None]))


def load_dataset_hcon(onehot=True):
    """
    :param onehot: bool, true if use one-hot encoding for the categorical
    :return:
        train and test set, test is the 200 with human voting, train dataset
        is the other 800 without human voting in the broward
    """
    # the broward dataset
    broward_ = pd.read_csv(r'../data/BROWARD_CLEAN.csv')

    # the voting decisions by human judge during the first run, now it is new data
    run1_ = pd.read_pickle(r'../data/HCON/HCON_long_lik.pkl')
    cols = ['race', 'sex', 'age', 'juv_fel_count', 'juv_misd_count',
            'priors_count', 'charge_degree (misd/fel)']
    X, y = broward_[cols], broward_['two_year_recid']
    X.index = broward_['id'].values
    y.index = X.index

    num_for_run = run1_.index.values.astype(int)
    select_list = [idx_ for idx_ in range(len(broward_))
                   if broward_.iloc[idx_]['id'] in num_for_run]

    X_test, X_train,  = X.iloc[select_list], X.drop(X.index[select_list])
    y_test, y_train = y.iloc[select_list], y.drop(y.index[select_list])

    if onehot:
        X_train, X_test = one_hot_vec(X_train, X_test)

    return X_train, X_test, y_train, y_test, broward_


def one_hot_vec(X_train, X_test):
    """
    encode the non-binomial categorical variable (race) into one-hot vector
    :param X_train:
    :param X_test:
    :return:
    """
    warnings.filterwarnings("ignore", category=FutureWarning)
    X_train.is_copy = False
    X_test.is_copy = False
    X_train['race_w'] = (X_train['race'] == 1).astype(int)
    X_train['race_b'] = (X_train['race'] == 2).astype(int)
    X_train['race_o'] = ((X_train['race'] != 1)
                         & (X_train['race'] != 2)).astype(int)

    X_test['race_w'] = (X_test['race'] == 1).astype(int)
    X_test['race_b'] = (X_test['race'] == 2).astype(int)
    X_test['race_o'] = ((X_test['race'] != 1)
                        & (X_test['race'] != 2)).astype(int)

    X_train.drop(['race'], axis=1, inplace=True)
    X_test.drop(['race'], axis=1, inplace=True)
    return X_train, X_test


def quadruplets_gen(X_df, lik_df, subsample=1e-5):
    """

    """
    # a set of unique values
    set_value = np.unique(lik_df)
    set_value.sort()

    constraints_ls = []
    for idx in range(len(set_value) - 1):
        for jdx in range(idx, len(set_value) - 1):
            for kdx in range(jdx + 1, len(set_value)):
                constraints_ls.append([idx, jdx, idx, kdx])

    constraints_index_ls = []
    for a, b, a, c in constraints_ls:
        # print(set_value[a], set_value[b], set_value[c])
        a_set = lik_df.loc[lik_df == set_value[a]].index.values
        b_set = lik_df.loc[lik_df == set_value[b]].index.values
        c_set = lik_df.loc[lik_df == set_value[c]].index.values

        constraints_index_ls += list(itertools.product(a_set, b_set, a_set, c_set))

    # pdb.set_trace()
    random.shuffle(constraints_index_ls)
    constraints_index_ls = constraints_index_ls[:int(len(constraints_index_ls) * subsample)]
    # constraints_index_ls = random.sample(constraints_index_ls, int(len(constraints_index_ls) * subsample))
    #     return constraints_index_ls

    '''
    array-like, shape=(n_constraints, 4, n_features), should have:
    d(pairs[i, 0],X[i, 1]) < d(X[i, 2], X[i, 3]) for all 0 <= i < n_constraints.
    '''
    constraints_feature_ls = []
    # pdb.set_trace()
    for a_idx, b_idx, a_idx, c_idx in constraints_index_ls:
        constraints_feature_ls.append([X_df.loc[a_idx].values,
                                       X_df.loc[b_idx].values,
                                       X_df.loc[a_idx].values,
                                       X_df.loc[c_idx].values, ])

    return np.array(constraints_index_ls), np.array(constraints_feature_ls)


def quadruplets_gen_strict(X_df, lik_df, subsample=1e-5, th=2):
    """
    generate with more restriction
    """
    # a set of unique values
    set_value = np.unique(lik_df)
    set_value.sort()

    constraints_ls = []
    for idx in range(len(set_value) - 2):
        for jdx in range(idx + 1, len(set_value) - 1):
            for kdx in range(jdx + 1, len(set_value)):
                # print(set_value[idx], set_value[jdx], set_value[kdx])
                if set_value[kdx] - set_value[jdx] >= th:
                    constraints_ls.append([idx, jdx, idx, kdx])

    constraints_index_ls = []
    for a, b, a, c in constraints_ls:
        # print(set_value[a], set_value[b], set_value[c])
        a_set = lik_df.loc[lik_df == set_value[a]].index.values
        b_set = lik_df.loc[lik_df == set_value[b]].index.values
        c_set = lik_df.loc[lik_df == set_value[c]].index.values

        constraints_index_ls += list(itertools.product(a_set, b_set, a_set, c_set))

    # pdb.set_trace()
    random.shuffle(constraints_index_ls)
    constraints_index_ls = constraints_index_ls[:int(len(constraints_index_ls) * subsample)]
    # constraints_index_ls = random.sample(constraints_index_ls, int(len(constraints_index_ls) * subsample))
    #     return constraints_index_ls

    '''
    array-like, shape=(n_constraints, 4, n_features), should have:
    d(pairs[i, 0],X[i, 1]) < d(X[i, 2], X[i, 3]) for all 0 <= i < n_constraints.
    '''
    constraints_feature_ls = []
    # pdb.set_trace()
    for a_idx, b_idx, a_idx, c_idx in constraints_index_ls:
        constraints_feature_ls.append([X_df.loc[a_idx].values,
                                       X_df.loc[b_idx].values,
                                       X_df.loc[a_idx].values,
                                       X_df.loc[c_idx].values, ])

    return np.array(constraints_index_ls), np.array(constraints_feature_ls)


if __name__ == '__main__':

    demo_df = pd.read_pickle(r'../data/HCON/HCON_long_demo.pkl')
    dec_df = pd.read_pickle(r'../data/HCON/HCON_long_dec.pkl')
    conf_df = pd.read_pickle(r'../data/HCON/HCON_long_conf.pkl')
    lik_df = pd.read_pickle(r'../data/HCON/HCON_long_lik.pkl')

    # quadruplets = np.array([[[1.2, 7.5], [1.3, 1.5], [6.4, 2.6], [6.2, 9.7]],
    #                         [[1.3, 4.5], [3.2, 4.6], [6.2, 5.5], [5.4, 5.4]],
    #                         [[3.2, 7.5], [3.3, 1.5], [8.4, 2.6], [8.2, 9.7]],
    #                         [[3.3, 4.5], [5.2, 4.6], [8.2, 5.5], [7.4, 5.4]]])
    # X = np.array([[1.2, 7.5],
    #               [1.3, 1.5],
    #               [6.4, 2.6],
    #               [6.2, 9.7]])

    X_train, X_test, y_train, y_test, broward_ = load_dataset_hcon()
    constraints_index_ls, quadruplets = quadruplets_gen_strict(X_df=X_test, lik_df=lik_df[13], subsample=1e-5, th=2)

    lsml = LSML(sigma=0.001)
    lsml.fit(X=X_train, comparisons=quadruplets)

