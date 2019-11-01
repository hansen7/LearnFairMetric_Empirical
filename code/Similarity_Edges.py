from metric_learn import MMC_Supervised, LMNN, MMC
import numpy as np, pandas as pd, seaborn as sns
import matplotlib.pyplot as plt
import warnings, os, re, sys
from sklearn.neighbors import DistanceMetric
from utils import preprocessing_data_with_unit_var, \
                    weighted_prediction_round, weighted_prediction_quantile, one_hot_vec
# from Similarity_Graph import connectivity_sanity_check, confidence_calibration_check,\
#     similarity_edges_from_metric_by_knn
from sklearn.exceptions import DataConversionWarning

sns.set()
plt.rcParams.update({'figure.max_open_warning': 0})
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


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


def vanila_knn_sanity_check_hcon(X_test, y_test, n_neighbours=5):
    # n_neighbours = 5  # numebr of nearest neighbors for the calculation
    # X_train, X_test, y_train, y_test, _ = load_dataset(onehot=True)
    # X_train = preprocessing_data_with_unit_var(X_train)
    # X_test = preprocessing_data_with_unit_var(X_test)

    same_class = []
    dist = DistanceMetric.get_metric('euclidean')

    for idx in range(len(y_test)):
        # because the 8th column is the ground truth
        X_test['distance'] = dist.pairwise(X_test.values[:, :9],
                                           X_test.iloc[idx].values.reshape(1, -1)[:, :9])

        neighbor_idx = (X_test.sort_values(by=['distance'], axis=0, ascending=True)
        [1:n_neighbours + 1]).index

        same_class.append((y_test.loc[neighbor_idx] == y_test.iloc[idx]).mean())

    some_df = pd.DataFrame(same_class,
                           index=X_test.index,
                           columns=['kNN_Acc'])
    fig = plt.figure(figsize=(16, 3))

    bx = plt.subplot(1, 1, 1)
    bx.bar(range(1, 201), some_df['kNN_Acc'], width=0.4, color='red', tick_label=y_test.index)
    bx.set_xlabel('COMPAS Defendant ID')
    bx.set_ylabel('Accuracy')

    plt.setp(bx.get_xticklabels(), rotation=60, horizontalalignment='center')
    plt.suptitle('Sanity Check, kNN in the raw feature space')
    plt.tight_layout()
    fig.show()

    return some_df


def similarity_edges_from_metric_by_knn_hcon(X_test, respondent_idx, n_neighbors, d_metric_dict):
    # _, X_test, _, y_test, _ = load_dataset(onehot=True)
    # X_test = preprocessing_data_with_unit_var(X_test)
    d_metric = d_metric_dict['R%d' % respondent_idx]

    similarity_edges = {}

    for idx in X_test.index:
        diff = X_test.values - X_test.loc[idx].values
        X_test['distance'] = np.dot(diff @ d_metric, diff.T).diagonal()
        max_d = X_test['distance'].sort_values(ascending=True).values[n_neighbors]

        neighbor_list = [my_idx for my_idx in X_test.loc[
            X_test['distance'] <= max_d].index.values if my_idx != idx]
        similarity_edges[idx] = neighbor_list
        X_test.drop(['distance'], axis=1, inplace=True)

    return similarity_edges


def connectivity_sanity_check_hcon(similarity_edges, respondent_idx, y_test_all):
    """
    summarize the connectivity from their amounts and the accuracy

    :param similarity_edges: edges between similar points, a list of tuples(x_i, x_j),
    indicating that x_i and x_j are similar
    :return: a dataframe to summarize the connectivity with the provided edges
    """
    y_test = y_test_all[respondent_idx]
    sum_df = pd.DataFrame(np.zeros_like(y_test.values),
                          index=y_test.index, columns=['Amount'])
    sum_df['Accuracy'] = sum_df['Amount']
    weighted_y_test = pd.DataFrame(pd.read_pickle(r'../data/HCON/HCON_long_lik.pkl')[respondent_idx])

    if type(similarity_edges) == dict:
        my_dict = similarity_edges
    else:
        my_dict = dict()
        for idx in y_test.index:
            my_dict[idx] = []
        for x_i, x_j in similarity_edges:
            my_dict[x_i].append(x_j)
            my_dict[x_j].append(x_i)

    for idx in my_dict.keys():
        sum_df.loc[idx, "Amount"] = len(my_dict[idx])
        try:
            sum_df.loc[idx, "Accuracy"] = \
                len(set(my_dict[idx]).intersection(
                    set(weighted_y_test.loc[weighted_y_test[respondent_idx] ==
                                            weighted_y_test.loc[idx, respondent_idx]].index))) \
                / len(my_dict[idx])

        except ZeroDivisionError:
            sum_df.loc[idx, "Accuracy"] = 0

    return sum_df


def distance_metric_performance_hcon(X_test, y_test, n_neighbors, d_metric_dict):

    acc_ls = []
    for respondent_idx in range(1, 21):
        knn_edges = similarity_edges_from_metric_by_knn_hcon(
            X_test=X_test, respondent_idx=respondent_idx,
            n_neighbors=n_neighbors, d_metric_dict=d_metric_dict)
        knn_sum = connectivity_sanity_check_hcon(knn_edges, respondent_idx, y_test)
        acc_ls.append(knn_sum['Accuracy'].values.mean())

    fig = plt.figure(figsize=(16, 3))
    bx = plt.subplot(1, 1, 1)
    bx.bar(range(1, 21), acc_ls, width=0.4, color='red', tick_label=range(1, 21))
    bx.set_xlabel('Respondent-ID')
    bx.set_ylabel('Neighborhood Propensity')

    plt.suptitle('Portion of Neighborhood Sharing Same Likelihood Judgements')
    plt.tight_layout()
    fig.show()

    return acc_ls


def learn_mmc_metric(X_test):
    mmc_dict = dict()

    for respondent_id in range(1, 21):

        # y_test = pd.DataFrame(pd.read_pickle(r'../data/HCON/HCON_long_lik.pkl')[respondent_id]).values.reshape(-1, 1)
        y_test = pd.DataFrame(pd.read_pickle(r'../data/HCON/HCON_long_lik.pkl')[respondent_id]).values.reshape(-1, 1)

        mask = (y_test[None] == y_test[:, None])[:, :, 0]
        a, b = np.nonzero(np.triu(mask, k=1))   # similarity pairs
        c, d = np.nonzero(np.triu(~mask, k=1))  # dissimilarity pairs
        mmc = MMC(convergence_threshold=0.001)

        try:
            mmc.fit(X_test.values, (a, b, c, d))
            L = mmc.transform(np.diag(np.ones(9)))
            M = np.dot(L, L.T)
        except ValueError:
            # it should be converged anyway,
            # if the ValueError happens, there is some bad patterns of the input
            print('R%d has no non-trivial dissimilarity constraints given for MMC.' % respondent_id)
            M = 0.01 * np.diag(np.ones(9))

        mmc_dict['R%d' % respondent_id] = M * 100
        print('R:%2d' % respondent_id, ' First Row of MMC Mahalabobis Matrix:', (M[0] * 100).round(3))

    return mmc_dict

