#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk

import warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.exceptions import DataConversionWarning
from sklearn import preprocessing


# General Entropy Calculation, input a vector
def entropy(labels):
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    counts = np.bincount(labels)
    probs = counts[np.nonzero(counts)] / n_labels
    n_classes = len(probs)

    return -np.sum(probs * np.log2(probs) / np.log2(n_classes))


# calculate the entropy for the classification matrix
def ent_mat(matrix):
    ent_list = []
    for idx in range(len(matrix[1])):
        ent_list.append(1 - entropy(matrix[:, idx].astype(np.int64)))
    return ent_list


# the remainder of the binary entropy, candidate for the consistency measure
def ent_cons(matrix):
    ent_list = []
    for idx in range(len(matrix[1])):
        ent_list.append(1 - entropy(matrix[:, idx].astype(np.int64)))
    return 1. - np.array(ent_list)


def test_ent_cons():
    a = [1, 0, 0, 1]
    print(entropy(a))
    n_points = 100
    a = []
    for i in range(1, n_points):
        a.append(i * [0] + (n_points - i) * [1])
    a = np.array(a)

    plt.figure(figsize=(18, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.linspace(0, 1, 100), ent_mat(a))
    plt.title('Binary Entropy')

    plt.subplot(1, 2, 2)
    plt.plot(np.linspace(0, 1, 100), ent_cons(a))
    plt.title('Binary Entropy Remainder - Consistency')

    plt.savefig(r'../results/Binary_Entropy_Function_Graph.png')


def preprocessing_data(X_test, y_test):
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)
    X_test_rescale = pd.DataFrame(preprocessing.scale(X_test),
                                  columns=X_test.columns,
                                  index=X_test.index)
    X_test_rescale.index.name = 'Defendant_ID'
    X_test_rescale['Ground_Truth'] = y_test.values
    return X_test_rescale


def preprocessing_data_with_unit_var(X_test, select_col=(
        'age', 'juv_fel_count', 'juv_misd_count', 'priors_count')):
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)
    select_col = list(select_col)
    not_select_col = list(set(X_test.columns) - set(select_col))

    X_test_rescale = pd.DataFrame(preprocessing.scale(X_test[select_col]),
                                  columns=X_test[select_col].columns,
                                  index=X_test.index)
    X_test_rescale += X_test[select_col].mean(axis=0)
    X_test_rescale = pd.concat([X_test_rescale, X_test[not_select_col]], axis=1)
    X_test_rescale.index.name = 'Defendant_ID'
    return X_test_rescale


def load_dataset(onehot=True):
    """
    :param onehot: bool, true if use one-hot encoding for the categorical
    :return:
        train and test set, test is the 50 with human voting, train is the
        other 950 without human voting in the broward
    """
    # the broward dataset
    broward_ = pd.read_csv(r'../data/BROWARD_CLEAN.csv')

    # the voting decisions by human judge during the first run, now it is new data
    run1_ = pd.read_csv(r'../new_data/0_Decision.csv')
    cols = ['race', 'sex', 'age', 'juv_fel_count', 'juv_misd_count',
            'priors_count', 'charge_degree (misd/fel)']
    X, y = broward_[cols], broward_['two_year_recid']
    X.index = broward_['id'].values
    y.index = X.index

    num_for_run = [int(item.replace('Defendant_', '').replace('_Decision', ''))
                   for item in run1_['Unnamed: 0']]
    select_list = [idx_ for idx_ in range(len(broward_))
                   if broward_.iloc[idx_]['id'] in num_for_run]
    # reduced_broward = broward_.iloc[select_list]
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.3, random_state=0)
    X_test, X_train, = X.iloc[select_list], X.drop(X.index[select_list])
    y_test, y_train = y.iloc[select_list], y.drop(y.index[select_list])

    if onehot:
        X_train, X_test = one_hot_vec(X_train, X_test)

    return X_train, X_test, y_train, y_test, broward_


def one_hot_vec(X_train, X_test):
    """ encode the non-binomial categorical variable (race) into one-hot vector
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


def load_para(n_rounds=10, n_defendants=50, n_respondents=20):
    """
    :param n_rounds: number of rounds for voting
    :param n_defendants: number of defendants
    :param n_respondents: number of respondents
    :return: the above three
    """
    _, X_test, _, y_test, _ = load_dataset()

    assert (X_test.shape[0] == n_defendants)
    return n_rounds, n_defendants, n_respondents


'''---------Build Up Weighted Prediction-------------'''


def weighted_prediction_res(respondent):
    """
    confidence-weighted prediction of specified respondent of all rounds

    :param respondent: respondent_id, from 0 to n_respondent-1
    :return: weighted prediction DataFrame of that respondent,
             n_rounds * n_defendants  -> 10 * 50 in this case
    """
    n_rounds, n_defendants, n_respondents = load_para()
    pred_ = []
    conf_ = []
    # looping around n_rounds voting
    for i in range(n_rounds):  # run decision from 1 to 10
        pred_df = pd.read_csv(r'../new_data/%d_Decision.csv' % i)
        conf_df = pd.read_csv(r'../new_data/%d_Confidence.csv' % i)
        pred_.append(pred_df.iloc[:, respondent + 1].values)
        conf_.append(conf_df.iloc[:, respondent + 1].values)

    pred_df = pd.DataFrame(np.array(pred_, dtype='float64'),
                           index=range(1, n_rounds + 1),
                           columns=range(1, n_defendants + 1))
    # change negative voting from 0 to -1
    pred_df[pred_df == 0] = -1
    conf_df = pd.DataFrame(np.array(conf_, dtype='float64'),
                           index=range(1, n_rounds + 1),
                           columns=range(1, n_defendants + 1))
    # generate weighted prediction DataFrame, ranging from -7 ~ 7 exclude 0
    weight_pred_df = pred_df.multiply(other=conf_df)
    return weight_pred_df.astype(int)


def weighted_prediction_round(round_idx):
    """
    confidence-weighted prediction of all respondents at round_idx

    :param round_idx: integer, ranging from 1 to 10
    :return: DataFrame of confidence weighted prediction at round_idx,
              n_defendants * n_respondents -> 50 * 35 in this case
    """
    # _, X_test, _, y_test, _ = load_dataset(onehot=True)
    pred_df = pd.read_csv(r'../new_data/%d_Decision.csv' % (round_idx - 1))
    conf_df = pd.read_csv(r'../new_data/%d_Confidence.csv' % (round_idx - 1))
    pred_df = pred_df.iloc[:, 1:]
    pred_df[pred_df == 0] = -1
    weighted_pred_df = pd.DataFrame(pred_df.values * conf_df.iloc[:, 1:].values,
                                    index=range(50))

    return weighted_pred_df


def weighted_prediction_quantile(weighted_df, div=2):
    """

    :param weighted_df: the confidence-weighted prediction, either for a single respondent
                         at all the time or for all the respondents at the single round
    :param num_div: the number of portions of positive/negative which will be divided into
    :return: a relabeled weighted prediction DataFrame
    """
    pos_df, neg_df = weighted_df * (weighted_df > 0), weighted_df * (weighted_df < 0)
    pos_df[pos_df == 0] = None
    neg_df[neg_df == 0] = None

    # basically is to deal with NaN values
    backbone_df = pd.DataFrame(np.zeros_like(weighted_df), index=weighted_df.index,
                               columns=weighted_df.columns)

    pos_cache = (pos_df.apply(lambda x: pd.qcut(
        x.dropna(), min(div, len(x.dropna())), labels=False, duplicates='drop'),
                              axis=0).applymap(lambda x: x + 1) + backbone_df).fillna(0)

    neg_cache = (neg_df.apply(lambda x: pd.qcut(
        x.dropna(), min(div, len(x.dropna())), labels=False, duplicates='drop'),
                              axis=0).applymap(lambda x: x + div + 1) + backbone_df).fillna(0)

    # container_df += pos_df.apply(lambda x: pd.qcut(x.dropna(), min(div, len(x.dropna())),
    #                                                labels=False, duplicates='drop'), axis=0).fillna(0)
    # container_df += neg_df.apply(lambda x: pd.qcut(
    #     x.dropna(), min(div, len(x.dropna())), labels=False, duplicates='drop'),
    #                              axis=0).applymap(lambda x: x+div).fillna(0)

    container_df = (pos_cache + neg_cache).astype(int)

    # print(container_df.head())
    # print(not container_df.isna().any().any())

    '''assert there is no NaN value in the returned DataFrame'''
    assert not container_df.isna().any().any()
    return container_df


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, broward_ = load_dataset()
    # print(X_test.head())
    # load_para()

    df_ = weighted_prediction_round(round_idx=1)
    rank_df = weighted_prediction_quantile(df_)
