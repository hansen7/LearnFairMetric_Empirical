import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np, pandas as pd, networkx as nx, plotly.io as pio
from utils import preprocessing_data_with_unit_var, load_dataset, \
    weighted_prediction_quantile, weighted_prediction_res,\
    weighted_prediction_round

# load the credential file of my own on the Plot.ly
plotly.tools.set_credentials_file(username='hansen96',
                                  api_key='WCwigwR73x86bN0v6goP')


def similarity_fig_from_weighted_prediction(round_idx, respondent_idx,
                                            similarity_edge_list=None, weighted_pred=None):
    """
    The similarity_edge_list has the priority for construction than the weighted prediction.

    :param round_idx: integer, ranging from 1 to 10
    :param respondent_idx: id of respondents, ranging from the 1 to 35
    :param similarity_edge_list: edges between similar points, a list of tuples(x_i, x_j),
    indicating that x_i and x_j are similar
    :param weighted_pred: confident of confidence weighted prediction,
    either at a specific round_idx or for a specific respondent at all times
    :return: a similarity connection graph of the 50 defendants, and the similarity_edge_list
    """
    _, X_test, _, y_test, _ = load_dataset(onehot=True)

    # (x,y) position of each nodes, here I set them into the circle for appreciation of beauty
    pos = dict()
    for i in range(50):
        pos[X_test.index.values[i]] = (np.cos(2*np.pi*i/50), np.sin(2*np.pi*i/50))
    # add all the nodes to the network
    G = nx.Graph()
    G.add_nodes_from(pos.keys())
    for n, p in pos.items():
        G.node[n]['pos'] = p

    # similarity_edge_list has the privilege to be used for the construction of the network
    if similarity_edge_list is None:
        if weighted_pred is None:
            # use the 14 classes weighted prediction if both are None
            y_test = weighted_prediction_round(
                round_idx=round_idx)[respondent_idx-1].values.reshape(-1, 1)
        else:
            y_test = weighted_pred[respondent_idx-1].values.reshape(-1, 1)
        # to integrate the labels into a pairwise fashion
        mask = (y_test[None] == y_test[:, None])[:, :, 0]
        a, b = np.nonzero(np.triu(mask, k=1))
        similarity_edge_list = [(X_test.index.values[a_],
                                 X_test.index.values[b_]) for a_, b_ in zip(a, b)]

    G.add_edges_from(similarity_edge_list)

    '''Make the Graph via Plot.ly'''
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in G.edges():
        x0, y0 = G.node[edge[0]]['pos']
        x1, y1 = G.node[edge[1]]['pos']
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # color-scale options
            # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'),
            line=dict(width=2)))

    for node in G.nodes():
        x, y = G.node[node]['pos']
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])

    for node, adjacencies in enumerate(G.adjacency()):
        node_trace['marker']['color'] += tuple([len(adjacencies[1])])
        node_info = '# of connections: ' + str(len(adjacencies[1])) + '<br>Defendant_ID: '\
                    + str(X_test.index.values[node])
        node_trace['text'] += tuple([node_info])

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        width=700,
                        height=600,
                        title='<br>Similarity Graph of COMPAS Defendants (Respondent-%2d_Week-%2d)'
                              % (respondent_idx, round_idx),
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=50, r=50, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

    return fig, list(G.edges())


def similarity_fig_from_metric_by_thres(round_idx, respondent_idx,
                                            d_metric_dict, d_thres):
    """
    Basically to build the similarity edges between node_i and node_j
    if there d^2(node_i, node_j) within the transformed feature space is
    less than the threshold value, d_thres.

    :param round_idx: integer, ranging from 1 to 10
    :param respondent_idx: id of respondents, ranging from the 1 to 35
    :param d_metric_dict: dict to save learned distance metric, indexed by the
    respondent_id and the round_idx
    :param d_thres: threshold value to determine whether the similarity
    edges should be built
    :return: a similarity connection graph and the similarity edge list
    """
    _, X_test, _, y_test, _ = load_dataset(onehot=True)
    X_test = preprocessing_data_with_unit_var(X_test)
    d_metric = d_metric_dict['R%d_W%d' % (respondent_idx, round_idx)]

    similarity_edges = []

    for idx in X_test.index:
        diff = X_test.values - X_test.loc[idx].values
        X_test['distance'] = np.dot(diff @ d_metric, diff.T).diagonal()
        neighbor_list = [my_idx for my_idx in X_test.loc[
            X_test['distance'] <= d_thres].index.values if my_idx != idx]
        for item in neighbor_list:
            similarity_edges.append((idx, item))
        X_test.drop(['distance'], axis=1, inplace=True)
    fig, edges = similarity_fig_from_weighted_prediction(
        round_idx=round_idx, respondent_idx=respondent_idx,
        similarity_edge_list=similarity_edges)

    return fig, edges


def similarity_fig_from_metric_by_nca(round_idx, respondent_idx, d_metric_dict):
    """
    nca -> neighborhood component analysis

    First is to define the neighborhood of data point x_i in the transformed space,
    it is the minimal hypersphere which centers at x_i, and its radius is equal to
    the farthest point which is similar with x_i. This function is to build similarity
    edges between x_i and all the data points within its neighborhood.

    :param round_idx: integer, ranging from 1 to 10
    :param respondent_idx: id of respondents, ranging from the 1 to 35
    :param d_metric_dict: dict to save learned distance metric, indexed by the
    respondent_id and the round_idx
    :return: a similarity connection graph and the similarity edge list
    """
    _, X_test, _, y_test, _ = load_dataset(onehot=True)
    X_test = preprocessing_data_with_unit_var(X_test)
    d_metric = d_metric_dict['R%d_W%d' % (respondent_idx, round_idx)]

    weighted_y_test = pd.DataFrame(weighted_prediction_quantile(
        weighted_prediction_round(round_idx))[respondent_idx - 1].values, index=y_test.index)

    similarity_edges = []

    for idx in X_test.index:
        diff = X_test.values - X_test.loc[idx].values
        X_test['distance'] = np.dot(diff @ d_metric, diff.T).diagonal()
        max_d = X_test.loc[(weighted_y_test ==
                            weighted_y_test.loc[idx]).values.ravel()]['distance'].max()

        #print('idx: %5d' % idx, ' max_d: %2.2f.' % max_d)
        neighbor_list = [my_idx for my_idx in X_test.loc[
            X_test['distance'] <= max_d].index.values if my_idx != idx]
        for item in neighbor_list:
            similarity_edges.append((idx, item))
        X_test.drop(['distance'], axis=1, inplace=True)
    fig, edges = similarity_fig_from_weighted_prediction(
        round_idx=round_idx, respondent_idx=respondent_idx,
        similarity_edge_list=similarity_edges)

    return fig, edges


def similarity_fig_from_metric_by_nca_part(round_idx, respondent_idx,
                                           n_neighbors, d_metric_dict):
    """
    similar as nca, only here the neighborhood is defined as the minimal hypersphere
    that contains the n_neighbors similar points of x_i

    :param round_idx: integer, ranging from 1 to 10
    :param respondent_idx: id of respondents, ranging from the 1 to 35
    :param n_neighbors: the number of similar points to define the neighborhood
    :param d_metric_dict: dict to save learned distance metric, indexed by the
    respondent_id and the round_idx
    :return: a similarity connection graph and the similarity edge list
    """
    _, X_test, _, y_test, _ = load_dataset(onehot=True)
    X_test = preprocessing_data_with_unit_var(X_test)
    d_metric = d_metric_dict['R%d_W%d' % (respondent_idx, round_idx)]
    # weighted_y_test = pd.DataFrame(weighted_prediction(round_idx)[respondent_idx - 1].values,
    #                                index=y_test.index)

    weighted_y_test = pd.DataFrame(weighted_prediction_quantile(
        weighted_prediction_round(round_idx))[respondent_idx - 1].values, index=y_test.index)

    similarity_edges = []

    for idx in X_test.index:
        diff = X_test.values - X_test.loc[idx].values
        X_test['distance'] = np.dot(diff @ d_metric, diff.T).diagonal()

        same_class = X_test.loc[(weighted_y_test == weighted_y_test.loc[idx]).values.ravel()]
        max_d = same_class['distance'].sort_values(ascending=True).values[
            min(len(same_class)-1, n_neighbors)]

        #print('idx: %5d'%idx, ' max_d: %2.2f.'%max_d)
        neighbor_list = [my_idx for my_idx in X_test.loc[
            X_test['distance'] <= max_d].index.values if my_idx != idx]
        for item in neighbor_list:
            similarity_edges.append((idx, item))
        X_test.drop(['distance'], axis=1, inplace=True)

    fig, edges = similarity_fig_from_weighted_prediction(
        round_idx=round_idx, respondent_idx=respondent_idx,
        similarity_edge_list=similarity_edges)

    return fig, edges


def similarity_fig_from_metric_by_knn(round_idx, respondent_idx,
                                      n_neighbors, d_metric_dict):
    """
    knn, the neighborhood is defined as the minimal hypersphere
    that contains the n_neighbors points of x_i

    :param round_idx: integer, ranging from 1 to 10
    :param respondent_idx: id of respondents, ranging from the 1 to 35
    :param n_neighbors: the number of similar points to define the neighborhood
    :param d_metric_dict: dict to save learned distance metric, indexed by the
    respondent_id and the round_idx
    :return: a similarity connection graph and the similarity edge list
    """
    _, X_test, _, y_test, _ = load_dataset(onehot=True)
    X_test = preprocessing_data_with_unit_var(X_test)
    d_metric = d_metric_dict['R%d_W%d' % (respondent_idx, round_idx)]

    weighted_y_test = pd.DataFrame(weighted_prediction_quantile(
        weighted_prediction_round(round_idx))[respondent_idx - 1].values, index=y_test.index)

    similarity_edges = []

    for idx in X_test.index:
        diff = X_test.values - X_test.loc[idx].values
        X_test['distance'] = np.dot(diff @ d_metric, diff.T).diagonal()

        max_d = X_test['distance'].sort_values(ascending=True).values[n_neighbors]

        neighbor_list = [my_idx for my_idx in X_test.loc[
            X_test['distance'] <= max_d].index.values if my_idx != idx]
        for item in neighbor_list:
            similarity_edges.append((idx, item))
        X_test.drop(['distance'], axis=1, inplace=True)

    fig, edges = similarity_fig_from_weighted_prediction(
        round_idx=round_idx, respondent_idx=respondent_idx,
        similarity_edge_list=similarity_edges)

    return fig, edges


def similarity_edges_from_metric_by_knn(round_idx, respondent_idx,
                                        n_neighbors, d_metric_dict):
    _, X_test, _, y_test, _ = load_dataset(onehot=True)
    X_test = preprocessing_data_with_unit_var(X_test)
    d_metric = d_metric_dict['R%d_W%d' % (respondent_idx, round_idx)]

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


def cal_distance_with_df(d_metric_dict):
    _, X_test, _, y_test, _ = load_dataset(onehot=True)
    X_test = preprocessing_data_with_unit_var(X_test)

    distance_dict = dict()
    for round_idx in range(1, 11):
        for respondent_idx in range(1, 36):
            distance_df = pd.DataFrame(np.zeros((len(X_test), len(X_test))),
                                       index=X_test.index, columns=X_test.index)
            d_metric = d_metric_dict['R%d_W%d' % (respondent_idx, round_idx)]

            for idx in X_test.index:
                diff = X_test.values - X_test.loc[idx].values
                distance_df[idx] = np.dot(diff @ d_metric, diff.T).diagonal()

            distance_dict["R%d_W%d" % (respondent_idx, round_idx)] = distance_df
            print(respondent_idx, round_idx)

    np.save(r'../new_data/lmnn_distance.npy', distance_dict)
    return None


def connectivity_sanity_check(similarity_edges, round_idx, respondent_idx, div=2):
    """
    summarize the connectivity from their amounts and the accuracy

    :param similarity_edges: edges between similar points, a list of tuples(x_i, x_j),
    indicating that x_i and x_j are similar
    :return: a dataframe to summarize the connectivity with the provided edges
    """
    _, _, _, y_test, _ = load_dataset(onehot=False)
    sum_df = pd.DataFrame(np.zeros_like(y_test.values),
                          index=y_test.index, columns=['Amount'])
    sum_df['Accuracy'] = sum_df['Amount']

    weighted_y_test = pd.DataFrame(weighted_prediction_quantile(
        weighted_prediction_round(round_idx), div=div)[respondent_idx - 1].values, index=y_test.index)

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
                    set(weighted_y_test.loc[weighted_y_test[0] ==
                                            weighted_y_test.loc[idx, 0]].index)))\
                /len(my_dict[idx])
        except ZeroDivisionError:
            # if defendant idx doesn't have any other in similarity
            sum_df.loc[idx, "Accuracy"] = 0
    return sum_df


def confidence_calibration_check(round_idx, respondent_idx, d_metric_dict, n_neighbors):
    _, X_test, _, y_test, _ = load_dataset(onehot=True)
    X_test = preprocessing_data_with_unit_var(X_test)
    d_metric = d_metric_dict['R%d_W%d' % (respondent_idx, round_idx)]

    weighted_y_test = pd.DataFrame(weighted_prediction_round(round_idx)
                                   [respondent_idx - 1].values,
                                   index=y_test.index,
                                   columns=['weighted_y_test'])
    weighted_y_test['y_test'] = y_test
    weighted_y_test['neighbor reci prop'] = y_test

    for idx in X_test.index:
        diff = X_test.values - X_test.loc[idx].values
        X_test['distance'] = np.dot(diff @ d_metric, diff.T).diagonal()

        max_d = X_test['distance'].sort_values(ascending=True).values[n_neighbors]

        neighbor_list = [my_idx for my_idx in X_test.loc[
            X_test['distance'] <= max_d].index.values if my_idx != idx]

        reci = 0
        for neighbor in neighbor_list:
            reci += weighted_y_test.loc[neighbor, 'y_test']
        weighted_y_test.loc[neighbor, 'neighbor reci prop'] = reci/n_neighbors

        X_test.drop(['distance'], axis=1, inplace=True)

    return weighted_y_test


if __name__ == '__main__':

    round_idx, respondent_idx, d_thres = 1, 1, 1.0
    d_metric_dict = np.load(r'../new_data/lmnn_k3_metric.npy').item()

    cal_distance_with_df(d_metric_dict)

    # fig, similarity_edges= similarity_fig_from_metric_by_nca_part(
    #     round_idx=round_idx, respondent_idx=respondent_idx,
    #     n_neighbors=3, d_metric_dict=d_metric_dict)
    #
    # connectivity_sum = connectivity_sanity_check(similarity_edges, round_idx, respondent_idx)
    # print(connectivity_sum)

    # fig = similarity_fig_from_mmc_metric_by_nca(
    #     round_idx=round_idx, respondent_idx=respondent_idx, d_metric_dict=d_metric_dict)

    # py.iplot(fig, config={'showLink': False},
    #          filename='LMNN_Sanity_Check/COMPAS_Similarity_NCAP_R-%d_W-%d' \
    #                   % (respondent_idx, round_idx))
    # fig = similarity_fig_from_weighted_prediction(round_idx=round_idx, respondent_idx=respondent_idx)
    # py.iplot(fig, filename='COMPAS_Similarity_R-%d_W-%d)' % (respondent_idx, round_idx))
    #
    # pio.write_image(fig, r'../results/similarity_R-%d_W-%d.png' % (respondent_idx, round_idx))
