# coding: utf-8
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
import random
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import csv
import nltk

nltk.download('punkt')  # for tokenization
nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()


def readCsv(path):
    df = pd.read_csv(path)
    return df


period1_df = readCsv('data/Period1.csv')
period2_df = readCsv('data/Period2.csv')
testdata_df = pd.read_csv('data/TestData.csv')
period1_df_len = period1_df.shape[0]
period2_df_len = period2_df.shape[0]
testdata_df_len = testdata_df.shape[0]

# node_info
with open("data/node_information.csv", "r") as f:
    reader = csv.reader(f)
    node_info = list(reader)

IDs = []
ID_pos = {}
for element in node_info:
    ID_pos[element[0]] = len(IDs)
    IDs.append(element[0])


def removeDuplicateEdge(df):
    df_len = df.shape[0]
    for i in range(df_len):
        source = int(df['source id'][i])
        target = int(df['target id'][i])
        if (source < target):
            temp = source
            source = target
            target = temp
            df.set_value(i, 'source id', source)
            df.set_value(i, 'target id', target)

    df.drop('year', axis=1, inplace=True)
    df = df.loc[df.duplicated() == False]
    df = df.reset_index(drop=True)


# remove dup edge
removeDuplicateEdge(period1_df)
removeDuplicateEdge(period2_df)
removeDuplicateEdge(testdata_df)


# retuen df column['node']
def getUniqueNode(list1, list2):
    temp = pd.concat([list1, list2], ignore_index=True)
    temp = temp.loc[temp.duplicated() == False]
    temp = temp.reset_index(drop=True)
    return temp


peroid1_node_uni = getUniqueNode(period1_df['source id'], period1_df['target id'])
peroid2_node_uni = getUniqueNode(period2_df['source id'], period2_df['target id'])
testdata_node_uni = getUniqueNode(testdata_df['source id'], testdata_df['target id'])
period_all_node_uni = getUniqueNode(peroid1_node_uni, peroid2_node_uni)
period_test_node_uni = getUniqueNode(period_all_node_uni, testdata_node_uni)


# retuen array((x,y))
def getUniqueEdge(data, input_type='dataframe'):
    # temp = [(df.loc[i, 'source id'],df.loc[i, 'target id']) for i in range(df.shape[0])]
    if input_type == 'dataframe':
        temp_set = set((data.loc[i, 'source id'], data.loc[i, 'target id']) for i in range(data.shape[0]))
    else:
        temp_set = set(data)

    return list(temp_set)


peroid1_edge = getUniqueEdge(period1_df)
peroid2_edge = getUniqueEdge(period2_df)
period_test_edge = [(testdata_df.loc[i, 'source id'], testdata_df.loc[i, 'target id']) for i in
                    range(testdata_df.shape[0])]
period_all_edge = list(set(peroid1_edge + peroid2_edge))

# data period info
print('period1 ', 'node:', len(peroid1_node_uni.values), 'edge:', len(peroid1_edge))
print('period2 ', 'node:', len(peroid2_node_uni.values), 'edge:', len(peroid2_edge))
print('period all ', 'node:', len(period_all_node_uni.values), 'edge:', len(period_all_edge))
print('period test ', 'node:', len(period_test_node_uni.values), 'edge:', len(period_all_edge))

period1_all_possible_edgs = []


# edge 大的在前面
def getAllPossibleEdge(nodes):
    temp_edges = []
    temp_nodes = []

    # range
    r = 97

    # !!!這裡每個300抽樣一次
    for i in range(0, len(nodes), r):
        for j in range(i + 1, len(nodes), r):
            if (nodes[i] == nodes[j]):
                continue
            else:
                # nodes
                if (nodes[i] not in temp_nodes):
                    temp_nodes.append(nodes[i])
                if (nodes[j] not in temp_nodes):
                    temp_nodes.append(nodes[j])
                # edges
                if (nodes[i] > nodes[j]):
                    temp_edges.append((nodes[i], nodes[j]))
                else:
                    temp_edges.append((nodes[j], nodes[i]))
        if i % (r * 10) == 0:
            print('round', i, ',has', len(temp_edges), 'edges', ',has', len(temp_nodes), 'nodes')

    return temp_nodes, temp_edges


# 取得period裡面nodes所有可能的edges
# period1_all_possible_nodes, period1_all_possible_edges = getAllPossibleEdge(peroid1_node_uni.values)
# period1_all_possible_edgs_sets = set(getUniqueEdge(period1_all_possible_edges, 'list'))
# print('get all possible edges in period1 done!!')

# 取得period1全部與已知的差集合
period1_df_set = set(period1_df)


# period1_all_possible_edgs_different_period2_edgs_set = period1_all_possible_edgs_sets.difference(period1_df_set)
# period1_all_possible_edgs_different_period2_edgs = list(period1_all_possible_edgs_different_period2_edgs_set)

# save to csv
def save2Csv(path, data):
    temp_df = pd.DataFrame(data=data)
    temp_df.to_csv(path, encoding='utf-8', index=False)
    print('get csv file!!')
    return temp_df


# temp_d = {'edges': period1_all_possible_edgs_different_period2_edgs}
# save2Csv('data/t1_only_point.csv', temp_d)

'''
創建圖
'''


def createGraph(nodes, edges):
    network = nx.Graph()

    network.add_nodes_from(nodes)
    network.add_edges_from(edges)

    print('Graph has nodes:', network.number_of_nodes(), ', edges:', network.number_of_edges())

    return network


G1 = createGraph(peroid1_node_uni.values, peroid1_edge)  # peroid1_edge
G2 = G1
G2.add_nodes_from(peroid2_node_uni.values)

G_all = createGraph(period_all_node_uni.values, period_all_edge)
G_test = G_all
G_test.add_nodes_from(testdata_node_uni.values)

'''
畫出圖
'''


# network graph
def printGraph(G, nodes):
    sub_graph = G.subgraph(nodes)  # 原圖太大取前幾個邊出來畫
    pos = nx.spring_layout(sub_graph)  # 圖的畫法
    nx.draw(sub_graph, pos=pos, node_size=40, vim=0.0, vmax=1.0, node_color="red")
    plt.show()


need_print = False
if need_print == True: printGraph(G1, list(peroid1_node_uni[0:100]))

'''
set training feature and label
'''


def generateTrainData(edges01, edge02):
    train_label = []
    train_edges = []

    print('Different edges has', len(edges01))
    # 注意順序
    for edge in edge02:
        train_label.append(1)
        train_edges.append(edge)

    for i, edge in enumerate(edges01):
        if edge not in edge02:
            train_label.append(0)
            train_edges.append(edge)

        if i % 10000 == 0:
            print('round', i)

    temp_d = {'edges': train_edges, 'label': train_label}
    temp_df = save2Csv('data/t1_only_point_with_label.csv', temp_d)
    return temp_df


def generateTrainData2(G, nodes01, nodes02, edges02, path):
    period1_not_in_2 = nodes01.append(nodes02).drop_duplicates(keep=False)
    period1_node_shuffle = random.Random(23).sample(list(period1_not_in_2), 650)
    sub_graph = G.subgraph(period1_node_shuffle)
    sub_graph_complement = nx.complement(sub_graph)
    # pos = nx.spring_layout(sub_graph)  # 圖的畫法
    # nx.draw(sub_graph_complement, pos=pos, node_size=40, vim=0.0, vmax=1.0, node_color="red")

    # tag label
    train_label = []
    # 注意順序
    for edge in edges02:
        train_label.append(1)
    for edge in list(sub_graph_complement.edges()):
        train_label.append(0)

    train_data_edge = edges02 + list(sub_graph_complement.edges())
    train_data = pd.DataFrame(data={'edges': train_data_edge, 'label': train_label})
    train_data = shuffle(train_data, random_state=32).reset_index(drop=True)
    train_data.head()
    train_data.to_csv(path, encoding='utf-8', index=False)

    return train_data


# !!!! 設定是否要重新train
need_generate_train_data = True
train_df = None
if need_generate_train_data == True:
    # train_df = generateTrainData(period1_all_possible_edgs_different_period2_edgs, peroid2_edge)
    print('generate train data')
    train_df = generateTrainData2(G2, peroid1_node_uni, peroid2_node_uni, peroid2_edge,
                                  'data/t1_only_point_with_label.csv')
else:
    train_df = readCsv('data/train_data.csv')

'''
    score function
'''


# common neighbor score (neighbor = 1 best)
def common_neighbor(network, input_node1, input_node2):
    source_neighbor = [n for n in network.neighbors(input_node1)]
    target_neighbor = [n for n in network.neighbors(input_node2)]
    intersection = list(set(source_neighbor) & set(target_neighbor))
    return len(intersection)


# Jaccard's cofficient
def jaccard_cofficient(network, input_node1, input_node2):
    cofficient = 0
    source_neighbor = [n for n in network.neighbors(input_node1)]
    target_neighbor = [n for n in network.neighbors(input_node2)]
    union = list(set(source_neighbor) | set(target_neighbor))
    intersection = list(set(source_neighbor) & set(target_neighbor))
    if len(union) == 0:
        return 0
    else:
        return (len(intersection) / len(union))


# Adamic/Adar
def adamic_adar(network, input_node1, input_node2):
    adamic_score = 0
    source_neighbor = [n for n in network.neighbors(input_node1)]
    target_neighbor = [n for n in network.neighbors(input_node2)]
    intersection = list(set(source_neighbor) & set(target_neighbor))

    if len(intersection) == 0:
        return 0
    else:
        for v in intersection:
            adamic_score += 1 / math.log(len([nv for nv in network.neighbors(v)]))
        return adamic_score


# clustering coefficient
def clustering_coefficient(network, input_node):
    node_degree = network.degree[input_node]
    node_triangle = nx.triangles(network, input_node)
    if node_degree - 1 <= 0:
        return 0
    else:
        return (2 * node_triangle) / (node_degree * (node_degree - 1))


        # perferential attachment


def perferential_attachment(network, input_node1, input_node2):
    source_neighbor = len([n for n in network.neighbors(input_node1)])
    target_neighbor = len([n for n in network.neighbors(input_node2)])
    return {'pa_mul': source_neighbor * target_neighbor, 'pa_add': source_neighbor + target_neighbor}


'''
add feature
'''


# train data feature


def calFeature(data, G, path):
    cn, jaccard, adamic, cc_mul, cc_add, pa_mul, pa_add = [], [], [], [], [], [], []
    overlap_title = []
    # temporal distance between the papers
    temp_diff = []
    # number of common authors
    comm_auth = []
    source_id, target_id = [], []

    for edge_id, edge in enumerate(data['edges'].values):
        source_id.append(edge[0])
        target_id.append(edge[1])

        # node info
        if (str(edge[0]) in ID_pos.keys() and str(edge[1]) in ID_pos.keys()):
            source_info = node_info[ID_pos[str(edge[0])]]
            target_info = node_info[ID_pos[str(edge[1])]]

            source_title = source_info[2].lower().split(" ")
            # remove stopwords
            source_title = [token for token in source_title if token not in stpwds]
            source_title = [stemmer.stem(token) for token in source_title]

            target_title = target_info[2].lower().split(" ")
            target_title = [token for token in target_title if token not in stpwds]
            target_title = [stemmer.stem(token) for token in target_title]

            source_auth = source_info[3].split(",")
            target_auth = target_info[3].split(",")

            overlap_title.append(len(set(source_title).intersection(set(target_title))))
            temp_diff.append(int(source_info[1]) - int(target_info[1]))
            comm_auth.append(len(set(source_auth).intersection(set(target_auth))))
        else:
            overlap_title.append(0)
            temp_diff.append(0)
            comm_auth.append(0)

        # neighbor base
        cn.append(common_neighbor(G, edge[0], edge[1]))
        jaccard.append(jaccard_cofficient(G, edge[0], edge[1]))

        # other
        adamic.append((adamic_adar(G, edge[0], edge[1])))
        source_cc = clustering_coefficient(G, edge[0])
        target_cc = clustering_coefficient(G, edge[1])
        cc_mul.append(source_cc * target_cc)
        cc_add.append(source_cc + target_cc)
        pa = perferential_attachment(G, edge[0], edge[1])
        pa_mul.append(pa['pa_mul'])
        pa_add.append(pa['pa_add'])

        if edge_id % 10000 == 0:
            print(edge_id, len(data))
    #
    data['source id'] = pd.Series(source_id, index=data.index)
    data['target id'] = pd.Series(target_id, index=data.index)
    data['cn'] = pd.Series(cn, index=data.index)
    data['jaccard'] = pd.Series(jaccard, index=data.index)
    data['adam'] = pd.Series(adamic, index=data.index)
    data['cc_mul'] = pd.Series(cc_mul, index=data.index)
    data['cc_add'] = pd.Series(cc_add, index=data.index)
    data['pa_mul'] = pd.Series(pa_mul, index=data.index)
    data['pa_add'] = pd.Series(pa_add, index=data.index)

    data['temp_diff'] = pd.Series(temp_diff, index=data.index)
    data['comm_auth'] = pd.Series(comm_auth, index=data.index)
    #data['overlap_title'] = pd.Series(overlap_title, index=data.index)

    print(data.head(10))

    data = data.drop('edges', axis=1)
    data.to_csv(path, index=False)


if need_generate_train_data == True:
    calFeature(train_df, G2, "data/train_data.csv")

'''
 cal test data
'''
if need_generate_train_data == True:
    test_temp = {'edges': period_test_edge}
    test_df = pd.DataFrame(data=test_temp)
    calFeature(test_df, G_test, "data/test_data.csv")
else:
    test_df = readCsv('data/test_data.csv')

'''
ML
'''
# train
train_feature = zip(train_df['cn'], train_df['jaccard'], train_df['adam'], train_df['cc_mul'], train_df['cc_add'],
                    train_df['pa_mul'], train_df['pa_add'], train_df['temp_diff'], train_df['comm_auth'])
train_feature = [[cn, jaccard, adam, cc_mul, cc_add, pa_mul, pa_add,temp_diff,comm_auth]
                 for cn, jaccard, adam, cc_mul, cc_add, pa_mul, pa_add,temp_diff,comm_auth in train_feature]

train_label = train_df['label'].tolist()

print('Run ML')

# svm
# svm = SVC(C=1.0, cache_size=4096)
# svm.fit(train_feature, train_label)

# tree
dt = tree.DecisionTreeClassifier()
dt = dt.fit(train_feature, train_label)

# rf
#rf = RandomForestClassifier(random_state=0, n_estimators=300)
#rf.fit(train_feature, train_label)

print('Run ML done')

# test
test_feature = zip(test_df['cn'], test_df['jaccard'], test_df['adam'], test_df['cc_mul'], test_df['cc_add'],
                   test_df['pa_mul'], test_df['pa_add'],test_df['temp_diff'], test_df['comm_auth'])
test_feature = [[cn, jaccard, adam, cc_mul, cc_add, pa_mul, pa_add,temp_diff,comm_auth]
                for cn, jaccard, adam, cc_mul, cc_add, pa_mul, pa_add,temp_diff,comm_auth in test_feature]

predict = dt.predict(test_feature)
print('Predict')

# out
row = [i for i in range(1, 10001)]
data = {'target id': row, 'label': predict}
predict = pd.DataFrame(data=data, columns=['target id', 'label'])
predict.to_csv("predict/predict.csv", index=False)
print('Get predict file')
