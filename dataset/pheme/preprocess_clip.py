import os,json,re,jieba,pickle,gensim
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import clip
from emoji import demojize
import itertools
from itertools import islice, chain
from collections import Counter
import networkx as nx
import sys
from datetime import datetime
from datetime import timedelta
import dateutil.parser
from scipy.sparse import csr_matrix
from collections import defaultdict
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

w2v_dim = 300
use_stopwords = False
max_len = 50


dic = {
    'non-rumor': 0,
    'false': 1,
    'unverified': 2,
    'true': 3,
}

stopwords_path = os.getcwd() + '/pheme_files_vol5/stopwords_eng1.txt'
stopwords_eng1 = []
with open(stopwords_path, 'r') as f:
    for line in f.readlines():
        stopwords_eng1.append(line.strip())

stopwords_path = os.getcwd() + '/pheme_files_vol5/stopwords_eng2.txt'
stopwords_eng2 = []
with open(stopwords_path, 'r') as f:
    for line in f.readlines():
        stopwords_eng2.append(line.strip())

def clean_str_cut(string, task):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    if "weibo" not in task:
        # http_pattern = re.compile(r"http\S+")
        # string = re.sub(http_pattern, "", string)
        string = re.sub(r"&\w+;", " ", string)
        string = re.sub(r"[^\w\s#@]", " ", string)
        string = re.sub(r'@[\w_]+', '', string)
        string = demojize(string)
        string = re.sub(r"[^A-Za-z0-9(),!?#@\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)

    # http_pattern = re.compile(r"http\S+")
    # string = re.sub(http_pattern, "", string)
    string = re.sub(r"&\w+;", " ", string)
    string = re.sub(r"[^\w\s#@]", " ", string)
    string = demojize(string)

    string = re.sub(r"#", "", string)
    string = string.lower()
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    words = list(
        jieba.cut(string.strip().lower(), cut_all=False)) if "weibo" in task else string.strip().lower().split()
    if use_stopwords:
        words = [w for w in words if w not in stopwords_eng2]

    filtered_words = []
    for word in words:
        if word == "http":
            break
        filtered_words.append(word)
    return filtered_words

def get_vectors(text):
    text = clip.tokenize([text]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)
    text_features = text_features.cpu().numpy()
    return text_features

def lines_per_n(f, n):
    for line in f:
        # res = ''.join(chain([line], itertools.islice(f, n - 1)))
        # print(res)
        yield ''.join(chain([line], itertools.islice(f, n - 1)))

def getDateTimeFromISO8601String(s):
    d = dateutil.parser.parse(s)
    return d

def get_graphs():
    links = []
    ts = []
    with open(root_path + '/pheme_files_vol5/edge.json','r',encoding='utf-8') as input:
        data = json.load(input)
        for item in data:
            name_parts = item['name'].split('_')
            source_node = name_parts[0]
            target_node = name_parts[1]
            time = item['time']
            timestamp = getDateTimeFromISO8601String(time)
            ts.append(time)
            links.append((source_node, target_node, timestamp))

    ts = [datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S.%f%z') for time_str in ts]

    #print (min(ts), max(ts)) #最小时间:2014-08-09T00:00:00+00:00 最大时间:2015-03-31T00:00:00+00:00,时间跨度:234天
    #print ("# interactions", len(links)) #边的数量:7756

    links.sort(key =lambda x: x[2])

    # split edges into three time intervals
    SLICE_MONTHS = 2
    START_DATE = min(ts) + timedelta(23) 
    END_DATE = max(ts) - timedelta(31)
    #print("Spliting Time Interval: \n Early Time : {}, Mid Time, Last Time : {}".format(EARLY_DATE, MID_DATE,LAST_DATE))


    slice_links = defaultdict(lambda: nx.MultiGraph())
    for (a, b, time) in links:
        datetime_object = time
        if datetime_object > END_DATE:
            days_diff = (END_DATE - START_DATE).days//30
        else:
            days_diff = (datetime_object - START_DATE).days//30

        # print(datetime_object, days_diff)
        slice_id = days_diff // SLICE_MONTHS
        slice_id = max(slice_id, 0)
    
        if slice_id not in slice_links.keys():
            slice_links[slice_id] = nx.MultiGraph()
            if slice_id > 0:
                slice_links[slice_id] = slice_links[slice_id-1].copy()
                # slice_links[slice_id].add_nodes_from(slice_links[slice_id-1].nodes(data=True)) #将上一个slice_id中的节点添加到当前slcie_id中，并确保当前时间片中没有边
                #assert (len(slice_links[slice_id].edges()) ==0)
        
        slice_links[slice_id].add_edge(a,b, date=datetime_object)



 

    # print(slice_links[0].edges(data=True))
    # for i in slice_links:
    #     fig, ax = plt.subplots()
    #     nx.draw(slice_links[i], ax=ax)
    #     if i == 0:
    #         plt.savefig( './pheme_files_vol5/early.png')
    #     elif i == 1:
    #         plt.savefig( './pheme_files_vol5/mid.png')
    #     elif i == 2:
    #         plt.savefig( './pheme_files_vol5/last.png')
        #plt.show()

    # print statics of each graph
    used_nodes = []
    for id, slice in slice_links.items():
        print("In snapshoot {:<2}, #Nodes={:<5}, #Edges={:<5}".format(id, \
                            slice.number_of_nodes(), slice.number_of_edges()))
        for node in slice.nodes():
            if node not in used_nodes:
                used_nodes.append(node)
    # print(len(used_nodes))
    # sys.exit()


    # remap nodes in graphs. Cause start time is not zero, the node index is not consistent
    nodes_consistent_map = {node:idx for idx, node in enumerate(used_nodes)}

    #更新节点
    for id, slice in slice_links.items():
        slice_links[id] = nx.relabel_nodes(slice, nodes_consistent_map)


    return nodes_consistent_map,slice_links

def read_corpus(root_path, file_name):
    X_tids = []
    X_uids = []
    old_id_post_map = {}

    index_dict = {}
    with open(root_path + file_name +  "node_id.json", 'r', encoding='utf-8') as input:
        index_dict = json.load(input)


    with open(root_path + file_name + "pheme.train", 'r', encoding='utf-8') as input:
        X_train_tid, X_train, y_train = [], [], []
        for line in input.readlines():
            tid, content, label = line.strip().split("\t")
            X_tids.append(tid)
            X_train_tid.append(tid)
            fenci_res = clean_str_cut(content, file_name)
            fenci_res = ' '.join(fenci_res)
            X_train.append(fenci_res)
            y_train.append(dic[label])
            old_id_post_map[tid] = fenci_res
    

    with open(root_path + file_name + "pheme.dev", 'r', encoding='utf-8') as input:
        X_dev_tid, X_dev, y_dev = [], [], []
        for line in input.readlines():
            tid, content, label = line.strip().split("\t")
            X_tids.append(tid)
            X_dev_tid.append(tid)
            fenci_res = clean_str_cut(content, file_name)
            fenci_res = ' '.join(fenci_res)

            X_dev.append(fenci_res)
            y_dev.append(dic[label])
            old_id_post_map[tid] = fenci_res


    with open(root_path + file_name + "pheme.test", 'r', encoding='utf-8') as input:
        X_test_tid, X_test, y_test = [], [], []
        for line in input.readlines():
            tid, content, label = line.strip().split("\t")
            X_tids.append(tid)
            X_test_tid.append(tid)
            fenci_res = clean_str_cut(content, file_name)
            fenci_res = ' '.join(fenci_res)

            X_test.append(fenci_res)
            y_test.append(dic[label])
            old_id_post_map[tid] = fenci_res

    #生成三个时刻的图并且按照时间更新index_dict
    index_dict,slice_links = get_graphs()
    with open(root_path + '/pheme_files_vol5/node_id_new.json','w',encoding='utf-8') as output:
        json.dump(index_dict,output,ensure_ascii=False,indent=4)

    
    X_train_tid = [index_dict.get(id) for id in X_train_tid]
    X_dev_tid = [index_dict.get(id) for id in X_dev_tid]
    X_test_tid = [index_dict.get(id) for id in X_test_tid]

    with open(root_path + '/pheme_files_vol5/comment_content.json','r',encoding='utf-8') as input:
        comment_map = {}
        test_id_comment_map = json.load(input)
        for k,v in test_id_comment_map.items():
            k = index_dict.get(k)
            fenci_res = clean_str_cut(v,file_name)
            fenci_res = ' '.join(fenci_res)

            comment_map[k] = fenci_res

    with open(root_path + '/pheme_files_vol5/user_tweet.json','r',encoding='utf-8') as input:
        old_user_post_map = {}
        old_user_post_map = json.load(input)

    old_id_post_map = {index_dict.get(old_id): comment for old_id, comment in old_id_post_map.items()}

    # print(X_train)
    # print(X_dev)
    # print(X_test)

    node_embedding_matrix = np.zeros((len(index_dict), 512))


    for i, words in old_id_post_map.items():
        new_id = i
        embedding = get_vectors(words)
        node_embedding_matrix[new_id, :] = embedding

    for i, words in comment_map.items():
        new_id = i
        embedding = get_vectors(words)
        node_embedding_matrix[new_id, :] = embedding

    for u,posts in old_user_post_map.items():
        new_uid = index_dict[u]
        embedding = 0.0
        count = 0
        for post in posts:
            new_pid = index_dict[post]
            embedding += node_embedding_matrix[new_pid,:]
            count += 1
        if count > 0:
            embedding = embedding / count
            node_embedding_matrix[new_uid,:] = embedding


    print(node_embedding_matrix.shape)


    pickle.dump([node_embedding_matrix],
                open(root_path + "/pheme_files_vol5/node_embedding.pkl", 'wb'))
    

    graphs = []
    for id, slice in slice_links.items():
        tmp_feature = []
        for node in slice.nodes():
            tmp_feature.append(node_embedding_matrix[node])
        slice.graph["feature"] = csr_matrix(tmp_feature) #将 tmp_feature 转换为一个稀疏矩阵
        graphs.append(slice)


    # print(graphs)
    for graph in graphs:
        print(graph.number_of_nodes())
        print(graph.number_of_edges())


    # save
    with open(root_path + '/pheme_files_vol5/id_post_new.json','w',encoding='utf-8') as output:
        json.dump(old_id_post_map,output,ensure_ascii=False,indent=4)
    with open(root_path + '/pheme_files_vol5/id_comment_new.json','w',encoding='utf-8') as output:
        json.dump(comment_map,output,ensure_ascii=False,indent=4)
        
    save_path = "./pheme_files_vol5/graphs_new.pkl" #是否可以理解为替代上面的node_embedding_matrix？好像是
    with open(save_path, "wb") as f:
        pickle.dump(graphs, f)
    print("Processed Data Saved at {}".format(save_path))

    # print(node_embedding_matrix)
    # print(node_embedding_matrix.shape) #[8650,300]
    pickle.dump([node_embedding_matrix],
                open(root_path + "/pheme_files_vol5/node_embedding.pkl", 'wb'))


    

    return X_train_tid, X_train, y_train, \
        X_dev_tid, X_dev, y_dev, \
        X_test_tid, X_test, y_test



def feature_extract(root_path,filename,w2v_path):
    X_train_tid, X_train, y_train, \
    X_dev_tid, X_dev, y_dev, \
    X_test_tid, X_test, y_test = read_corpus(root_path, filename)

    # print(X_train_tid)
    # print(X_train)
    # print(y_train)

    #X_train,X_dev,X_test都是bert的句向量，维度为[num,768]
    #len_train:1412
    #len_dev:403
    #len_test:203

    X_train = [get_vectors(text) for text in X_train]
    X_dev = [get_vectors(text) for text in X_dev]
    X_test = [get_vectors(text) for text in X_test]
    
    
 
    pickle.dump([X_train_tid, X_train, y_train], open(root_path + "/pheme_files_vol5/train.pkl", 'wb'))
    pickle.dump([X_dev_tid, X_dev, y_dev], open(root_path + "/pheme_files_vol5/dev.pkl", 'wb'))
    pickle.dump([X_test_tid, X_test, y_test], open(root_path + "/pheme_files_vol5/test.pkl", 'wb'))

    print('Process Finished!')





    

if __name__ == "__main__":
    # with open(os.getcwd() + '/process.py',encoding='UTF-8') as f:
    #     exec(f.read())
    root_path = os.getcwd()
    filename = '/pheme_files_vol5/'
    feature_extract(root_path=root_path, filename=filename,w2v_path=os.getcwd() + "/pheme_files_vol5/twitter_w2v.bin")