import json, os, random, re,csv,sys
import numpy as np
import pandas as pd
from datetime import datetime

def create_input_files4pheme_new(mid_img):
    all_maps_lists_needs = get_map_and_list_pheme(mid_img) #all_maps_lists_needs = [tweet_content_map, tweet_label_map,user_tweet_map,edge]
    create_train_dev_test(all_maps_lists_needs, [70,10,20]) #划分数据集

def create_train_dev_test(all_maps_lists_needs : list,ratio):
    tweet_content_map, tweet_label_map, user_tweet_map,edge = all_maps_lists_needs
    # print(tweet_label_map)
    rumor = [(k, v) for k, v in tweet_content_map.items() if tweet_label_map[k] == 'false']
    nonrumor = [(k, v) for k, v in tweet_content_map.items() if tweet_label_map[k] == 'non-rumor']
    print('rumor_num:',len(rumor))
    print('nonrumor_num:',len(nonrumor))
    rumor_train, rumor_dev, rumor_test = split_dataset(ratio, rumor)
    nonrumor_train, nonrumor_dev, nonrumor_test = split_dataset(ratio, nonrumor)
    train = rumor_train + nonrumor_train
    dev = rumor_dev + nonrumor_dev
    test = rumor_test + nonrumor_test
    # print(train)
    # print(dev)
    # print(test)
    random.seed(666)
    random.shuffle(train)
    random.shuffle(dev)
    random.shuffle(test)
    write_train_dev_test_to_file(suffix='.train', dataset=train, tweet_label_map=tweet_label_map)
    write_train_dev_test_to_file(suffix='.dev', dataset=dev, tweet_label_map=tweet_label_map)
    write_train_dev_test_to_file(suffix='.test', dataset=test, tweet_label_map=tweet_label_map)

def split_dataset(ratio: list, dataset: list):
    random.seed(666)
    random.shuffle(dataset) #将数据随机排序
    n = len(dataset)
    train_num = int(n * ratio[0] / 100)
    dev_num = int(n * ratio[1] / 100)
    train = dataset[:train_num]
    dev = dataset[train_num:train_num + dev_num]
    test = dataset[train_num + dev_num:]
    return train, dev, test

def write_train_dev_test_to_file(suffix: str, dataset: list, tweet_label_map: dict):
    path = os.getcwd() + '/pheme_files_vol5/pheme' + suffix
    with open(path, 'w',encoding='utf-8') as f:
        for item in dataset:
            id_ = item[0]
            content = re.sub(pattern='[\t\n]', repl='', string=item[1])
            label = tweet_label_map[id_]
            line = str(id_) + '\t' + content + '\t' + label + '\n'
            f.write(line)


def get_map_and_list_pheme(mid_img):
    tweet_content_map = {}
    review_content_map = {}
    tweet_label_map = {}
    user_tweet_map = {}

    edge = []
    dir_path =  os.getcwd() + '/phemewithreactions/'
    process_rumor_norumor4pheme(dir_path, 'non_rumors', mid_img,tweet_label_map, tweet_content_map,review_content_map,user_tweet_map,edge)
    process_rumor_norumor4pheme(dir_path, 'rumors',mid_img, tweet_label_map, tweet_content_map, review_content_map,user_tweet_map,edge)

    nodes = []
    del_index = []
    for i in range(0,len(edge)):
        node1 = edge[i]['name'].split('_')[0]
        node2 = edge[i]['name'].split('_')[1]
        if len(node1) !=  len(node2):
            nodes.append(node1)
            nodes.append(node2)
        else:
            if node1 in nodes:
                nodes.append(node2)
            else:
                del_index.append(i)
                del review_content_map[node2]


    for index in reversed(del_index):
        if 0 <= index < len(edge):
            removed_item = edge.pop(index)
    nodes  = list(set(nodes))
    
    new_id_mapping = {}
    for i in range(0,len(nodes)):
        new_id_mapping[nodes[i]] = i

    with open(os.getcwd() + '/pheme_files_vol5/edge.json', 'w') as f:
        f.write(json.dumps(edge, indent=4))
    with open(os.getcwd() + '/pheme_files_vol5/comment_content.json', 'w') as f:
        f.write(json.dumps(review_content_map, indent=4))
    with open(os.getcwd() + '/pheme_files_vol5/user_tweet.json', 'w') as f:
        f.write(json.dumps(user_tweet_map, indent=4))

    with open(os.getcwd() + '/pheme_files_vol5/node_id.json', "w", encoding="utf-8") as output_file:
        json.dump(new_id_mapping, output_file, ensure_ascii=False, indent=4)



    return [tweet_content_map, tweet_label_map,user_tweet_map,edge]

def process_rumor_norumor4pheme(d, filename,mid_img, tweet_label_map, tweet_content_map,review_content_map,user_tweet_map,edge):
    path = os.path.join(d, filename)
    files = os.listdir(path)
    # print(len(edge))

    for file in files:

        if file not in mid_img:
            # print('break ',file)
            continue
        tweet_label_map[file] = 'non-rumor' if filename == 'non_rumors' else 'false'
        path2 = os.path.join(path, file, 'source-tweets')
        source = os.listdir(path2)[0]
        path2 = os.path.join(path2, source)
        with open(path2, 'r') as f:
            res = json.load(f)
            tweet_content_map[file] = res['text'] #推文内容
            user_id = res['user']['id'] #发布推文的用户id
            create_time = res['created_at']
            parsed_time = datetime.strptime(create_time, "%a %b %d %H:%M:%S %z %Y")
            create_time = parsed_time.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
            nodetonode = str(user_id)+'_'+str(file)
            # print(nodetonode)
            edge.append({'name':nodetonode,'time':create_time})

            # print('edge_num:',len(edge))

            if len(user_tweet_map) < 100:
                if user_id not in user_tweet_map:
                    user_tweet_map[user_id] = []
                user_tweet_map[user_id].append(file) #原始长度len:765


        path2 = os.path.join(path, file, 'reactions')
        if not os.path.exists(path2):
            continue
        reactions = os.listdir(path2)
        structure = os.path.join(path,file,'structure.json')
        tmp = []
        with open(structure,'r') as f:
            res = json.load(f)
            # print(res)
            reply_relations = get_edges(res)
            # print(len(reply_relations))
            time = []

            for i in range(0,len(reply_relations)):
                id = reply_relations[i].split("_")[1]
                reaction = os.path.join(path, file, 'reactions',str(id)+'.json')
                with open(reaction,'r') as fb:
                    info = json.load(fb)
                    text = info['text']
                    if len(text)<130:
                        reply_relations[i] = 0
                        continue
                    review_content_map[id] = info['text']
                    parsed_time = datetime.strptime(info['created_at'], "%a %b %d %H:%M:%S %z %Y")
                    create_at = parsed_time.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
                    time.append(create_at)

            # print(reply_relations)
            reply_relations = [x for x in reply_relations if x!=0]
            # print(len(reply_relations))
            # print(len(time))

            for i in range(0,len(time)):
                edge.append({'name':reply_relations[i],'time':time[i]})

    print('edge_num:', len(edge))
    
def get_edges(data, parent_key=None, edges=None):
    if edges is None:
        edges = []

    for key, value in data.items():
        if parent_key is not None:
            edge = f"{parent_key}_{key}"
            edges.append(edge)

        if isinstance(value, dict):
            get_edges(value, key, edges)

    return edges


        

def main():
    path = os.getcwd() + '/content2.csv'
    with open(path, 'r',encoding='UTF-8') as f:
        reader = csv.reader(f)
        result = list(reader)[1:]
        mid_img = [] #每个新闻的图片id
        #print(result)
        for line in result:
            mid_img.append(line[1])


    #print(mid_img)
    print(len(mid_img)) #2000
    

    #for i in range(0,len(mid_img)):

    create_input_files4pheme_new(mid_img)

if __name__ == '__main__':
    main()