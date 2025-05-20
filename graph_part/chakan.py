import pickle
import os

def count_zeros_in_y_train():
    # 指定训练集 pickle 文件的路径
    train_pickle_path = os.path.dirname(os.getcwd()) + '/dataset/pheme/pheme_files/train.pkl'

    # 加载训练集 pickle 文件
    _, _, y_train = pickle.load(open(train_pickle_path, 'rb'))

    # 计算标签为 0 的数量
    num_zeros = y_train.count(1)

    return num_zeros

# 调用函数以获取 y_train 中标签为 0 的数量
num_zeros = count_zeros_in_y_train()
print("Number of zeros in y_train:", num_zeros)
