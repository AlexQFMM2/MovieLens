import pickle
import torch
import pandas as pd
import numpy as np

# Save parameters to file  
def save_params(params):  
    pickle.dump(params, open('preprocess.p', 'wb'))  

# Load parameters from file  
def load_params():  
    return pickle.load(open('preprocess.p', mode='rb'))  


# Load parameters  
title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = load_params()  

# 参数设置  
embed_dim = 32  # 嵌入矩阵的维度  
uid_max = max(features.take(0, 1)) + 1  # 用户ID个数  
gender_max = max(features.take(1, 1)) + 1  # 性别个数  
age_max = max(features.take(2, 1)) + 1  # 年龄类别个数  
job_max = max(features.take(3, 1)) + 1  # 职业个数  
movie_id_max = max(features.take(4, 1)) + 1  # 电影ID个数  
movie_categories_max = max(genres2int.values()) + 1  # 电影类型个数  
movie_title_max = len(title_set)  # 电影名单词个数  

# 组合方法  
combiner = "sum"  
sentences_size = title_count  # = 15  
window_sizes = [2, 3, 4, 5]  # 使用列表替代集合  
filter_num = 8  
movieid2idx = {val[0]: i for i, val in enumerate(movies.values)}  

# 超参  
num_epochs = 5  
batch_size = 256  
dropout_keep = 0.5  
learning_rate = 0.0001  
show_every_n_batches = 20  
save_dir = './save'  

"""  
- title_count: Title字段的长度(15)  
- title_set: Title文本的集合  
- genres2int: 电影类型转数字的字典  
- features: 是输入X  
- targets_values: 是学习目标y  
- ratings: 评分数据集的Pandas对象  
- users: 用户数据集的Pandas对象  
- movies: 电影数据的Pandas对象  
- data: 三个数据集组合在一起的Pandas对象  
- movies_orig: 没有做数据处理的原始电影数据  
- users_orig: 没有做数据处理的原始用户数据  
"""  

def get_tensor(data):  
    df = pd.DataFrame(data)  

    # 处理 DataFrame，将列表填充到相同的长度  
    max_length = max(df[5].apply(len).max(), df[6].apply(len).max())  

    # 创建填充后的列表  
    title_tensor = df[5].apply(lambda x: x + [0] * (max_length - len(x)))  
    genres_tensor = df[6].apply(lambda x: x + [0] * (max_length - len(x)))  

    # 这里不需要用浮点数，直接使用long  
    title_tensor = torch.tensor(title_tensor.tolist(), dtype=torch.long)  
    genres_tensor = torch.tensor(genres_tensor.tolist(), dtype=torch.long)  

    # 提取用户特征  
    user_ids = df.iloc[:, 0].values.astype(np.int64)  # 转换为整型  
    user_genders = df.iloc[:, 1].values.astype(np.int64)  # 转换为整型  
    user_ages = df.iloc[:, 2].values.astype(np.int64)  # 转换为整型  
    user_jobs = df.iloc[:, 3].values.astype(np.int64)  # 转换为整型  

    # 转换标量特征为张量，并确保都为 Long 类型  
    uid_tensor = torch.tensor(user_ids, dtype=torch.long)  
    gender_tensor = torch.tensor(user_genders, dtype=torch.long)  
    age_tensor = torch.tensor(user_ages, dtype=torch.long)  
    job_tensor = torch.tensor(user_jobs, dtype=torch.long)

    movies_features = df[4].values  
    movies_tensor = torch.tensor(movies_features.astype(np.float32), dtype=torch.long)  

    return uid_tensor,gender_tensor,age_tensor, job_tensor,movies_tensor, title_tensor, genres_tensor  

def get_users(data):
    df = pd.DataFrame(data)  
    # 提取用户特征  
    user_ids = df.iloc[:, 0].values.astype(np.int64)  # 转换为整型  
    user_genders = df.iloc[:, 1].values.astype(np.int64)  # 转换为整型  
    user_ages = df.iloc[:, 2].values.astype(np.int64)  # 转换为整型  
    user_jobs = df.iloc[:, 3].values.astype(np.int64)  # 转换为整型  

    # 转换标量特征为张量，并确保都为 Long 类型  
    uid_tensor = torch.tensor(user_ids, dtype=torch.long)  
    gender_tensor = torch.tensor(user_genders, dtype=torch.long)  
    age_tensor = torch.tensor(user_ages, dtype=torch.long)  
    job_tensor = torch.tensor(user_jobs, dtype=torch.long)

    return uid_tensor, gender_tensor, age_tensor, job_tensor

def get_movies(data):  
    df = pd.DataFrame(data)

    # 处理 DataFrame，将列表填充到相同的长度  
    max_length = max(df['Title'].apply(len).max(), df['Genres'].apply(len).max())  

    # 创建填充后的列表  
    title_tensor = df['Title'].apply(lambda x: x + [0] * (max_length - len(x)))  
    genres_tensor = df['Genres'].apply(lambda x: x + [0] * (max_length - len(x)))  

    # 这里不需要用浮点数，直接使用long  
    title_tensor = torch.tensor(title_tensor.tolist(), dtype=torch.long)  
    genres_tensor = torch.tensor(genres_tensor.tolist(), dtype=torch.long) 

    movies_features = df['MovieID'].values  
    movies_tensor = torch.tensor(movies_features.astype(np.float32), dtype=torch.long)

    return movies_tensor, title_tensor, genres_tensor

