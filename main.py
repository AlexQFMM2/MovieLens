import pickle
import keras
import MyModel as Main_Model
import load as ld
import numpy as np
import torch
import pandas as pd
import MovieFeatureLayer
import random

def init_model():
    X = ld.features
    Y = ld.targets_values
    my_net.train_model(X,Y)

my_net = Main_Model.MyModel()
#init_model()
my_net.load_checkpoint()
# 载入电影特征矩阵  
movie_matrics = np.load('movie_matrics.npy')  
#print("Movie Matrics:", movie_matrics)  

# 载入用户特征矩阵  
users_matrics = np.load('user_matrics.npy')  
#print("User Matrics:", user_matrics)

movies = ld.movies
movieid2idx = ld.movieid2idx
sentences_size = ld.sentences_size
users = ld.users
movies_orig = ld.movies_orig
users_orig = ld.users_orig

# 指定用户和电影进行评分
#这部分就是对网络做正向传播，计算得到预测的评分
def rating_movie(mv_net, user_id_val, movie_id_val):  
    user_data = users.iloc[[user_id_val - 1]]
    movie_data = movies.iloc[[movie_id_val -1]]
    uid_tensor, gender_tensor, age_tensor, job_tensor = ld.get_users(user_data)

    movies_tensor, title_tensor , genres_tensor = ld.get_movies(movie_data)

    # 进行推理  
    inference_val = mv_net(uid_tensor, gender_tensor, age_tensor, job_tensor, movies_tensor, genres_tensor, title_tensor)  
    
    return inference_val.detach().numpy()  # 返回 NumPy 数组  

"""
res = rating_movie(my_net, 234, 1401)
print(res)
"""

def recommend_same_type_movie(movie_id_val, movieid2idx, movies_orig, movie_matrics, top_k=20):  
    # 确保输入是 PyTorch 张量  
    if isinstance(movie_matrics, np.ndarray):  
        movie_matrics = torch.tensor(movie_matrics, dtype=torch.float32)  

    # 归一化电影矩阵  
    norm_movie_matrics = torch.sqrt(torch.sum(movie_matrics**2, dim=1, keepdim=True))  # 计算L2范数  
    normalized_movie_matrics = movie_matrics / norm_movie_matrics  # 归一化  

    # 推荐同类型的电影  
    probs_embeddings = normalized_movie_matrics[movieid2idx[movie_id_val]].reshape(1, -1)  # 确保形状正确  
    probs_similarity = torch.matmul(probs_embeddings, normalized_movie_matrics.t())  # 计算相似度  
    sim = probs_similarity.squeeze().numpy()  # 将结果转换为 NumPy 数组  

    print("您看的电影是：{}".format(movies_orig[movieid2idx[movie_id_val]]))  
    print("猜你还喜欢看：")  
    
    # 使用概率分布推荐电影  
    p = np.squeeze(sim)  
    p[np.argsort(p)[:-top_k]] = 0  # 仅保留前 top_k 个电影的相似度  

    # 确保归一化前不会出现全零情况  
    if np.sum(p) == 0:  
        print("没有找到可推荐的电影或任意相似度为零。")  
        return []  

    p = p / np.sum(p)  # 进行归一化  

    results = set()  
    while len(results) < 5:  # 确保结果长度为5  
        # 使用 len(p) 而不是硬编码的3883  
        c = np.random.choice(len(p), 1, p=p)[0]  
        results.add(c)  

    for val in results:  
        print(movies_orig[val])  

    return results  

def recommend_your_favorite_movie(user_id_val, movieid2idx, movies_orig, users_matrics, movie_matrics, top_k=10):  
    # 确保输入是 PyTorch 张量  
    if isinstance(movie_matrics, np.ndarray):  
        movie_matrics = torch.tensor(movie_matrics, dtype=torch.float32)  

    if isinstance(users_matrics, np.ndarray):  
        users_matrics = torch.tensor(users_matrics, dtype=torch.float32)  

    # 归一化电影矩阵  
    norm_movie_matrics = torch.sqrt(torch.sum(movie_matrics**2, dim=1, keepdim=True))  # 计算L2范数  
    normalized_movie_matrics = movie_matrics / norm_movie_matrics  # 归一化  

    # 推荐您喜欢的电影  
    probs_embeddings = users_matrics[user_id_val - 1].reshape(1, -1)  # 确保形状正确  
    probs_similarity = torch.matmul(probs_embeddings, normalized_movie_matrics.t())  # 计算相似度  
    sim = probs_similarity.squeeze().numpy()  # 将结果转换为 NumPy 数组  
    
    print("猜您喜欢：")  
    p = np.squeeze(sim)  
    
    # 确保只保留前 top_k 的相似度  
    p[np.argsort(p)[:-top_k]] = 0  
    if np.sum(p) == 0:  
        print("没有可推荐的电影。")  
        return []  
    
    p = p / np.sum(p)  # 进行归一化  

    results = set()  
    while len(results) < 5:  # 确保结果长度为 5  
        c = np.random.choice(len(p), 1, p=p)[0]  # 使用 len(movies_orig) 动态获取电影数量  
        results.add(c)  

    for val in results:  
        print(movies_orig[val])  

    return results  

def recommend_other_favorite_movie(movie_id_val, movieid2idx, movies_orig, users_orig, users_matrics, movie_matrics, top_k=20):  
    # 确保输入是 PyTorch 张量  
    if isinstance(movie_matrics, np.ndarray):  
        movie_matrics = torch.tensor(movie_matrics, dtype=torch.float32)  

    if isinstance(users_matrics, np.ndarray):  
        users_matrics = torch.tensor(users_matrics, dtype=torch.float32)  

    # 选择电影特征  
    probs_movie_embeddings = users_matrics[movieid2idx[movie_id_val]].reshape(1, -1)  

    # 计算用户喜欢此电影的相似度  
    probs_user_favorite_similarity = torch.matmul(probs_movie_embeddings, users_matrics.t())  
    favorite_user_id = np.argsort(probs_user_favorite_similarity.numpy())[-top_k:]  

    # 输出观看的电影信息  
    print("您看的电影是：{}".format(movies_orig[movieid2idx[movie_id_val]]))  
    #print("喜欢看这个电影的人是：{}".format(users_orig[favorite_user_id - 1]))  

    # 获取喜欢此电影的用户的特征  
    probs_users_embeddings = users_matrics[favorite_user_id - 1].reshape(-1, 200)  
    
    # 计算这些用户与电影的相似度  
    probs_similarity = torch.matmul(probs_users_embeddings, movie_matrics.t())  
    sim = probs_similarity.numpy()  

    # 找到用户喜欢的电影  
    p = np.argmax(sim, axis=1)  
    print("喜欢看这个电影的人还喜欢看：")  

    if len(set(p)) < 5:  
        results = set(p)  
    else:  
        results = set()  
        while len(results) < 5:  
            c = p[random.randint(0, top_k - 1)]  # 从前 top_k 个用户中随机挑选  
            results.add(c)  

    for val in results:  
        print(movies_orig[val])  

    return results  

def test():
    print("\n")
    recommend_your_favorite_movie(user_id_val=16, movieid2idx=movieid2idx, movies_orig=movies_orig, users_matrics=users_matrics, movie_matrics=movie_matrics)
    print("\n")
    recommend_same_type_movie(movie_id_val=27, movieid2idx=movieid2idx, movies_orig=movies_orig, movie_matrics=movie_matrics)    
    print("\n")
    recommend_other_favorite_movie(movie_id_val=27, movieid2idx=movieid2idx, movies_orig=movies_orig, users_orig=users_orig, users_matrics=users_matrics, movie_matrics=movie_matrics)  

test()