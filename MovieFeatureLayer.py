import torch  
import torch.nn as nn  
import torch.nn.functional as F  
import load as ld  
import pandas as pd  
import numpy as np  
import os  

# 基础参数  
embed_dim = ld.embed_dim  
movie_id_max = ld.movie_id_max  # 电影ID个数  
movie_categories_max = ld.movie_categories_max  # 电影类型个数  
movie_title_max = ld.movie_title_max  # 电影名单词个数  

# 组合方法  
combiner = "sum"  
sentences_size = ld.title_count  # = 15  
window_sizes = [2, 3, 4, 5]  # 使用列表替代集合  
filter_num = 8  
movieid2idx = {val[0]: i for i, val in enumerate(ld.movies.values)}  

# 超参数  
num_epochs = 5  
batch_size = 256  
dropout_keep = 0.5  
learning_rate = 0.0001  
show_every_n_batches = 20  

class MovieFeatureLayer(nn.Module):  
    def __init__(self):  
        super(MovieFeatureLayer, self).__init__()  
        self.movie_id_embedding = nn.Embedding(movie_id_max, embed_dim)  
        self.category_embedding = nn.Embedding(movie_categories_max, embed_dim)  # 新增的类别嵌入层  

        self.fc1 = nn.Linear(embed_dim * 2, 64)  # 电影ID嵌入和类别嵌入拼接后是 2 * embed_dim   
        self.fc2 = nn.Linear(64 + filter_num * len(window_sizes), 200)  # + filter_num * len(window_sizes) 从 CNN 输出  
        self.relu = nn.ReLU()  # 正确的激活函数定义  

    # 合并电影类型的多个嵌入向量  
    def get_movie_categories_layers(self, movie_categories):  
        category_embeddings = self.category_embedding(movie_categories)  
        return torch.mean(category_embeddings, dim=1)  

    # Movie Title 的文本卷积网络实现  
    def get_movie_cnn_layer(self, movie_titles):  
        movie_title_embed_layer = nn.Embedding(movie_title_max, embed_dim)(movie_titles)  
        movie_title_embed_layer = movie_title_embed_layer.unsqueeze(1)  
        pool_layer_lst = []  
        for window_size in window_sizes:  
            conv_layer = F.relu(nn.Conv2d(in_channels=1, out_channels=filter_num, kernel_size=(window_size, embed_dim))(movie_title_embed_layer))  
            maxpool_layer = F.max_pool2d(conv_layer, kernel_size=(conv_layer.size(2), 1), stride=(1, 1))  
            pool_layer_lst.append(maxpool_layer)  
        pool_layer = torch.cat(pool_layer_lst, dim=1)    
        pool_layer_flat = pool_layer.view(pool_layer.size(0), -1)  
        dropout_layer = F.dropout(pool_layer_flat, p=1-dropout_keep, training=True)  
        return pool_layer_flat, dropout_layer  

    # movie 全连接后 返回一个 movie_features  
    def forward(self, movie_ids, movie_categories, movie_titles):  
        movie_id_embed = self.movie_id_embedding(movie_ids)   
        movie_category_embed = self.get_movie_categories_layers(movie_categories)  

        # 拼接电影ID嵌入和电影类别嵌入  
        tmp_features = torch.cat((movie_id_embed, movie_category_embed), dim=1)  
        
        output = self.fc1(tmp_features)  # 正确使用全连接层  
        output = self.relu(output)  # 使用激活函数  

        movie_title_flat, movie_title_dropout = self.get_movie_cnn_layer(movie_titles)   

        movie_features = torch.cat((output, movie_title_dropout), dim=1)  # 拼接  
        output = self.fc2(movie_features)  # 第二次全连接  
        output = self.relu(output)  # 使用激活函数  

        return output   
