import torch  
import torch.nn as nn  
import load as ld  
import pandas as pd  
import numpy as np  
import os

# 载入参数  
embed_dim_uid = 32  
embed_dim_other = 16  
uid_max = ld.uid_max  
age_max = ld.age_max  
gender_max = ld.gender_max  
job_max = ld.job_max  
X = ld.features  

class UserFeatureLayer(nn.Module):  
    def __init__(self):  
        super(UserFeatureLayer, self).__init__()  
        
        # 嵌入层  
        self.uid_embedding = nn.Embedding(uid_max, embed_dim_uid)  
        self.gender_embedding = nn.Embedding(gender_max, embed_dim_other)  
        self.age_embedding = nn.Embedding(age_max, embed_dim_other)  
        self.job_embedding = nn.Embedding(job_max, embed_dim_other)  

        # 计算拼接后的特征维度  
        total_feature_dim = embed_dim_uid + 3 * embed_dim_other  # 32 + 16 + 16 + 16 = 80  
        
        # 全连接层  
        self.fc1 = nn.Linear(total_feature_dim, 128)  
        self.fc2 = nn.Linear(128, 200)  
        self.relu = nn.ReLU()  

    def forward(self, uid, user_gender, user_age, user_job):  
        # 获取嵌入  
        uid_embed = self.uid_embedding(uid)  
        gender_embed = self.gender_embedding(user_gender)  
        age_embed = self.age_embedding(user_age)  
        job_embed = self.job_embedding(user_job)  

        # 拼接特征  
        user_feature_layer = torch.cat((uid_embed, gender_embed, age_embed, job_embed), dim=1)  

        # 通过全连接层获取最终用户特征  
        output = self.fc1(user_feature_layer)  
        output = self.relu(output)  
        output = self.fc2(output)  
        output = self.relu(output)  
        
        return output  

