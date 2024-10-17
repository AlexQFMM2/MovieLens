import os  
import time  
import torch  
import torch.nn as nn  
import torch.optim as optim  
import numpy as np  
from sklearn.model_selection import train_test_split  
import load as ld  
import MovieFeatureLayer as Model_Movie  
import UserFeatureLayer as Model_User  

def get_batches(Xs, ys, batch_size):  
    for start in range(0, len(Xs), batch_size):  
        end = min(start + batch_size, len(Xs))  
        yield Xs[start:end], ys[start:end]  

class MyModel(nn.Module):  
    def __init__(self, batch_size=256, learning_rate=0.001):  
        super(MyModel, self).__init__()  
        self.batch_size = batch_size  
        self.best_loss = float('inf')  
        self.losses = {'train': [], 'test': []}  

        # 实例化用户和电影特征层模型  
        self.U_model = Model_User.UserFeatureLayer()  
        self.M_model = Model_Movie.MovieFeatureLayer()  

        # 优化器  
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)  
        self.criterion = nn.MSELoss()  

        # 创建检查点目录  
        self.model_dir = 'model_dir'  
        self.checkpoint_dir = os.path.join(self.model_dir, 'checkpoints')  
        os.makedirs(self.checkpoint_dir, exist_ok=True)  

    def forward(self, uid_tensor, gender_tensor, age_tensor, job_tensor, movies_tensor, genres_tensor, title_tensor):  
        user_features = self.U_model(uid_tensor, gender_tensor, age_tensor, job_tensor)  
        movie_features = self.M_model(movies_tensor, genres_tensor, title_tensor)  
        inference = torch.sum(user_features * movie_features, dim=1)  
        return inference.unsqueeze(1)  

    def save_checkpoint(self, filename='checkpoint.pth'):  
        torch.save({  
            'model_state_dict': self.state_dict(),  
            'optimizer_state_dict': self.optimizer.state_dict(),  
            'best_loss': self.best_loss,  
        }, os.path.join(self.checkpoint_dir, filename))  

    def load_checkpoint(self, filename='checkpoint.pth'):  
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, filename))  
        self.load_state_dict(checkpoint['model_state_dict'])  
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  
        self.best_loss = checkpoint['best_loss']  
        print("恢复检查点成功")  

    def compute_loss(self, labels, logits):  
        return self.criterion(logits, labels)  

    def compute_metrics(self, labels, logits):  
        return torch.mean(torch.abs(logits - labels))  

    def extract_movie_features(self, data_batches):  
        movie_matrics = []  
        self.M_model.eval()  
        
        with torch.no_grad():  
            for batch in data_batches:  
                x, _ = batch  
                _, _, _, _, movies_tensor, title_tensor, genres_tensor = ld.get_tensor(x)  

                features = self.M_model(movies_tensor, genres_tensor, title_tensor)  
                movie_matrics.append(features.numpy())  
        
        # 处理空结果  
        if len(movie_matrics) == 0:  
            print("Warning: No movie features extracted.")  
            return np.array([])  # 或者使用其他默认值  

        return np.vstack(movie_matrics)  

    def extract_user_features(self, data_batches):  
        user_matrics = []  
        self.U_model.eval()  

        with torch.no_grad():  
            for batch in data_batches:  
                x, _ = batch  
                uid_tensor, gender_tensor, age_tensor, job_tensor, _, _, _ = ld.get_tensor(x)  
                features = self.U_model(uid_tensor, gender_tensor, age_tensor, job_tensor)  
                user_matrics.append(features.numpy())  
        
        # 处理空结果  
        if len(user_matrics) == 0:  
            print("Warning: No user features extracted.")  
            return np.array([])  # 或者使用其他默认值  
            
        return np.vstack(user_matrics)
    
    def train_step(self, x, y):  
        self.optimizer.zero_grad()  
        logits = self.forward(*x)  
        loss = self.compute_loss(y, logits)  
        loss.backward()  
        self.optimizer.step()  
        return loss, logits  

    def train_model(self, features, targets_values, epochs=5, log_freq=50):  
        
        for epoch_i in range(epochs):  
            train_X, test_X, train_y, test_y = train_test_split(features, targets_values, test_size=0.2, random_state=0)  
            train_batches = get_batches(train_X, train_y, self.batch_size)  
            batch_num = len(train_X) // self.batch_size  

            train_start = time.time()  
            avg_loss = 0.0  

            for batch_i in range(batch_num):  
                x, y = next(train_batches)  
                uid_tensor, gender_tensor, age_tensor, job_tensor, movies_tensor, title_tensor, genres_tensor = ld.get_tensor(x)  
                loss, logits = self.train_step((uid_tensor, gender_tensor, age_tensor, job_tensor, movies_tensor, genres_tensor, title_tensor),  
                                               torch.tensor(np.reshape(y, [self.batch_size, 1]), dtype=torch.float32))  

                avg_loss += loss.item()  

                step = self.optimizer.state[self.optimizer.param_groups[0]['params'][0]]['step']  
                if step % log_freq == 0:  
                    rate = log_freq / (time.time() - train_start)  
                    print('Epoch {:>3} Batch {:>4}/{}   Loss: {:.6f}'.format(epoch_i, batch_i, batch_num, loss.item(), rate))  
                    train_start = time.time()  

            self.losses['train'].append(avg_loss / batch_num)  
            self.testing((test_X, test_y), epoch_i)  
            
            train_batches = list(train_batches)  # 将生成器转换为列表  

            mov_batches = train_batches  
            user_batches = train_batches  

            # 提取并保存 movie_matrics  
            movie_matrics = self.extract_movie_features(mov_batches)  
            with open('movie_matrics.npy', 'wb') as f:  
                np.save(f, movie_matrics)  

            # 提取并保存 user_matrics  
            user_matrics = self.extract_user_features(user_batches)  
            with open('user_matrics.npy', 'wb') as f:  
                np.save(f, user_matrics)

    def testing(self, test_dataset, step_num):  
        print("start testing")  
        test_X, test_y = test_dataset  
        test_batches = get_batches(test_X, test_y, self.batch_size)  
        avg_loss = 0.0  

        for batch_i in range(len(test_X) // self.batch_size):  
            x, y = next(test_batches)  
            uid_tensor, gender_tensor, age_tensor, job_tensor, movies_tensor, title_tensor, genres_tensor = ld.get_tensor(x)  
            logits = self.forward(uid_tensor, gender_tensor, age_tensor, job_tensor, movies_tensor, genres_tensor, title_tensor)  
            test_loss = self.compute_loss(torch.tensor(np.reshape(y, [self.batch_size, 1]), dtype=torch.float32), logits)  
            avg_loss += test_loss.item()  

        avg_loss /= (len(test_X) // self.batch_size)  
        print('Model test set loss: {:.6f}'.format(avg_loss))  

        if avg_loss < self.best_loss:  
            self.best_loss = avg_loss  
            print("best loss = {}".format(self.best_loss))  
            self.save_checkpoint()