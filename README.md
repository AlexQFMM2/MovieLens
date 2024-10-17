# 基于Pytorch的电影推荐

## 先看结果

```
PS G:\desktop\learn Python\depth\movies> python .\main.py
恢复检查点成功


猜您喜欢：
[18 'Four Rooms (1995)' 'Thriller']
[117 "Young Poisoner's Handbook, The (1995)" 'Crime']
[56 'Kids of the Round Table (1995)' "Adventure|Children's|Fantasy"]       
[26 'Othello (1995)' 'Drama']
[159 'Clockers (1995)' 'Drama']


您看的电影是：[27 'Now and Then (1995)' 'Drama']
猜你还喜欢看：
[77 'Nico Icon (1995)' 'Documentary']
[49 'When Night Is Falling (1995)' 'Drama|Romance']
[18 'Four Rooms (1995)' 'Thriller']
[117 "Young Poisoner's Handbook, The (1995)" 'Crime']
[55 'Georgia (1995)' 'Drama']


您看的电影是：[27 'Now and Then (1995)' 'Drama']
喜欢看这个电影的人还喜欢看：
[49 'When Night Is Falling (1995)' 'Drama|Romance']
[99 'Heidi Fleiss: Hollywood Madam (1995)' 'Documentary']
[9 'Sudden Death (1995)' 'Action']
[107 'Muppet Treasure Island (1996)' "Adventure|Children's|Comedy|Musical"]
PS G:\desktop\learn Python\depth\movies> 
```

## 基本信息

```
数据集：movieLens ml-1m
语言: python
框架: pytorch

灵感来源: https://github.com/khanhnamle1994/movielens
```



## 模型构建

**MyModel**

```
输入: 用户特征(usersModel)，电影特征(movieModel)
输出: 预测评分
```

**userModel**

```
输入（嵌入层）: 用户id(1*32) ， 性别(1*16) ， 年龄(1*16)， 职业(1*16)
通过两个全连接层 转换为1*200的矩阵
```

**MovieModel**

```
输入(嵌入层) ： 电影id ， 电影类型
输入(文本卷积网络)： 电影名
```

具体实现直接查看源代码



## 辅助文件

**data_prepare.py**

处理原始数据，并保存至'preprocess.p'

### 原始数据

**users.dat**

```
1::F::1::10::48067
2::M::56::16::70072
3::M::25::15::55117
```

'UserID', 'Gender', 'Age', 'JobID', 'Zip-code'

把性别转换成01存储，最后一个用不上

**movies.dat**

```
1::Toy Story (1995)::Animation|Children's|Comedy
2::Jumanji (1995)::Adventure|Children's|Fantasy
3::Grumpier Old Men (1995)::Comedy|Romance
```

'MovieID', 'Title', 'Genres'

注意要把Title的年份去掉

然后把title和genres的长度设为一致

除了title和genres是list以外，其他的都是int



**load.py**

用来加载一些常用的变量和方法



**main.py**

通过模型处理好的数据实现功能

