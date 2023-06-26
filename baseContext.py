import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

movies = pd.read_csv("tmdb_5000_movies.csv")

# 使用TF-IDF向量器处理电影概述
tfidf = TfidfVectorizer(stop_words='english')
movies['overview'] = movies['overview'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies['overview']) # 对概述进行TF-IDF处理

# 使用余弦相似度计算电影之间的相似性
cosine_sim = linear_kernel(tfidf_matrix,tfidf_matrix)

# 创建一个映射，映射电影标题到其在数据集中的索引
indices = pd.Series(movies.index,index=movies['title']).drop_duplicates()

def get_recommendations(title,cosine_sim=cosine_sim):
    idx = indices[title]

    # 获取所有电影的相似度分数
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 根据相似度分数排序
    sim_scores = sorted(sim_scores,key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]

    return movies['title'].iloc[movie_indices]

print(get_recommendations('The Dark Knight Rises'))

