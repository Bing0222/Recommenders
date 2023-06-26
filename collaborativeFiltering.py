import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

# 加载数据
ratings_df = pd.read_csv("E:/download/ml-latest-small/ratings.csv")

# Surprise库期望数据的格式为：user, item, rating
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5)

trainset = data.build_full_trainset()
svd.fit(trainset)

recommendations = []
for movieId in trainset.all_items():
    if movieId not in trainset.ur[1]:
        predicted_rating = svd.predict(1, movieId)
        recommendations.append((movieId, predicted_rating.est))

recommendations.sort(key=lambda x:x[1], reverse=True)
print(recommendations[:10])
