#####################################################################################
#                             User BasedRecommendation
######################################################################################

import pandas as pd
import random
#######################################################################################
# Görev 1:  Veri Hazırlama
#######################################################################################

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

# Adım 1:   movie, ratingverisetleriniokutunuz.
movie = pd.read_csv('recommender_systems/datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('recommender_systems/datasets/movie_lens_dataset/rating.csv')


# random veri çekmek için kullanılıyor

#p = 0.05

# rating = pd.read_csv(
#         'recommender_systems/datasets/movie_lens_dataset/rating.csv',
#         header=0,
#         skiprows = lambda i: i > 0 and random.random() > p
#)

rating.columns
movie.columns
movie.head()
rating.head()

# Adım 2:  rating veri setine Id’lere ait film isimlerini vetürünü movie veri setinden ekleyiniz.

df = rating.merge(movie, how="left", on="movieId")
df.head()

# Adım 3:  Toplam oy kullanılma sayısı 1000'in altında olan filmlerin isimlerini
# listede tutunuz ve veri setinden çıkartınız.

comment_counts = pd.DataFrame(df["title"].value_counts())
rare_movies = comment_counts[comment_counts["title"] <= 1000].index.tolist()
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.shape

# Adım 4: index'te userID'lerin sutunlarda film isimlerinin ve değer olarak ratinglerin
# bulunduğu dataframe için pivot table oluşturunuz.

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

# NaN değerler var çünkü bazı kullanıcıların oy vermediği filmlerde listede

user_movie_df.head()


# Adım 5:  Yapılan tüm işlemleri fonksiyonlaştırınız.

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('recommender_systems/datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('recommender_systems/datasets/movie_lens_dataset/rating.csv')
    df = rating.merge(movie, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 100].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()

####################################################################################################
# Görev 2:  Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmes
####################################################################################################

# Adım 1: Rastgele bir kullanıcı id’si seçiniz.

user_movie_df.head()
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)

# Adım 2:  Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_df adında
# yeni bir dataframe oluşturunuz.

random_user_df = user_movie_df[user_movie_df.index == random_user]

# Adım 3:  Seçilen kullanıcıların oy kullandığı filmleri movies_watched adında bir listeye
# atayınız

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

# Kaç tane filmi oyladığını verir
len(movies_watched)

####################################################################################################
# Görev 3:  Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişilmesi
####################################################################################################

# Adım 1:  Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve
# movies_watched_df adında yeni bir dataframe oluşturunuz

user_movie_df.head()

movies_watched_df = user_movie_df[movies_watched]


# Adım 2:  Her bir kullancının seçili user'in izlediği filmlerin kaçını izlediğinin
# bilgisini taşıyan user_movie_count adında yeni bir dataframe oluşturunuz

movies_watched_df.head()

user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]

user_movie_count.head()

# Adım 3:  Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenlerin kullanıcı
# id’lerinden users_same_movies adında bir liste oluşturunuz.

perc = len(movies_watched) * 60 / 100
users_same_movies  = user_movie_count[user_movie_count["movie_count"] > perc]["userId"].tolist()
# Kaç tane kullanıcı seçilen kullanıcının izlediği filmlerin %60 dan fazlasını izlediğini verir
len(users_same_movies)

###################################################################################
# Görev 4:  ÖneriYapılacakKullanıcıileEnBenzerKullanıcılarınBelirlenmesi
###################################################################################

# Adım 1: user_same_movieslistesiiçerisindekiseçili userile benzerlik gösteren kullanıcıların id’lerininbulunacağı şekilde movies_watched_dfdataframe’inifiltreleyiniz

movies_watched_df.head()

# iki alternatif yol
# 1
# movies_watches_df=movies_watched_df[movies_watched_df.index.isin(users_same_movies)]

movies_watched_df = movies_watched_df.loc[users_same_movies]

movies_watched_df.shape
random_user_df

# Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturunuz

final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])

corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])

corr_df.head()
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

# Adım 3: Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerindeolan) kullanıcıları
# filtreleyerek top_user sadında yeni bir dataframe oluşturunuz.

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)
top_users.head()
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

# Adım 4:  top_users dataframe’ine rating veri seti ile merge ediniz.
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

#top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]


########################################################################################
# Görev 5:  Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
########################################################################################

#Adım 1:   Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_rating
# adında yeni bir değişken oluşturunuz

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

# Adım 2:  Film id’sive her bir filme ait tüm kullanıcıların weighted rating’lerinin ortalama değerini
# içeren recommendation_df adında yeni bir dataframe oluşturunuz.

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

# Adım 3:  recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri
# seçiniz ve weighted rating’e göre sıralayınız.

recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)

# Adım 4:  movie veri setinden film isimlerini getiriniz ve tavsiye edilecek ilk 5 filmi seçiniz.


recommendation_df = recommendation_df.reset_index()
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)

movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"].head(5)


#####################################################################################
#                             ItemBasedRecommendation
######################################################################################

#####################################################################################
# Görev 1:  Kullanıcının izlediği en son veenyüksekpuanverdiğifilmegöreitem-based öneriyapınız.
######################################################################################

# Adım 1:   movie, rating veri setlerini okutunuz

movie = pd.read_csv('recommender_systems/datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('recommender_systems/datasets/movie_lens_dataset/rating.csv')
rating.head()

# Adım 2:  Seçili kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sini alınız
random_user
movieID = rating[(rating["userId"] == random_user) & (rating["rating"] == 5)].sort_values("timestamp", ascending=False).iloc[0, 1]


movie_title = movie.loc[(movie.movieId == movieID), ["title"]].values[0].tolist()

# Adım 3:  User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini seçilen film id’sine
# göre filtreleyiniz.

item_movie_df = user_movie_df[movie_title[0]]

# Adım 4:  Filtrelenen dataframe’i kullanarak seçili filmle  diğer filmlerin korelasyonunu
# bulunuz ve sıralayınız.

corr_df_item_based = user_movie_df.corrwith(item_movie_df).sort_values(ascending=False)


# Adım 5:  Seçili film’in kendisi haricinde ilk 5 film’i öneri olarak veriniz.

corr_df_item_based = corr_df_item_based[corr_df_item_based.index != movie_title[0]]
corr_df_item_based = corr_df_item_based.reset_index()

item_based_recommended = corr_df_item_based["title"].head(5).tolist()

