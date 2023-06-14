import pandas as pd
import numpy as np

movies = pd.read_csv(r'D:\programs\movie-recommender-system\tmdb_5000_movies.csv')
credits = pd.read_csv(r"D:\programs\movie-recommender-system\tmdb_5000_credits.csv")

#to merge the datasets
movies = movies.merge(credits,on = 'title')

# columns to keep-
# id
# genres
# keywords
# original_language
# title
# overview
# cast
# crew

# required columns
movies = movies[["id","genres","keywords","original_language","title","overview","cast","crew"]]

movies.isnull().sum() #movies with missing data
movies.dropna(inplace=True) #remove missing data movies
movies.duplicated().sum() #check duplicate data

import ast

# to show genres in proper format
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i["name"])
    return L

movies['genres'] = movies['genres'].apply(convert) 
movies['keywords']= movies['keywords'].apply(convert) 

#to get name of top 3 cast
def convert2(obj):
    L=[]
    counter =0
    for i in ast.literal_eval(obj):
        if counter !=3:
            L.append(i['name'])
            counter = counter+1
        else:
            break
    return L
movies['cast'] = movies['cast'].apply(convert2) 

#fetch director name from crew
def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']== 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)  #to display director name
 

movies['overview']= movies['overview'].apply(lambda x:x.split())

# to remove spaces
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
# now data is in proper format

#merging all columns into single column 
movies['tags']=movies['overview'] + movies['keywords'] + movies['cast'] + movies['crew']

#merging all columns into single column 
new_df= movies[['id','title','tags']]


# converting list into string
# convert into lowercase
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))

new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())

#--------------------------------------------------------
#vectorisation 
# conversion of text into vectors
# will get info of similiar movies by checking nearby vectors
#using sklearn library

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000, stop_words='english')  
#max fatures 
# stop_words to remove words that dont contribute any meaning

vectors = cv.fit_transform(new_df['tags']).toarray() 
# movie is in vector form 

# for stemming , will use nltk library
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    y =[]
    for i in text.split():
        y.append(ps.stem(i)) 
        #coverting list into string
    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem) 
#stemming applied
#repeat the steps again after stemming is applied

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000, stop_words='english') 
vectors = cv.fit_transform(new_df['tags']).toarray()
# print(vectors)
# cv.get_feature_names()

# calculating distance between vectors(movies)
# distance is inversely proportional to similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)

# using this we will generate 5 similiar movies based on similarity matrix
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse =True, key = lambda x:x[1])[1:6] 
    # top 5 similiar movies will be recommended
    for i in movies_list:
        # print(i[0])  to print movie index
        print(new_df.iloc[i[0]].title) 
        #print movie title

recommend('Batman') 

#to dump data
import pickle
pickle.dump(new_df.to_dict(),open('movies.pkl','wb'))
#rather than sending the dataframe ,  will send in dictionary 

# to dump similarity matrix
pickle.dump(similarity,open('similarity.pkl','wb'))


