import streamlit as st
import pickle
import pandas as pd
import requests


#fetching posters through API
def fetch_posters(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=e6e79c39e903349fc7910469504bbe4f&language=en-US'.format(movie_id))
    data = response.json() #converting into json
    
    return"https://image.tmdb.org/t/p/w500/" + data['poster_path'] 


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse =True, key = lambda x:x[1])
    recommended_movies =[]
    recommended_movies_posters = []
    for i in movies_list[1:6]:
        movie_id = movies.iloc[i[0]].id 
        #useful to fetch posters
        #i[0] to get movie index 
        recommended_movies.append(movies.iloc[i[0]].title)
        #fetch poster
        recommended_movies_posters.append(fetch_posters(movie_id)) 

    return recommended_movies,recommended_movies_posters 
# it will provide name, posters of movies


#to load movies
movies_dict = pickle.load(open('D:\programs\movie-recommender-system\movies.pkl','rb'))
movies = pd.DataFrame(movies_dict)

#to load similarity
similarity = pickle.load(open('D:\programs\movie-recommender-system\similarity.pkl','rb'))

st.title('Movie Shelf')
#fetch movie name
selected_movie_name = st.selectbox(
    'What would you like to search?', movies['title'].values
)

#recommendation button code
if st.button('Display Recommendation'):
    names,posters = recommend(selected_movie_name)
    c1 , c2 , c3 , c4 , c5 = st.columns(5)
    with c1:
        st.text(names[0])
        st.image(posters[0])
    with c2:
        st.text(names[1])
        st.image(posters[1])
    with c3:
        st.text(names[2])
        st.image(posters[2])
    with c4:
        st.text(names[3])
        st.image(posters[3])
    with c5:
        st.text(names[4])
        st.image(posters[4])

