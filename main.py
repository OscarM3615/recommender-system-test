import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

movies: pd.DataFrame = pd.read_csv('movies.csv')

# Term frequency-inverse document frequency
tfdif = TfidfVectorizer(stop_words='english')

movies['overview'] = movies['overview'].fillna('')

# Construct the required TF-IDF matrix by applying fit_transform
overview_matrix = tfdif.fit_transform(movies['overview'])
similarity_matrix = linear_kernel(overview_matrix, overview_matrix)
mapping = pd.Series(movies.index, index=movies['title'])

def recommend_movies(movie_input, result_count=15):
    movie_index = mapping[movie_input]

    # get similarity values with other movies
    similarity_score = list(enumerate(similarity_matrix[movie_index]))

    # sort in descending order the similarity score of movie inputted with other movies
    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    # get the score of the most similar movies, skip first (will be the same)
    similarity_score = similarity_score[1:result_count + 1]

    # return movie names using the Series
    movie_indices = [i[0] for i in similarity_score]
    return movies['title'].iloc[movie_indices].to_list()

def main():
    should_exit = False

    while not should_exit:
        try:
            movie = input('Enter a movie name to get recommendations: ')
            recommendations = recommend_movies(movie)
            print('[i] Results:\n' + '\n'.join(f'- {x}' for x in recommendations))
        except KeyboardInterrupt:
            print('')
            should_exit = True
        except KeyError:
            print('[*] The movie does not exist int the dataset.')

if __name__ == '__main__':
    main()
