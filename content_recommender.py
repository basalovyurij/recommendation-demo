import os.path

import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import tempfile
from zipfile import ZipFile


DATA_DIR = 'movielens_data'
DATASET = 'ml-latest-small'


def download_movielens():
    if os.path.exists(DATA_DIR):
        return

    url = f'http://files.grouplens.org/datasets/movielens/{DATASET}.zip'
    os.mkdir(DATA_DIR)
    print(f'Downloading movielens...')
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size_in_bytes= int(r.headers.get('content-length', 0))
        with tempfile.NamedTemporaryFile(mode='rb+') as temp_f:
            downloaded = 0
            dl_iteration = 0
            chunk_size = 8192
            total_chunks = total_size_in_bytes / chunk_size if total_size_in_bytes else 100
            for chunk in r.iter_content(chunk_size=chunk_size):
                downloaded += chunk_size
                dl_iteration += 1
                percent = (100 * dl_iteration * 1.0/total_chunks)
                if dl_iteration % 10 == 0 and percent < 100:
                    print(f'Completed {percent:2f}%')
                elif percent >= 99.9:
                    print(f'Download completed. Now unzipping...')
                temp_f.write(chunk)
            with ZipFile(temp_f, 'r') as zipf:
                zipf.extractall(DATA_DIR)
                print(f"\nUnzipped.\n\nFiles downloaded and unziped to: {DATA_DIR}")


# the function to extract titles
def extract_title(title):
    year = title[len(title) - 5:len(title) - 1]

    # some movies do not have the info about year in the column title. So, we should take care of the case as well.
    if year.isnumeric():
        title_no_year = title[:len(title) - 7]
        return title_no_year
    else:
        return title


# the function to extract years
def extract_year(title):
    year = title[len(title)-5:len(title)-1]

    # some movies do not have the info about year in the column title. So, we should take care of the case as well.
    if year.isnumeric():
        return int(year)
    else:
        return np.nan


download_movielens()

movies = pd.read_csv(os.path.join(DATA_DIR, DATASET, 'movies.csv'))

movies.rename(columns={'title': 'title_year'}, inplace=True)  # change the column name from title to title_year
movies['title_year'] = movies['title_year'].apply(lambda x: x.strip())  # remove leading and ending whitespaces in title_year
movies['title'] = movies['title_year'].apply(extract_title)  # create the columns for title and year
movies['year'] = movies['title_year'].apply(extract_year)

movies = movies[~(movies['genres'] == '(no genres listed)')].reset_index(drop=True)  # remove the movies without genre information and reset the index

movies['genres'] = movies['genres'].str.replace('|', ' ', regex=False)  # remove '|' in the genres column

counts = dict()  # count the number of occurences for each genre in the data set
for i in movies.index:
   for g in movies.loc[i,'genres'].split(' '):
      if g not in counts:
         counts[g] = 1
      else:
         counts[g] = counts[g] + 1

plt.figure(figsize=(9, 7))
plt.bar(list(counts.keys()), counts.values(), color='g')  # create a bar chart
plt.xticks(rotation=45)
plt.xlabel('Genres')
plt.ylabel('Counts')
plt.show()

movies['genres'] = movies['genres'].str.replace('Sci-Fi', 'SciFi')  # change 'Sci-Fi' to 'SciFi' and 'Film-Noir' to 'Noir'
movies['genres'] = movies['genres'].str.replace('Film-Noir', 'Noir')

tfidf_vector = TfidfVectorizer(stop_words='english')  # create an object for TfidfVectorizer
tfidf_matrix = tfidf_vector.fit_transform(movies['genres'])  # apply the object to the genres column
sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)  # sim_matrix = linear_kernel(tfidf_matrix,tfidf_matrix)


movie_user_likes = 'Monsters, Inc.'
how_many = 20
movie_index = movies[movies.title == movie_user_likes].index.values[0]
movie_list = list(enumerate(sim_matrix[int(movie_index)]))
similar_movies = list(filter(     # remove the typed movie itself
    lambda x: x[0] != int(movie_index),
    sorted(movie_list, key=lambda x: x[1], reverse=True)))

print(f'Here\'s the list of movies similar to \033[1m{movie_user_likes}\033[0m.\n')
for i, s in similar_movies[:how_many]:
    print(movies[movies.index == i]['title'].values[0])
