import os.path
import pandas as pd
import requests
import tempfile
import turicreate as tc
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


download_movielens()

data = tc.SFrame.read_csv(os.path.join(DATA_DIR, DATASET, 'ratings.csv'))
train, test = tc.recommender.util.random_split_by_user(data, 'userId', 'movieId')

f_model = tc.factorization_recommender.create(train, user_id='userId', item_id='movieId', target='rating')
s_model = tc.item_similarity_recommender.create(train, 'userId', 'movieId', target='rating')
rf_model = tc.ranking_factorization_recommender.create(train, 'userId', 'movieId', target='rating')
models = [f_model, s_model, rf_model]
model_names = ['Factorization', 'Item similarity', 'Ranking factorization']
tc.recommender.util.compare_models(test, models, model_names=model_names, metric='rmse', target='rating')
