from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
df = pd.read_csv('movie_data.csv')


@app.route('/')
def hello_world():
    return render_template("index.html")


@app.route('/search', methods=['POST'])
def search_movie():
    data = df.copy()
    movie = str(request.form.get("movie"))
    movie_id = data[data.Title == movie]['Movie_id'].values[0]
    requested_columns = []
    for key, val in request.form.items():
        if key != 'movie':
            requested_columns.append(val)

    data['important_features'] = get_important_columns(requested_columns)
    cm = CountVectorizer().fit_transform(data['important_features'])
    cosine_similarity_matrix = cosine_similarity(cm)

    # Cosine similarity scores [ (movie_id, similarity score), (...)]
    cs_scores = enumerate(cosine_similarity_matrix[movie_id])
    sorted_cs_scores = sorted(cs_scores, key = lambda x:x[1], reverse=True)

    # Ignore 1st datapoint as it is the same movie
    sorted_cs_scores = sorted_cs_scores[1:]

    movies = []
    for i in range(10):
        curr_id = sorted_cs_scores[i][0]
        movie_title = data[data.Movie_id == curr_id]['Title'].values[0]
        movies.append(movie_title)
    return render_template('index.html', movies=movies)


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

def get_important_columns(req_features):
    imp_columns = []
    for i in range(0, df.shape[0]):
        str_builder = ""
        for column in req_features:
            str_builder += str(df[column][i])
            str_builder += ' '
        imp_columns.append(str_builder)
    return imp_columns


if __name__ == '__main__':
    app.run(debug=True)
