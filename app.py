from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
df = pd.read_csv('movie_data.csv')

user_scores = []
optimal_scores = []
optimal_chart_data = 0
user_chart_data = 0
movie_titles = list(df['Title'])


@app.route('/')
def hello_world():
    return render_template("index.html", movie_titles=movie_titles)


@app.route('/optimal')
def view_optimal_results():
    bar_chart_data= [optimal_chart_data, user_chart_data]
    return render_template('optimal.html', movie_titles=movie_titles, user_results=user_scores, optimal_results=optimal_scores, optimal_chart_data=bar_chart_data)


@app.route('/search', methods=['POST'])
def search_movie():
    data = df.copy()
    data_optimal = df.copy()

    movie = str(request.form.get("movie"))

    # If form is missing paramaters return an error message
    if len(request.form.to_dict()) == 1:
        return render_template('index.html', previous_input = movie, movie_titles=movie_titles, error_message="Please select a paramater.")

    try:
        movie_id = data[data.Title == movie]['Movie_id'].values[0]
    except:
        return render_template('index.html', movie_titles=movie_titles, error_message="Movie not found :(")

    requested_columns = []
    for key, val in request.form.items():
        if key != 'movie':
            requested_columns.append(val)

    optimal_columns = ['Actors', 'Director', 'Description', 'Genre']

    data_optimal['important_features'] = get_important_columns(optimal_columns)
    cm_optimal = CountVectorizer().fit_transform(data_optimal['important_features'])
    cosine_similarity_matrix_optimal = cosine_similarity(cm_optimal)

    data['important_features'] = get_important_columns(requested_columns)
    cm = CountVectorizer().fit_transform(data['important_features'])
    cosine_similarity_matrix = cosine_similarity(cm)

    # Cosine similarity scores [ (movie_id, similarity score), (...)]
    cs_scores = enumerate(cosine_similarity_matrix[movie_id])
    cs_scores_optimal = enumerate(cosine_similarity_matrix_optimal[movie_id])

    sorted_cs_scores_optimal = sorted(cs_scores_optimal, key = lambda x:x[1], reverse=True)
    sorted_cs_scores = sorted(cs_scores, key = lambda x:x[1], reverse=True)

    get_chart_data(sorted_cs_scores_optimal, sorted_cs_scores)

    # Ignore 1st datapoint as it is the same movie
    doughnut_chart_data = get_doughnut_data(sorted_cs_scores)

    sorted_cs_scores = sorted_cs_scores[1:11]
    sorted_cs_scores_optimal = sorted_cs_scores_optimal[1:11]

    movies = []
    movies_optimal = []
    for i in range(10):
        curr_id = sorted_cs_scores[i][0]
        curr_id_optimal = sorted_cs_scores_optimal[i][0]

        movie_title = data[data.Movie_id == curr_id]['Title'].values[0]
        movie_title_optimal = data_optimal[data_optimal.Movie_id == curr_id_optimal]['Title'].values[0]

        movie_data = (movie_title, round(100 * sorted_cs_scores[i][1], 1))
        movie_data_optimal = (movie_title_optimal, round(100 * sorted_cs_scores_optimal[i][1], 1))

        movies.append(movie_data)
        movies_optimal.append(movie_data_optimal)

    global user_scores
    user_scores = movies
    global optimal_scores
    optimal_scores= movies_optimal

    return render_template('index.html', results=movies, data_values=doughnut_chart_data)


def get_chart_data(cs_optimal_scores, cs_user_scores):
    global optimal_chart_data
    global user_chart_data

    optimal_chart_data = len([x for x in cs_optimal_scores if x[1] > 0.0])
    user_chart_data = len([x for x in cs_user_scores if x[1] > 0.0])


def get_doughnut_data(cs_scores):

    # data = [80%+, 60-80, 40-60, 20-40, 20-]
    data = [0,0,0,0,0]
    for item in cs_scores:
        s = item[1] * 100
        if s >= 60:
            data[0] += 1
        elif s < 60 and s >= 40:
            data[1] += 1
        elif s < 40 and s >= 20:
            data[2] += 1
        elif s < 20 and s >= 10:
            data[3] += 1
        elif s < 10 and s > 0:
            data[4] += 1
    return data


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
