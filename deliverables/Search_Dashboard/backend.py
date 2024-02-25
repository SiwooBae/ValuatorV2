from flask import Flask, render_template, request
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import umap
from angle_emb import AnglE

df = pd.read_csv('../database/embeddings_combined.csv')

with sqlite3.connect('../database/valuator.db') as conn:
    description_df = pd.read_sql("""
    SELECT Symbol, companyName, description, industry, sector, country, IPOdate
    FROM profile_v2
    WHERE
        isFund = 0 AND isEtf = 0
    GROUP BY companyName
    ORDER BY symbol ASC
    """, conn)

embedding_cols = [str(i) for i in range(1024)]
scaler = StandardScaler()
df[embedding_cols] = scaler.fit_transform(df[embedding_cols].values)

PCA_reducer = None
UMAP_reducer = None

model = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()


def semantic_search(query, country_filter=None, top=10, dim_reducer=None):
    similar_companies = []
    my_idea = scaler.transform(model.encode(query))

    if country_filter is not None:
        filtered_df = df[df['country'] == country_filter]
    else:
        filtered_df = df

    if dim_reducer is not None:
        my_idea = dim_reducer.transform(my_idea)
        search_database = dim_reducer.transform(filtered_df[embedding_cols])
    else:
        search_database = filtered_df[embedding_cols]

    search_result = cosine_similarity(my_idea, search_database)

    top_indices = np.argsort(search_result[0])[::-1][:top]

    for index in top_indices:
        similar_companies.append(
            f"Score: {search_result[0][index]} "
            f"Symbol: {filtered_df['Symbol'].iloc[index]} "
            f"Company: {filtered_df['companyName'].iloc[index]} ")

    return similar_companies


def find_peers_of(symbol, country=None, top=10, dim_reducer=None):
    query = description_df[description_df['Symbol'] == symbol]['description'].tolist()
    query = query[0]
    return semantic_search(query, country, top=top, dim_reducer=dim_reducer)


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        if query:
            similar_companies = semantic_search(query)
            return render_template('results.html', query=query, similar_companies=similar_companies)
    return render_template('index.html')

@app.route('/peer_search', methods=['POST'])
def peer_search():
    if request.method == 'POST':
        symbol = request.form['symbol']
        if symbol:
            peer_companies = find_peers_of(symbol)
            return render_template('results.html', query=symbol, companies=peer_companies, search_type="Peer Search")
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='localhost', port=8080, debug=True)
