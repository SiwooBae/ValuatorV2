from flask import Flask, render_template, request, jsonify
from Heimdall import Heimdall

heimdall = Heimdall()
app = Flask(__name__, static_url_path='/static')


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    try:
        similar_companies = heimdall.semantic_search(query)
        return jsonify(similar_companies)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/peer_search', methods=['POST'])
def peer_search():
    symbol = request.form.get('symbol')
    try:
        peer_companies = heimdall.find_peers_of(symbol)
        return jsonify(peer_companies)
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(host='localhost', port=8080, debug=True)
