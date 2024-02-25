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
    country = request.form.get('country', '')
    try:
        if country == '':
            similar_companies = heimdall.semantic_search(query)
            # Use country filter if provided
        else:
            similar_companies = heimdall.semantic_search(query, country_filter=country)
        return jsonify(similar_companies)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/peer_search', methods=['POST'])
def peer_search():
    symbol = request.form.get('symbol')
    country = request.form.get('country', '')
    try:
        if country == '':
            peer_companies = heimdall.find_peers_of(symbol)  # Use country filter if provided
        else:
            peer_companies = heimdall.find_peers_of(symbol, country=country)
        return jsonify(peer_companies)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/description', methods=['POST'])
def show_description():
    company_symbol = request.form.get('symbol')
    try:
        description = heimdall.show_description(company_symbol)
        return jsonify({"description": description})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)
