from urllib.request import urlopen
import certifi
import json
from io import StringIO
import pandas as pd

APIKEY = '???'


def get_jsonparsed_data(url):
    response = urlopen(url + "&apikey=" + APIKEY, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)

# This class is responsible for downloading the newest data from FinancialModelingPrep API to my database.
class Updater:
    def __init__(self):
        self.dropped_features = ['link', 'finalLink']
        self.stock_exchange = ['NYSE', 'NASDAQ']
        self.url_root = "https://financialmodelingprep.com/api"

    def get_balance_sheet(self, symbol, period='quarter'):
        data = get_jsonparsed_data(self.url_root + "/v3/balance-sheet-statement/" + symbol + "?" + "period=" + period
                                   )
        return pd.read_json(data)

    def get_income_statement(self, symbol, period='quarter'):
        data = get_jsonparsed_data(self.url_root + "/v3/income-statement/" + symbol + "?" + "period=" + period
                                   )
        return pd.read_json(data)

    def get_cash_flow_statement(self, symbol, period='quarter'):
        data = get_jsonparsed_data(self.url_root + "/v3/cash-flow-statement/" + symbol + "?" + "period=" + period
                                   )
        return pd.read_json(data)

    def get_balance_sheet_bulk(self, year, period='quarter'):
        url = self.url_root + "/v4/balance-sheet-statement-bulk?year=" + str(year) + "&" + "period=" + period
        response = urlopen(url + "&apikey=" + APIKEY, cafile=certifi.where())
        data = StringIO(response.read().decode("utf-8"))
        return pd.read_csv(data)

    def get_income_statement_bulk(self, year, period='quarter'):
        url = self.url_root + "/v4/income-statement-bulk?year=" + str(year) + "&" + "period=" + period
        response = urlopen(url + "&apikey=" + APIKEY, cafile=certifi.where())
        data = StringIO(response.read().decode("utf-8"))
        return pd.read_csv(data)

    def get_cash_flow_statement_bulk(self, year, period='quarter'):
        url = self.url_root + "/v4/cash-flow-statement-bulk?year=" + str(year) + "&" + "period=" + period
        response = urlopen(url + "&apikey=" + APIKEY, cafile=certifi.where())
        data = StringIO(response.read().decode("utf-8"))
        return pd.read_csv(data)

    def get_all_profiles(self):
        return get_jsonparsed_data(self.url_root + "/v4/profile/all?")

    def get_all_symbols_with_financial_statements(self):
        return get_jsonparsed_data(self.url_root + "/v3/financial-statement-symbol-lists?")

    def get_symbol_changes(self):
        return get_jsonparsed_data(self.url_root + "/v4/symbol_change?")

    def get_delisted_symbols(self):
        return get_jsonparsed_data(self.url_root + "/v4/delisted-companies?")

    def get_stock_peers(self, symbol):
        return get_jsonparsed_data(self.url_root + "/v4/stock_peers?symbol=" + symbol)

    def get_daily_price(self, symbol, date_from, date_to):
        return get_jsonparsed_data(
            self.url_root + "/v3/historical-chart/1day/" + symbol + "?from="+str(date_from)+"&to="+str(date_to))
