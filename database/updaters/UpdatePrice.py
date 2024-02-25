from database.updaters.Updater import Updater
import yfinance as yf
import sqlite3
import datetime
import pandas as pd

MIN_date = datetime.date(year=2000, month=1, day=1)
today = datetime.date.today()  # ends this year

conn = sqlite3.connect('../valuator.db')
u = Updater()


def log_error(sym, date, msg):
    insert_query = """
                INSERT OR REPLACE INTO cv_error_log (symbol, error_date, msg)
                VALUES (?, ?, ?)
            """
    params = (sym, date, msg)
    conn.execute(insert_query, params)
    conn.commit()
    print("error occured! message:", msg)


def log_source(sym, source):
    insert_query = """
                    INSERT OR REPLACE INTO cv_source
                    VALUES (?, ?, ?)
                """
    params = (sym, source, today)
    conn.execute(insert_query, params)
    conn.commit()
    print("using", source, "API...")


def push_into_db(sym, prices):
    symbol_min_date = prices['date'].min()
    df = pd.DataFrame(pd.date_range(start=symbol_min_date, end=today), columns=['date'])
    df = df.merge(prices, how='left', on='date')
    df = df.fillna(method='ffill', limit=15)
    df['symbol'] = sym
    df = df[['symbol', 'date', 'close', 'volume']]
    try:
        df.to_sql('cv', conn, if_exists='append', index=False)
    except Exception as error:
        if type(error).__name__ == "OverflowError":
            df['close'] = df['close'].astype(str)
            df['volume'] = df['volume'].astype(str)
            df.to_sql('cv', conn, if_exists='append', index=False)
            log_error(sym, today, "had to convert to str for the value being too large")
        else:
            log_error(sym, today, str(error))
    else:
        print("successfully put", sym, "into the database.")


def insert_using_Yahoo(sym, begin_date):
    symbol_prices = yf.Ticker(sym).history(start=begin_date, end=today).reset_index()
    if symbol_prices.empty:
        raise Exception("yahoo doesn't have price data for this symbol.")
    symbol_prices = symbol_prices.rename(columns={'Date': 'date', 'Close': 'close', 'Volume': 'volume'})
    symbol_prices = symbol_prices[['date', 'close', 'volume']]
    symbol_prices['date'] = symbol_prices['date'].dt.tz_localize(None)
    log_source(sym, "yahoo")
    push_into_db(sym, symbol_prices)


def insert_using_fmp(sym, begin_date):
    symbol_prices = u.get_daily_price(sym, begin_date, today)
    symbol_prices = pd.DataFrame(symbol_prices)[['date', 'close', 'volume']]
    symbol_prices['date'] = pd.to_datetime(symbol_prices['date'])
    log_source(sym, "fmp")
    push_into_db(sym, symbol_prices)


## uncomment the following code to download ALL symbols that have finanncial statements

# symbols = conn.execute("""
#     SELECT DISTINCT symbol
#     FROM balance
# 	    INNER JOIN income USING (symbol, date)
# 	    INNER JOIN cashflow USING (symbol, date)
# """).fetchall()

# The current code (following sql query) only fetches US companies that have profiles.

symbols = conn.execute("""
    SELECT DISTINCT symbol
    FROM profile_v2
    WHERE country = "US" -- only us companies for consistentcy
        AND symbol NOT LIKE "%-%"
        AND symbol NOT LIKE "%.%"
""").fetchall()

symbols = {elem[0] for elem in symbols}

print("fetching existing symbol pairs...")
existing_symbol_pairs = conn.execute("""
SELECT cv_latest.symbol, cv_source_latest.source as source, cv_latest.maxdate
FROM (SELECT symbol, MAX(date) as maxdate FROM cv GROUP BY symbol) as cv_latest
    JOIN cv_source_latest USING (symbol)
    """).fetchall()

existing_symbols = {elem[0] for elem in existing_symbol_pairs}

new_symbols = symbols - existing_symbols

# print(len(new_symbols), "new symbols detected. Updating new symbols...")
# for i, symbol in enumerate(new_symbols):  # for new symbols,
#     try:
#         # try yahoo first
#         insert_using_Yahoo(symbol, MIN_date)
#     except Exception as error_yahoo:
#         log_error(symbol, today, str(error_yahoo))
#         try:
#             # try fmp datasource
#             insert_using_fmp(symbol, MIN_date)
#         except Exception as error_fmp:
#             # if API doesn't respond, spit out error
#             log_error(symbol, today, str(error_fmp))
#             # and skip the loop
#             continue
#     print("progress:", str(i), "out of", str(len(new_symbols)))

print(len(existing_symbols), "existing symbols detected. Updating existing symbols...")
for i, pair in enumerate(existing_symbol_pairs):  # for existing symbols,
    symbol = pair[0]
    og_source = pair[1]
    latest_date = datetime.datetime.strptime(pair[2], '%Y-%m-%d %H:%M:%S').date()
    # the source of the latest data
    if og_source is None or og_source == "yahoo":
        try:
            # try yahoo first
            insert_using_Yahoo(symbol, latest_date)
        except Exception as error_yahoo:
            log_error(symbol, today, str(error_yahoo))
            try:
                # try fmp datasource
                insert_using_fmp(symbol, latest_date)
            except Exception as error_fmp:
                # if API doesn't respond, spit out error
                log_error(symbol, today, str(error_fmp))
                # and skip the loop
                continue

    elif og_source == "fmp":
        try:
            # try fmp first
            insert_using_fmp(symbol, latest_date)
        except Exception as error_fmp:
            log_error(symbol, today, str(error_fmp))
            try:
                # try yahoo datasource
                insert_using_Yahoo(symbol, latest_date)
            except Exception as error_yahoo:
                # if API doesn't respond, spit out error
                log_error(symbol, today, str(error_yahoo))
                # and skip the loop
                continue

    print("progress:", str(i + 1), "out of", str(len(existing_symbols)))
