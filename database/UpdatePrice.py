from database.Updater import Updater
import yfinance as yf
import sqlite3
import datetime
import pandas as pd

MIN_date = datetime.date(year=2000, month=1, day=1)
today = datetime.date.today()  # ends this year

conn = sqlite3.connect('valuator.db')
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
                    INSERT OR REPLACE INTO cv_source (symbol, source)
                    VALUES (?, ?)
                """
    params = (sym, source)
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
    FROM profile
    WHERE country = "US" -- only us companies for consistentcy
        AND symbol NOT LIKE "%-%"
        AND symbol NOT LIKE "%.%"
""").fetchall()

symbols = {elem[0] for elem in symbols}

existing_symbol_pairs = conn.execute("""
    SELECT symbol, source, MAX(date) FROM cv LEFT JOIN cv_source USING (symbol) GROUP BY symbol, source
    """).fetchall()

existing_symbols = {elem[0] for elem in existing_symbol_pairs}

new_symbols = symbols - existing_symbols

for i, symbol in enumerate(new_symbols):  # for new symbols,
    try:
        # try yahoo first
        symbol_prices = yf.Ticker(symbol).history(start=MIN_date, end=today).reset_index()
        if symbol_prices.empty:
            raise Exception("yahoo doesn't have price data for this symbol.")
        symbol_prices = symbol_prices.rename(columns={'Date': 'date', 'Close': 'close', 'Volume': 'volume'})
        symbol_prices = symbol_prices[['date', 'close', 'volume']]
        symbol_prices['date'] = symbol_prices['date'].dt.tz_localize(None)
    except:
        try:
            # try fmp datasource
            symbol_prices = u.get_daily_price(symbol, MIN_date, today)
            symbol_prices = pd.DataFrame(symbol_prices)[['date', 'close', 'volume']]
            symbol_prices['date'] = pd.to_datetime(symbol_prices['date'])
        except Exception as error:
            # if API doesn't respond, spit out error
            log_error(symbol, today, str(error))
            # and skip the loop
            continue
        else:
            log_source(symbol, "fmp")
    else:
        log_source(symbol, "yahoo")
    push_into_db(symbol, symbol_prices)
    print("progress:", str(i), "out of", str(len(new_symbols)))

#FIXME: the following for loop is not tested yet.
for i, pair in enumerate(existing_symbol_pairs):  # for existing symbols,
    symbol = pair[0]
    latest_date = pair[1]  # date of the last price info in current db
    og_source = pair[2] # the source of the latest data
    if og_source is None:
        try:
            # try yahoo first
            symbol_prices = yf.Ticker(symbol).history(start=latest_date, end=today).reset_index()
            if symbol_prices.empty:
                raise Exception("yahoo doesn't have price data for this symbol.")
            symbol_prices = symbol_prices.rename(columns={'Date': 'date', 'Close': 'close', 'Volume': 'volume'})
            symbol_prices = symbol_prices[['date', 'close', 'volume']]
            symbol_prices['date'] = symbol_prices['date'].dt.tz_localize(None)
        except:
            try:
                # try fmp datasource
                symbol_prices = u.get_daily_price(symbol, latest_date, today)
                symbol_prices = pd.DataFrame(symbol_prices)[['date', 'close', 'volume']]
                symbol_prices['date'] = pd.to_datetime(symbol_prices['date'])
            except Exception as error:
                # if API doesn't respond, spit out error
                log_error(symbol, today, str(error))
                # and skip the loop
                continue
            else:
                log_source(symbol, "fmp")
        else:
            log_source(symbol, "yahoo")
        push_into_db(symbol, symbol_prices)

    elif og_source == "yahoo": # keep using the same source
        symbol_prices = yf.Ticker(symbol).history(start=latest_date, end=today).reset_index()
        if symbol_prices.empty: # if it is empty
            log_error(symbol, today, "yahoo doesn't have price data for this symbol in this date interval.")
            continue
        symbol_prices = symbol_prices.rename(columns={'Date': 'date', 'Close': 'close', 'Volume': 'volume'})
        symbol_prices = symbol_prices[['date', 'close', 'volume']]
        symbol_prices['date'] = symbol_prices['date'].dt.tz_localize(None)
        push_into_db(symbol, symbol_prices)

    elif og_source == "fmp":
        symbol_prices = u.get_daily_price(symbol, latest_date, today)
        symbol_prices = pd.DataFrame(symbol_prices)[['date', 'close', 'volume']]
        symbol_prices['date'] = pd.to_datetime(symbol_prices['date'])
        push_into_db(symbol, symbol_prices)

    print("progress:", str(i), "out of", str(len(existing_symbols)))
