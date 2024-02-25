import sqlite3
import datetime
import pandas as pd
from classes.OutlierPruner import OutlierPruner
import numpy as np
import xgboost as xgb
import sklearn.metrics as metrics
import scipy
from tqdm import tqdm
from itertools import product


def reorder_dataframe(df, column_order):
    """
    Reorders a DataFrame based on a given list of column names.

    :param df: pandas DataFrame
    :param column_order: List of column names to order the DataFrame by
    :return: Reordered DataFrame
    """
    # Remove duplicates while preserving order
    seen = set()
    unique_column_order = [col for col in column_order if col not in seen and not seen.add(col)]

    # Filter out columns that are not in the DataFrame
    valid_columns = [col for col in unique_column_order if col in df.columns]

    extra_columns = [col for col in df.columns if col not in valid_columns]

    # Reorder the DataFrame
    reordered_df = df[valid_columns + extra_columns]

    return reordered_df


conn = sqlite3.connect('../database/valuator.db')

param = {}  # a dict that stores all the parameter values

param["date_begin"] = [datetime.datetime(2000, 1, 1)]  # earliest of the whole data
param["val_split_point"] = [datetime.datetime(2020, 1, 1)] # valid date begins with this date
param["date_end"] = [datetime.datetime(2023, 12, 1)]  # latest of the whole data

# preprocessing steps for x (financial statements)
param["TTM"] = [True, False]  # uses trailing twelve months average data instead of raw
param["adjust_price_w_CPI"] = [True, False]  # allows adjusting prices with CPI data
param["apply_asset_constraint"] = [True, False]  # allows applying the asset constraint (asset = equity + liability)
param[
    "apply_nonnegative_constraint"] = [True,
                                       False]  # deletes rows where their supposedly non-negative columns have negative column values
param["divide_by_total_assets"] = [True,
                                   False]  # divide the relevant columns with total assets to diminish multicolinearity
param["filter_by_volume"] = [100, 1000, 10000, 100000]

param["prune_outlier"] = [True, False]  # prune outlier
param["outlier_quantile"] = [0.99, 0.999]  # cutoff value determined by quantile. 0.99 means keep 99% of the values
param["outlier_thresh"] = [0, 2]  # rows up to this many outlier value will survive

# preprocessing steps for y (close)
param[
    "adjust_using_SPY"] = [True]  # use market-adjusted log returns instead of regular returns. Removes any market-induced movements
param["window_length"] = [120, 360, 540]  # days forward to calculate mean (or std) log returns
param[
    "min_periods"] = [120]  # minimum days needed to calculate the mean (or std) log returns. Higher value may lead to survivorship bias
param["use_information_ratio"] = [True, False]  # uses information ratio instead of mean

combinations = list(product(*param.values()))
log = pd.DataFrame(combinations, columns=param.keys())

log.loc[log['prune_outlier'] == False, ['outlier_quantile', 'outlier_thresh']] = -1
log = log.drop_duplicates()

balance_ignored_features = ['goodwillAndIntangibleAssets',
                            'totalLiabilitiesAndStockholdersEquity',
                            'netDebt',
                            'cashAndShortTermInvestments']
income_ignored_features = ['grossProfitRatio',
                           'costAndExpenses',
                           'EBITDARatio',
                           'operatingIncomeRatio',
                           'incomeBeforeTaxRatio',
                           'netIncomeRatio',
                           'EPSDiluted',
                           'weightedAverageShsOut',
                           'weightedAverageShsOutDil',
                           'interestIncome']
cashflow_ignored_features = []  # nothing yet
macro_ignored_features = ['realGDP', "consumerSentiment", "smoothedUSRecessionProbabilities"]


df = pd.read_sql(f"""
SELECT * FROM bic
WHERE symbol in
        (SELECT symbol FROM profile_v2
        WHERE country = "US"
        AND isFund = 0
        AND isEtf = 0
        AND currency = "USD"
        AND (exchangeShortName = "NASDAQ" OR exchangeShortName = "NYSE"))
    AND symbol NOT LIKE '%W'
    AND fillingDate BETWEEN date('2000-01-01') AND date('2023-12-01')
ORDER BY symbol, date ASC
""", con=conn)

macro_df = pd.read_sql("""
SELECT * FROM macro_US
""", con=conn)

close_volume = pd.read_sql(f"""
SELECT date, symbol, close, volume
FROM cv
WHERE symbol in (SELECT symbol FROM profile_v2
    WHERE country = "US"
    AND isFund = 0
    AND isEtf = 0
    AND currency = "USD"
    AND (exchangeShortName = "NASDAQ" OR exchangeShortName = "NYSE"))
    AND date BETWEEN date('2000-01-01') AND date('2023-12-01')
""", con=conn, index_col=["symbol", "date"])

spy = pd.read_sql(f"""
SELECT date, symbol, close
FROM cv
WHERE symbol = "SPY" AND date BETWEEN date('2000-01-01') AND date('2023-12-01')
""", con=conn, index_col=["symbol", "date"])


spy = spy["close"].unstack(level=0)

cursor = conn.execute("""SELECT * FROM balance""")
balance_columns = [description[0] for description in cursor.description]

cursor = conn.execute("""SELECT * FROM income""")
income_columns = [description[0] for description in cursor.description]

cursor = conn.execute("""SELECT * FROM cashflow""")
cashflow_columns = [description[0] for description in cursor.description]

df['fillingDate'] = pd.to_datetime(df['fillingDate'])
df['date'] = pd.to_datetime(df['date'])
macro_df['date'] = pd.to_datetime(macro_df['date'])
CPI_base = macro_df[macro_df['date'] == datetime.datetime(2020, 1, 1)]['CPI'].values
macro_df['CPI'] = macro_df['CPI'] / CPI_base

df = df.merge(macro_df, how="left", on="date")
df = df.set_index(['symbol', 'fillingDate'])
df = df.drop(columns=['date', 'reportedCurrency'])

df['federalFunds'] = df['federalFunds'] * df['CPI']
df = df.divide(df['CPI'], axis=0)
df = df.drop(columns='CPI')

df = df.drop(
    columns=balance_ignored_features + income_ignored_features + cashflow_ignored_features + macro_ignored_features)

other_dropped_columns = [column for column in df.columns if
                         "other" in str(column).lower() or "1" in str(column).lower()]
df = df.drop(columns=other_dropped_columns)

volumes = close_volume['volume']
volumes.index = volumes.index.set_levels([volumes.index.levels[0], pd.to_datetime(volumes.index.levels[1])])
df_outer = df.merge(volumes, how='inner', left_index=True, right_on=['symbol', 'date'])

close_prices = close_volume['close'].unstack(level=0)
close_prices = pd.merge(close_prices, spy, "left", on='date')
close_prices[close_prices <= 0] = pd.NA  # turn any nonpositive numbers to Null values.
close_prices = close_prices.dropna(how='all', axis=1)  # now drop all symbols that have no values at all
close_prices = np.log(close_prices)  # apply log to every value
close_prices = close_prices.diff()  # calculate difference

close_prices_outer = close_prices.subtract(close_prices["SPY"], axis=0)

#grid search begins here

for i in tqdm(range(366, len(log))):
    param = log.iloc[i].to_dict()
    df = df_outer
    close_prices = close_prices_outer

    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=param["window_length"])
    close_prices_mean = close_prices.rolling(window=indexer,
                                             min_periods=param["min_periods"]).mean()
    close_prices_var = close_prices.rolling(window=indexer,
                                            min_periods=param["min_periods"]).var()
    close_prices_mean = close_prices_mean * 365  # annualize the data
    close_prices_var = close_prices_var * 365

    close_prices_mean.index = pd.to_datetime(close_prices_mean.index)
    close_prices_mean = close_prices_mean.stack()

    close_prices_var.index = pd.to_datetime(close_prices_var.index)
    close_prices_var = close_prices_var.stack()

    close_prices_mean.name = "log_ret_mean"
    close_prices_var.name = "log_ret_var"

    close_prices_total = pd.concat([close_prices_mean, close_prices_var], axis=1)

    if param["use_information_ratio"]:
        close_prices_total['log_ret_mean'] = close_prices_total['log_ret_mean'] / np.sqrt(
            close_prices_total['log_ret_var'])

    df = reorder_dataframe(df, balance_columns + income_columns + cashflow_columns)

    if param["TTM"]:
        df = df.groupby(level=0).rolling(4, min_periods=4).mean().droplevel(0)
        df = df.dropna(axis=0, how="any")

    if param["apply_asset_constraint"]:
        asset_current_mask = np.abs(
            (df['totalCurrentAssets'] + df['totalNonCurrentAssets']) / df['totalAssets']) < 1.001
        asset_total_mask = np.abs((df['totalStockholdersEquity'] + df['totalLiabilities']) / df['totalAssets']) < 1.001
        mask = np.logical_and(asset_total_mask, asset_current_mask)
        df = df[mask]

    if param["apply_nonnegative_constraint"]:
        nonnegative_constraint_columns = []
        for column in df.columns:
            if df[column].min() < 0 and np.sum(df[column] < 0) < 0.01 * len(df):
                nonnegative_constraint_columns.append(column)

        for column in nonnegative_constraint_columns:
            df = df[df[column] >= 0]

    if param["divide_by_total_assets"]:
        df = df[df['totalAssets'] > 0]
        unaffected_cols = ['federalFunds', 'totalAssets', 'volume']
        affected_cols = [col for col in df.columns if col not in unaffected_cols]
        df[affected_cols] = df[affected_cols].div(df['totalAssets'], axis=0)
        df['totalAssets'] = np.log(df['totalAssets'])

    if param["filter_by_volume"] > 0:
        df.dropna(how='any', inplace=True)
        df = df[df['volume'] > param["filter_by_volume"]]

    df['volume'] = np.log(df['volume'])

    df_ready = df.merge(close_prices_total, how='inner', left_index=True, right_on=['symbol', 'date'])
    df_ready = df_ready.swaplevel(0, 1, axis=0)

    train_mask = df_ready.index.get_level_values('date') < (
            param["val_split_point"] - datetime.timedelta(days=param["window_length"]))
    valid_mask = df_ready.index.get_level_values('date') >= param["val_split_point"]
    train = df_ready.loc[train_mask].sample(frac=1)
    valid = df_ready.loc[valid_mask].sample(frac=1)

    if len(train) < 100 or len(valid) < 100:
        param["ret_r_mean"] = -1
        param["ret_r_var"] = -1
        param["ret_cls_auc"] = -1

        param["date_begin"] = param["date_begin"].date()
        param["val_split_point"] = param["val_split_point"].date()
        param["date_end"] = param["date_end"].date()

        placeholders = ', '.join(['?'] * len(param))
        columns = ', '.join(param.keys())
        sql = 'INSERT INTO finstat_experiment_log({}) VALUES ({})'.format(columns, placeholders)

        cursor = conn.cursor()
        cursor.execute(sql, tuple(param.values()))
        conn.commit()
        print("warning: dataset has become to small to train a model on.")
        continue

    if param["prune_outlier"]:
        pruner = OutlierPruner()
        print("shape before pruning:", train.shape, valid.shape)
        pruner.fit(train, ['federalFunds', 'volume', "log_ret_mean", "log_ret_var"], q=param["outlier_quantile"])
        train = pruner.transform(train, thresh=param["outlier_thresh"])
        valid = pruner.transform(valid, thresh=param["outlier_thresh"])
        print("shape after pruning:", train.shape, valid.shape)

    train_x, train_y_mean, train_y_var = train.drop(columns=["log_ret_mean", "log_ret_var"]).to_numpy(), train[
        "log_ret_mean"].to_numpy(), train["log_ret_var"].to_numpy()

    valid_x, valid_y_mean, valid_y_var = valid.drop(columns=["log_ret_mean", "log_ret_var"]).to_numpy(), valid[
        "log_ret_mean"].to_numpy(), valid["log_ret_var"].to_numpy()

    train_y_var = np.log(train_y_var)
    valid_y_var = np.log(valid_y_var)

    try:
        xgb_mean_model = xgb.XGBRegressor(eta=0.03, device='cuda', objective='reg:pseudohubererror',
                                          early_stopping_rounds=30,
                                          max_bin=512, n_estimators=300)
        xgb_mean_model.fit(train_x, train_y_mean, eval_set=[(valid_x, valid_y_mean)])
        valid_pred_mean = xgb_mean_model.predict(valid_x)
        param["ret_r_mean"], ret_p_mean = scipy.stats.pearsonr(valid_pred_mean, valid_y_mean)
    except:
        param["ret_r_mean"] = -1

    try:
        xgb_var_model = xgb.XGBRegressor(eta=0.01, device='cuda', objective='reg:pseudohubererror',
                                         early_stopping_rounds=30,
                                         max_bin=512, n_estimators=500)
        xgb_var_model.fit(train_x, train_y_var, eval_set=[(valid_x, valid_y_var)])
        valid_pred_var = xgb_var_model.predict(valid_x)
        param["ret_r_var"], ret_p_var = scipy.stats.pearsonr(valid_pred_var, valid_y_var)
    except:
        param["ret_r_var"] = -1

    train_y_mean_cls = train_y_mean >= 0
    valid_y_mean_cls = valid_y_mean >= 0

    weight = len(train_y_mean_cls) / np.sum(train_y_mean_cls) - 1  # len(negative case) / len(positive case)

    try:
        xgb_mean_cls_model = xgb.XGBClassifier(
            eta=0.001,
            device='cuda',
            objective='binary:logistic',
            early_stopping_rounds=30,
            max_bin=512,
            n_estimators=300,
            scale_pos_weight=weight,
            eval_metric=['error', 'auc'])
        xgb_mean_cls_model.fit(train_x, train_y_mean_cls, eval_set=[(valid_x, valid_y_mean_cls)])
        valid_pred_mean_cls = xgb_mean_cls_model.predict_proba(valid_x)[:, 1]
        param["ret_cls_auc"] = metrics.roc_auc_score(valid_y_mean_cls, valid_pred_mean_cls)
    except:
        param["ret_cls_auc"] = -1

    param["date_begin"] = param["date_begin"].date()
    param["val_split_point"] = param["val_split_point"].date()
    param["date_end"] = param["date_end"].date()

    placeholders = ', '.join(['?'] * len(param))
    columns = ', '.join(param.keys())
    sql = 'INSERT INTO finstat_experiment_log({}) VALUES ({})'.format(columns, placeholders)

    cursor = conn.cursor()
    cursor.execute(sql, tuple(param.values()))
    conn.commit()

    print("Successfully logged the data.")

cursor.close()
conn.close()