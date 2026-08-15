import pandas as pd
import numpy as np
import akshare as ak
import cvxpy as cp
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from sqlalchemy import create_engine
from scipy import stats, optimize, interpolate
from scipy.stats import skew, kurtosis, pearsonr, spearmanr, norm
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ①数据准备和数据清洗
def load_data():
   stocks = ["000001", "000002", "600016", "601288", "601398", "601318",
        "000568", "000858", "603288", "600887", "600809",
         "600436", "603259", "000661", "300122",
        "300750", "002594", "601012", "600438", "002415", "300059",
        "600309", "601888", "600019",  "002460",
        "002371", "603501",  "600584"]
   all_data = []
   
   for code in stocks:
       df = pd.read_csv(f"{code}.csv")
       '''df = ak.stock_zh_a_hist(symbol=code, start_date="20210101", 
                              end_date="20221231", adjust="qfq")'''
       # 列索引为原始中文列名['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']

       df['日期'] = pd.to_datetime(df['日期'])
       df['stock'] = code
       all_data.append(df)   #32个DataFrame

   return pd.concat(all_data).sort_values(['stock', '日期'])
def clean_data(df):
    df = df.drop_duplicates(subset=['stock', '日期'])
    numeric_cols = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额','换手率']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df = df.groupby('stock').filter(lambda x: len(x) >= 60)
    df = df[df['成交量'] > 0]
    df = df[df['涨跌幅'].abs() <= 20]
    df = df.sort_values(['stock', '日期'])
    price_cols = ['开盘', '收盘', '最高', '最低']
    df[price_cols] = df.groupby('stock')[price_cols].ffill()
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df

df = load_data()
df = clean_data(df)

# ②构造各种因子
df['ret_1d']=df.groupby('stock')['收盘'].pct_change()
df['指数动量']=df.groupby('stock')['ret_1d'].transform(lambda x:x.ewm(halflife=5,adjust=False).mean())

def calc_rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - 100/(1+rs)
    return rsi
df['负RSI'] = -df.groupby('stock')['收盘'].transform(lambda x: calc_rsi(x, window=14))

df['波动率'] = df.groupby('stock')['ret_1d'].transform(lambda x:x.rolling(20).std())

df['负换手率'] = -df['换手率']

vol_ma5 = df.groupby('stock')['成交量'].transform(lambda x:x.rolling(5).mean())
vol_ma20 = df.groupby('stock')['成交量'].transform(lambda x:x.rolling(20).mean())
df['成交量趋势'] = vol_ma5 / vol_ma20 - 1

df['future_20'] = df.groupby('stock')['收盘'].shift(-20) / df['收盘'] - 1
df['future_20'] = df['future_20'].ffill()

# 复合因子
factor_cols = ['指数动量', '负RSI', '波动率','负换手率' ,'成交量趋势']
for f in factor_cols:
    df[f'{f}_z'] = df.groupby('日期')[f].transform(lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0)
# 标准化一天内

z_cols = [f'{f}_z' for f in factor_cols]
ic_records = []
for date, group in df.groupby('日期'):
    group = group.dropna(subset=z_cols + ['future_20'])
    if len(group) < 10:
        continue
    ic_day = {'日期': date}
    for f, z_name in zip(factor_cols, z_cols):
        ic_day[f] = group[z_name].corr(group['future_20'], method='spearman')
    ic_records.append(ic_day)

ic_df = pd.DataFrame(ic_records).set_index('日期').sort_index()

window = 60
composite_list = []
for date, group in df.groupby('日期'):
    past_ic = ic_df[ic_df.index < date].tail(window)
    
    if len(past_ic) < 20:
        w=np.array([1/len(factor_cols)]*len(factor_cols))
    else:
        ic_mean = past_ic.mean().values
        w = ic_mean / np.sum(np.abs(ic_mean))

    z = group[z_cols].values
    comp = z @ w
    group['复合因子'] = comp
    composite_list.append(group[['日期', 'stock', '复合因子']])

comp_df = pd.concat(composite_list)
df = df.merge(comp_df, on=['日期', 'stock'], how='left') # df['复合因子'] = comp_df['复合因子']

# ③因子回测和检验
def backtest_layered(df, factor_col, n_layers=4, hold_days=20):
    df = df.copy()
    df['future'] = df.groupby('stock')['收盘'].shift(-hold_days) / df['收盘'] - 1
    df = df.dropna(subset=['future'])
    df['layer'] = df.groupby('日期')[factor_col].transform(lambda x: pd.qcut(x.rank(method='first'), n_layers, labels=False, duplicates='drop')+1)
    layer_ret = df.groupby(['日期','layer'])['future'].mean().unstack()
    long_short = layer_ret[n_layers] - layer_ret[1]
    nav = (1+long_short/hold_days).cumprod()
    annual_ret = (1+long_short.mean())**(252/hold_days)-1
    sharpe = (long_short.mean()*252/hold_days-0.025)/(long_short.std()*np.sqrt(252/hold_days))
    max_dd = (nav/nav.cummax()-1).min()
    return {
        'nav': nav,
        'annual_ret': annual_ret,
        'sharpe': sharpe,
        'max_dd': max_dd
    }


def calc_ic(df, factor, future_ret):
    ic_results = []
    for date,group in df.groupby('日期'):
        if len(group) >= 5:
            ic, pvalue = spearmanr(group[factor], group[future_ret])
            ic_results.append({'date':date,'IC':ic,'pvalue':pvalue})
    ic_df = pd.DataFrame(ic_results)
    ic_mean = ic_df['IC'].mean()
    ic_std = ic_df['IC'].std()
    ic_ir = ic_mean/ic_std if ic_std != 0 else 0
    pvalue_mean = ic_df['pvalue'].mean()
    return {
    'ic_mean': ic_mean,
    'ic_std': ic_std,
    'ic_ir': ic_ir,
    'pvalue_mean': pvalue_mean
}


# ④主程序绩效评估并可视化
factor_ = ['指数动量', '负RSI', '波动率','负换手率' ,'成交量趋势','复合因子']
all_results = {}
for i in factor_:
    result_test = backtest_layered(df, factor_col=i, n_layers=4, hold_days=20)
    all_results[i] = result_test
    print(result_test)

for i in factor_:
    result_ic = calc_ic(df, i, 'future_20')
    print(result_ic)

nav_df = pd.DataFrame({i: all_results[i]['nav'] for i in factor_})
nav_df.plot(figsize=(12, 6), title='各因子多空净值对比')
plt.show()
