"""因子引擎 - 123个技术因子(6大类): 趋势/动量/波动/量价/形态/统计"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def calc_ma_factors(df):
    """趋势类因子: 均线系统、价格偏离、趋势斜率"""
    close = df['close']
    factors = pd.DataFrame(index=df.index)
    
    for w in [5, 10, 20, 30, 60, 120]:
        ma = close.rolling(w).mean()
        factors[f'ma{w}'] = ma
        factors[f'close_to_ma{w}'] = close / ma - 1  # 价格偏离均线
        factors[f'pct_above_ma{w}'] = (close > ma).astype(int)  # 是否在均线之上
        # 均线斜率(趋势方向)
        factors[f'ma{w}_slope'] = ma.pct_change(5)
    
    # 均线排列(多头/空头)
    if all(f'ma{w}' in factors.columns for w in [5, 20, 60]):
        factors['ma_bull_align'] = ((factors['ma5'] > factors['ma20']) & 
                                     (factors['ma20'] > factors['ma60'])).astype(int)
        factors['ma_bear_align'] = ((factors['ma5'] < factors['ma20']) & 
                                     (factors['ma20'] < factors['ma60'])).astype(int)
    
    # 均线发散度
    if 'ma5' in factors.columns and 'ma60' in factors.columns:
        factors['ma_spread'] = factors['ma5'] / factors['ma60'] - 1
    
    return factors


def calc_momentum_factors(df):
    """动量类因子: 收益率、RSI、MACD"""
    close = df['close']
    factors = pd.DataFrame(index=df.index)
    
    # 不同周期收益率
    for w in [1, 3, 5, 10, 20, 30, 60]:
        factors[f'ret_{w}d'] = close.pct_change(w)
    
    # RSI
    for w in [6, 14, 24]:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(w).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(w).mean()
        rs = gain / loss.replace(0, np.nan)
        factors[f'rsi_{w}'] = 100 - 100 / (1 + rs)
    
    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9).mean()
    factors['macd_dif'] = dif
    factors['macd_dea'] = dea
    factors['macd_signal'] = dif - dea  # 柱状线
    factors['macd_cross'] = ((dif > dea) & (dif.shift(1) <= dea.shift(1))).astype(int)  # 金叉
    factors['macd_death_cross'] = ((dif < dea) & (dif.shift(1) >= dea.shift(1))).astype(int)  # 死叉
    
    # 动量加速度
    factors['momentum_accel'] = close.pct_change(5).diff(5)
    
    return factors


def calc_volatility_factors(df):
    """波动类因子: ATR、布林带、历史波动率"""
    close = df['close']
    high = df['high']
    low = df['low']
    factors = pd.DataFrame(index=df.index)
    
    # 真实波幅 / ATR
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    for w in [5, 10, 14, 20]:
        factors[f'atr_{w}'] = tr.rolling(w).mean()
    factors['atr_pct_14'] = factors['atr_14'] / close  # ATR占价格比
    
    # 布林带
    for w in [20, 30]:
        ma = close.rolling(w).mean()
        std = close.rolling(w).std()
        factors[f'bb_upper_{w}'] = ma + 2 * std
        factors[f'bb_lower_{w}'] = ma - 2 * std
        factors[f'bb_width_{w}'] = 4 * std / ma
        factors[f'bb_pos_{w}'] = (close - ma + 2 * std) / (4 * std)  # 布林带位置
    
    # 历史波动率
    ret = close.pct_change()
    for w in [5, 10, 20, 60]:
        factors[f'volatility_{w}'] = ret.rolling(w).std() * np.sqrt(252)
    
    # 波动率变化
    factors['vol_change'] = factors['volatility_20'].pct_change(5)
    
    return factors


def calc_volume_price_factors(df):
    """量价类因子: OBV、量比、换手率相关性"""
    close = df['close']
    volume = df['volume']
    factors = pd.DataFrame(index=df.index)
    
    # OBV
    obv = (np.sign(close.diff()) * volume).cumsum()
    factors['obv'] = obv
    factors['obv_ma10'] = obv.rolling(10).mean()
    factors['obv_ma20'] = obv.rolling(20).mean()
    factors['obv_diverge'] = obv / obv.rolling(20).mean() - 1
    
    # 量比
    vol_ma5 = volume.rolling(5).mean()
    vol_ma20 = volume.rolling(20).mean()
    factors['vol_ratio_5'] = volume / vol_ma5.replace(0, np.nan)
    factors['vol_ratio_20'] = volume / vol_ma20.replace(0, np.nan)
    
    # 量价相关性
    for w in [10, 20]:
        factors[f'vol_price_corr_{w}'] = close.pct_change().rolling(w).corr(volume.pct_change())
    
    # 量增价涨/量缩价跌
    vol_up = (volume > volume.shift(1)).astype(int)
    price_up = (close > close.shift(1)).astype(int)
    factors['vol_up_price_up'] = (vol_up & price_up).astype(int)
    factors['vol_down_price_down'] = ((~vol_up.astype(bool)) & (~price_up.astype(bool))).astype(int)
    
    # 成交额趋势
    amount = df['amount']
    factors['amount_ma5'] = amount.rolling(5).mean()
    factors['amount_ma20'] = amount.rolling(20).mean()
    factors['amount_ratio'] = factors['amount_ma5'] / factors['amount_ma20'].replace(0, np.nan)
    
    return factors


def calc_pattern_factors(df):
    """形态类因子: 缺口、K线形态、新高新低"""
    close = df['close']
    high = df['high']
    low = df['low']
    open_ = df['open']
    factors = pd.DataFrame(index=df.index)
    
    # 缺口
    factors['gap_up'] = (open_ > high.shift(1)).astype(int)
    factors['gap_down'] = (open_ < low.shift(1)).astype(int)
    
    # K线实体和影线
    body = (close - open_).abs()
    total_range = high - low
    factors['body_ratio'] = body / total_range.replace(0, np.nan)  # 实体占比
    factors['upper_shadow'] = (high - pd.concat([close, open_], axis=1).max(axis=1)) / total_range.replace(0, np.nan)
    factors['lower_shadow'] = (pd.concat([close, open_], axis=1).min(axis=1) - low) / total_range.replace(0, np.nan)
    
    # 新高/新低
    for w in [20, 60, 120]:
        factors[f'nhigh_{w}'] = (close >= high.rolling(w).max()).astype(int)
        factors[f'nlow_{w}'] = (close <= low.rolling(w).min()).astype(int)
    
    # 连涨/连跌天数
    up = (close > close.shift(1)).astype(int)
    down = (close < close.shift(1)).astype(int)
    factors['consec_up'] = up * (up.groupby((up != up.shift()).cumsum()).cumcount() + 1)
    factors['consec_down'] = down * (down.groupby((down != down.shift()).cumsum()).cumcount() + 1)
    
    # 阳线/阴线比例
    for w in [5, 10, 20]:
        factors[f'up_ratio_{w}'] = up.rolling(w).mean()
    
    return factors


def calc_statistics_factors(df):
    """统计类因子: 偏度、峰度、分位数、自相关"""
    close = df['close']
    ret = close.pct_change()
    factors = pd.DataFrame(index=df.index)
    
    # 偏度和峰度
    for w in [20, 60]:
        factors[f'skewness_{w}'] = ret.rolling(w).skew()
        factors[f'kurtosis_{w}'] = ret.rolling(w).kurt()
    
    # 收益分位数
    for w in [20, 60]:
        factors[f'ret_quantile_{w}'] = ret.rolling(w).rank(pct=True)
    
    # 自相关
    for w in [20, 60]:
        factors[f'autocorr_{w}'] = ret.rolling(w).apply(lambda x: x.autocorr() if len(x) > 2 else 0, raw=False)
    
    # 极值比
    for w in [20, 60]:
        factors[f'max_ret_{w}'] = ret.rolling(w).max()
        factors[f'min_ret_{w}'] = ret.rolling(w).min()
        factors[f'range_ret_{w}'] = factors[f'max_ret_{w}'] - factors[f'min_ret_{w}']
    
    # 价格位置(相对区间高低点)
    for w in [20, 60, 120]:
        high_w = df['high'].rolling(w).max()
        low_w = df['low'].rolling(w).min()
        range_w = high_w - low_w
        factors[f'price_position_{w}'] = (close - low_w) / range_w.replace(0, np.nan)
    
    return factors


def calc_all_factors(df):
    """计算全部123个因子
    
    Args:
        df: 日线数据DataFrame，必须包含 open/high/low/close/volume/amount
    
    Returns:
        DataFrame with all factors as columns
    """
    if df is None or len(df) < 120:
        return pd.DataFrame()
    
    df = df.copy()
    
    # 按类别计算
    trend = calc_ma_factors(df)
    momentum = calc_momentum_factors(df)
    volatility = calc_volatility_factors(df)
    volume_price = calc_volume_price_factors(df)
    pattern = calc_pattern_factors(df)
    statistics = calc_statistics_factors(df)
    
    # 合并
    all_factors = pd.concat([trend, momentum, volatility, volume_price, pattern, statistics], axis=1)
    
    # 去除无穷大和极端值
    all_factors = all_factors.replace([np.inf, -np.inf], np.nan)
    
    # Windsorize: 将超过3倍标准差的值截断
    for col in all_factors.columns:
        s = all_factors[col]
        if s.std() > 0:
            mean = s.mean()
            std = s.std()
            all_factors[col] = s.clip(mean - 3 * std, mean + 3 * std)
    
    return all_factors


def prepare_features(labeled_df):
    """从标注数据准备特征矩阵和标签
    
    优化内存: float32 + 分块处理，避免大copy
    
    Args:
        labeled_df: 带标签的数据
    
    Returns:
        X, y, feature_names
    """
    exclude_cols = {'date', 'label', 'label_desc', 'stock_code', 'stock_name',
                    'open', 'high', 'low', 'close', 'volume', 'amount'}
    feature_cols = [c for c in labeled_df.columns if c not in exclude_cols]
    
    # 只取数值列，转float32节省内存
    X = labeled_df[feature_cols].select_dtypes(include=[np.number])
    # 逐列转float32，避免整表copy
    for c in X.columns:
        if X[c].dtype == np.float64:
            X[c] = X[c].astype(np.float32)
    
    # 去除全NaN列
    X = X.dropna(axis=1, how='all')
    
    # 去除方差为0的列
    var = X.var()
    zero_var_cols = var[var == 0].index.tolist()
    if zero_var_cols:
        X = X.drop(columns=zero_var_cols)
    
    # 去除高相关性列(>0.98)减少冗余
    # 采样计算相关矩阵(避免全量计算爆内存)
    if len(X.columns) > 50 and len(X) > 10000:
        sample_X = X.sample(n=min(10000, len(X)), random_state=42)
        corr_matrix = sample_X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_cols = [col for col in upper.columns if any(upper[col] > 0.98)]
        if high_corr_cols:
            X = X.drop(columns=high_corr_cols)
    
    feature_names = X.columns.tolist()
    y = labeled_df['label'].values.astype(np.float32)
    
    return X, y, feature_names
