"""信号生成模块 - 概率拐点检测→买卖信号"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
with open(BASE_DIR / 'config.json', 'r', encoding='utf-8') as f:
    CONFIG = json.load(f)


def generate_signals(prob_series, dates=None, buy_threshold=None, sell_threshold=None,
                     consecutive_days=None, min_holding_days=None):
    """从概率序列生成买卖信号
    
    Args:
        prob_series: 主升浪概率序列
        dates: 对应日期
        buy_threshold: 买入阈值
        sell_threshold: 卖出阈值
        consecutive_days: 连续确认天数
        min_holding_days: 最小持仓天数
    
    Returns:
        DataFrame with columns: date, prob, signal (1=buy, -1=sell, 0=hold)
    """
    if buy_threshold is None:
        buy_threshold = CONFIG['signal']['buy_threshold']
    if sell_threshold is None:
        sell_threshold = CONFIG['signal']['sell_threshold']
    if consecutive_days is None:
        consecutive_days = CONFIG['signal']['consecutive_days']
    if min_holding_days is None:
        min_holding_days = CONFIG['signal']['min_holding_days']
    
    n = len(prob_series)
    signals = np.zeros(n)
    
    holding = False
    hold_days = 0
    buy_idx = -1
    
    for i in range(n):
        prob = prob_series[i]
        
        if not holding:
            # 买入条件：概率连续N天超过阈值
            if i >= consecutive_days - 1:
                if all(prob_series[j] >= buy_threshold for j in range(i - consecutive_days + 1, i + 1)):
                    signals[i] = 1
                    holding = True
                    hold_days = 0
                    buy_idx = i
        else:
            hold_days += 1
            # 卖出条件：持仓超过最小天数 + 概率低于卖出阈值
            if hold_days >= min_holding_days and prob <= sell_threshold:
                signals[i] = -1
                holding = False
                hold_days = 0
    
    result = pd.DataFrame({'prob': prob_series, 'signal': signals})
    if dates is not None:
        result['date'] = dates
    
    return result


def predict_stock(model, price_df, feature_names=None):
    """对单只股票生成主升浪概率预测
    
    Args:
        model: LightGBM模型
        price_df: 日线数据
        feature_names: 训练时使用的特征名
    
    Returns:
        DataFrame with date, prob, signal
    """
    from factor_engine import calc_all_factors
    
    if price_df is None or len(price_df) < 120:
        return pd.DataFrame()
    
    factors = calc_all_factors(price_df)
    if factors.empty:
        return pd.DataFrame()
    
    # 对齐特征
    if feature_names is not None:
        for col in feature_names:
            if col not in factors.columns:
                factors[col] = np.nan
        factors = factors[feature_names]
    
    # 去除非数值列
    factors = factors.select_dtypes(include=[np.number])
    
    # 预测
    probs = model.predict(factors, num_iteration=model.best_iteration)
    
    result = pd.DataFrame({
        'date': price_df['date'].values,
        'prob': probs
    })
    
    # 生成信号
    signals_df = generate_signals(probs, dates=price_df['date'].values)
    result['signal'] = signals_df['signal'].values
    
    return result
