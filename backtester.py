"""简易回测模块"""
import numpy as np
import pandas as pd


def backtest(price_df, signals_df, initial_capital=100000):
    """简单回测
    
    Args:
        price_df: 日线数据(date, close)
        signals_df: 信号数据(date, prob, signal)
        initial_capital: 初始资金
    
    Returns:
        dict with backtest metrics
    """
    merged = price_df[['date', 'close']].merge(signals_df[['date', 'signal']], on='date', how='inner')
    
    if len(merged) < 10:
        return {'error': 'insufficient data'}
    
    capital = initial_capital
    shares = 0
    trades = []
    portfolio_values = []
    
    for i, row in merged.iterrows():
        if row['signal'] == 1 and shares == 0:
            # 买入
            shares = capital / row['close']
            trades.append({'date': row['date'], 'action': 'buy', 'price': row['close']})
            capital = 0
        elif row['signal'] == -1 and shares > 0:
            # 卖出
            capital = shares * row['close']
            trades.append({'date': row['date'], 'action': 'sell', 'price': row['close']})
            shares = 0
        
        portfolio = capital + shares * row['close']
        portfolio_values.append(portfolio)
    
    # 最终市值
    final_value = capital + shares * merged.iloc[-1]['close']
    total_return = (final_value - initial_capital) / initial_capital
    
    # 计算年化
    days = (merged.iloc[-1]['date'] - merged.iloc[0]['date']).days
    annual_return = (1 + total_return) ** (365 / max(days, 1)) - 1 if total_return > -1 else -1
    
    # 最大回撤
    pv_series = pd.Series(portfolio_values)
    running_max = pv_series.cummax()
    drawdown = (pv_series - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'final_value': final_value,
        'n_trades': len(trades),
        'trades': trades
    }
