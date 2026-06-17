"""标签生成模块 - 根据主升浪标注生成二分类训练标签"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

BASE_DIR = Path(__file__).parent
with open(BASE_DIR / 'config.json', 'r', encoding='utf-8') as f:
    CONFIG = json.load(f)


def generate_labels(price_df, uptrend_start, uptrend_end, pre_days=None, mode=None):
    """为单只股票生成训练标签
    
    Args:
        price_df: 日线数据，必须包含date列
        uptrend_start: 主升浪开始日期 str YYYY-MM-DD
        uptrend_end: 主升浪结束日期 str YYYY-MM-DD
        pre_days: 主升浪前多少天标记为1（默认从config读取）
        mode: 'binary' or 'three_class'（默认从config读取）
    
    Returns:
        DataFrame with columns: date, label, label_desc
    """
    if pre_days is None:
        pre_days = CONFIG['label']['pre_days']
    if mode is None:
        mode = CONFIG['label']['mode']
    
    if price_df.empty:
        return pd.DataFrame()
    
    df = price_df[['date']].copy()
    df['label'] = 0
    df['label_desc'] = 'normal'
    
    start_ts = pd.Timestamp(uptrend_start)
    end_ts = pd.Timestamp(uptrend_end)
    
    # 主升浪期间标记为1
    mask_uptrend = (df['date'] >= start_ts) & (df['date'] <= end_ts)
    df.loc[mask_uptrend, 'label'] = 1
    df.loc[mask_uptrend, 'label_desc'] = 'uptrend'
    
    # 主升浪前pre_days天标记为1（这是我们要预测的关键时期）
    pre_start_idx = None
    for i, row in df.iterrows():
        if row['date'] >= start_ts:
            pre_start_idx = i
            break
    
    if pre_start_idx is not None:
        pre_begin = max(0, pre_start_idx - pre_days)
        for i in range(pre_begin, pre_start_idx):
            df.loc[i, 'label'] = 1
            df.loc[i, 'label_desc'] = 'pre_uptrend'
    
    if mode == 'three_class':
        # 三分类：0=normal, 1=pre_uptrend, 2=uptrend
        df.loc[df['label_desc'] == 'pre_uptrend', 'label'] = 1
        df.loc[df['label_desc'] == 'uptrend', 'label'] = 2
    
    return df


def create_labeled_dataset(annotations_df, price_dict):
    """基于标注和价格数据创建完整训练集
    
    Args:
        annotations_df: 标注DataFrame (stock_code, start_date, end_date, note)
        price_dict: {stock_code: price_df}
    
    Returns:
        合并后的DataFrame
    """
    all_labeled = []
    
    for _, row in annotations_df.iterrows():
        code = str(row['stock_code']).strip()
        start_date = str(row['start_date']).strip()
        end_date = str(row['end_date']).strip()
        note = str(row.get('note', '')).strip()
        
        if code not in price_dict or price_dict[code].empty:
            continue
        
        price_df = price_dict[code]
        labels_df = generate_labels(price_df, start_date, end_date)
        
        if labels_df.empty:
            continue
        
        merged = price_df.merge(labels_df, on='date', how='left')
        merged['label'] = merged['label'].fillna(0).astype(int)
        merged['label_desc'] = merged['label_desc'].fillna('normal')
        merged['stock_code'] = code
        merged['stock_name'] = note
        
        all_labeled.append(merged)
    
    if not all_labeled:
        return pd.DataFrame()
    
    result = pd.concat(all_labeled, ignore_index=True)
    return result
