"""全市场扫描脚本 - 扫描沪深300+中证500，选出主升浪股票"""
import sys
import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from data_loader import get_stock_data
from factor_engine import calc_all_factors
from model_trainer import load_latest_model
from signal_generator import predict_stock


def get_stock_name(code):
    """获取股票名称，从本地CSV读取"""
    csv_file = BASE_DIR / 'data' / 'full_a_stocks.csv'
    if not hasattr(get_stock_name, '_name_map'):
        get_stock_name._name_map = {}
        if csv_file.exists():
            try:
                import pandas as pd
                df = pd.read_csv(csv_file, encoding='utf-8-sig', dtype={'code': str})
                df['code'] = df['code'].astype(str).str.zfill(6)
                get_stock_name._name_map = dict(zip(df['code'].tolist(), df['name'].tolist()))
            except Exception:
                pass
    return get_stock_name._name_map.get(str(code).zfill(6), '')


def scan_stocks(stock_codes=None, model=None, meta=None, exclude_codes=None, 
                top_n=30, upcoming_n=15, cache_dir=None):
    """扫描股票，按主升浪概率排名
    
    Args:
        stock_codes: 待扫描股票代码列表
        model: LightGBM模型
        meta: 模型meta信息(含feature_names)
        exclude_codes: 需排除的股票代码(如已标注的)
        top_n: 返回TOP N主升浪概率最高的
        upcoming_n: 返回即将启动的股票数
        cache_dir: 扫描缓存目录
    
    Returns:
        dict with top_uptrend, upcoming
    """
    if exclude_codes is None:
        exclude_codes = set()
    
    if cache_dir is None:
        cache_dir = BASE_DIR / 'data' / 'scan_cache'
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    if model is None:
        model, meta = load_latest_model()
        if model is None:
            print("未找到训练模型，请先训练！")
            return {}
    
    feature_names = meta.get('top_features', []) if meta else []
    feature_names = [f[0] if isinstance(f, (list, tuple)) else f for f in feature_names]
    
    # 获取模型使用的全部特征名
    model_feature_names = model.feature_name()
    
    results = []
    total = len(stock_codes)
    
    for idx, code in enumerate(stock_codes):
        code = str(code).strip()
        if code in exclude_codes:
            continue
        
        # 进度
        if (idx + 1) % 50 == 0:
            print(f"扫描进度: {idx+1}/{total}", flush=True)
        
        # 获取数据(用缓存)
        price_df = get_stock_data(code, cache_dir=cache_dir)
        if price_df.empty or len(price_df) < 120:
            continue
        
        # 计算因子和预测
        try:
            pred_df = predict_stock(model, price_df, feature_names=model_feature_names)
            if pred_df.empty:
                continue
            
            # 最近5日平均概率
            recent = pred_df.tail(5)
            avg_prob = recent['prob'].mean()
            latest_prob = pred_df.iloc[-1]['prob']
            
            # 概率趋势(5日变化)
            if len(pred_df) >= 10:
                prob_5d_ago = pred_df.iloc[-6]['prob']
                prob_trend = latest_prob - prob_5d_ago
            else:
                prob_trend = 0
            
            # 当前价格
            latest_price = price_df.iloc[-1]['close']
            
            # 20日涨幅
            if len(price_df) >= 20:
                ret_20d = (price_df.iloc[-1]['close'] / price_df.iloc[-20]['close'] - 1)
            else:
                ret_20d = 0
            
            results.append({
                'code': code,
                'avg_prob_5d': round(float(avg_prob), 4),
                'latest_prob': round(float(latest_prob), 4),
                'prob_trend_5d': round(float(prob_trend), 4),
                'latest_price': round(float(latest_price), 2),
                'ret_20d': round(float(ret_20d), 4)
            })
        except Exception as e:
            continue
    
    if not results:
        return {}
    
    results_df = pd.DataFrame(results)
    
    # 获取股票名称
    print(f"\n获取股票名称中...", flush=True)
    for i, row in results_df.iterrows():
        name = get_stock_name(row['code'])
        results_df.loc[i, 'name'] = name
        time.sleep(0.3)  # 避免请求过快
    
    # TOP N 主升浪概率最高
    top_uptrend = results_df.nlargest(top_n, 'latest_prob')
    
    # 即将启动：概率在0.40-0.55之间且趋势上升
    upcoming_mask = (results_df['latest_prob'] >= 0.40) & (results_df['latest_prob'] < 0.55) & (results_df['prob_trend_5d'] > 0)
    upcoming = results_df[upcoming_mask].nlargest(upcoming_n, 'prob_trend_5d')
    
    return {
        'top_uptrend': top_uptrend.to_dict('records'),
        'upcoming': upcoming.to_dict('records'),
        'total_scanned': len(results)
    }


if __name__ == '__main__':
    print("=" * 60)
    print("ML主升浪策略 - 全市场扫描")
    print("=" * 60)
    
    # 加载标注文件，获取需排除的代码
    annotations_file = BASE_DIR / 'data' / 'annotations.csv'
    exclude_codes = set()
    if annotations_file.exists():
        ann = pd.read_csv(annotations_file, encoding='utf-8-sig')
        exclude_codes = set(str(c).strip() for c in ann['stock_code'])
    
    # 从本地股票池文件读取
    pool_file = BASE_DIR / 'data' / 'stock_pool.txt'
    if pool_file.exists():
        with open(pool_file, 'r', encoding='utf-8') as f:
            all_codes = [line.strip() for line in f if line.strip()]
        print(f"从本地股票池读取: {len(all_codes)} 只")
    else:
        # 在线获取
        print("获取沪深300成分股...")
        import akshare as ak
        csi300_df = ak.index_stock_cons(symbol='000300')
        csi300 = csi300_df.iloc[:, 0].tolist()
        print(f"沪深300: {len(csi300)} 只")
        
        print("获取中证500成分股...")
        csi500_df = ak.index_stock_cons(symbol='000905')
        csi500 = csi500_df.iloc[:, 0].tolist()
        print(f"中证500: {len(csi500)} 只")
        
        all_codes = list(set(csi300 + csi500))
        
        # 保存到本地
        with open(pool_file, 'w', encoding='utf-8') as f:
            for c in sorted(all_codes):
                f.write(c + '\n')
    
    print(f"合计: {len(all_codes)} 只, 排除已标注 {len(exclude_codes)} 只")
    
    # 扫描
    results = scan_stocks(
        stock_codes=all_codes,
        exclude_codes=exclude_codes,
        top_n=30,
        upcoming_n=15
    )
    
    if not results:
        print("扫描失败！")
        sys.exit(1)
    
    # 保存结果
    signals_dir = BASE_DIR / 'signals'
    signals_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON
    with open(signals_dir / 'scan_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # CSV
    top_df = pd.DataFrame(results['top_uptrend'])
    up_df = pd.DataFrame(results.get('upcoming', []))
    
    all_results = pd.concat([top_df, up_df], ignore_index=True)
    all_results.to_csv(signals_dir / 'scan_results.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n扫描完成！共扫描 {results['total_scanned']} 只股票")
    print(f"结果已保存至 signals/scan_results.json 和 signals/scan_results.csv")
