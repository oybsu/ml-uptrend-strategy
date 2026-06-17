"""主入口 - ML主升浪策略 train/predict/full"""
import sys
import os
import warnings
import time
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from data_loader import get_stock_data, load_annotations
from label_generator import create_labeled_dataset
from factor_engine import calc_all_factors, prepare_features
from model_trainer import train_model, load_latest_model
from signal_generator import predict_stock


def run_train():
    """训练流程: 加载数据→生成标签→计算因子→训练模型"""
    print("=" * 60)
    print("Step 1: 加载标注数据")
    print("=" * 60)
    
    # 复制标注文件到项目目录
    src = Path(__file__).parent.parent / '.opencode-router' / 'media' / 'inbound' / '2026-06-16' / 'weixin' / 'ws_e7d8d07574e4' / 'o9cq80-XplOLxrjXfNEN8nqvsgyU-im.wechat' / 'annotations.csv'
    dst = BASE_DIR / 'data' / 'annotations.csv'
    if src.exists() and not dst.exists():
        import shutil
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"标注文件已复制到 {dst}")
    
    annotations = load_annotations()
    print(f"标注: {len(annotations)} 只股票")
    
    # 下载数据
    print("\n" + "=" * 60)
    print("Step 2: 下载股价数据")
    print("=" * 60)
    
    price_dict = {}
    failed = []
    for idx, row in annotations.iterrows():
        code = str(row['stock_code']).strip()
        name = str(row.get('note', '')).strip()
        print(f"  [{idx+1}/{len(annotations)}] {code} {name}...", end=' ', flush=True)
        
        df = get_stock_data(code)
        if df.empty:
            failed.append(code)
            print("FAIL")
        else:
            price_dict[code] = df
            print(f"OK ({len(df)} bars)")
    
    print(f"成功: {len(price_dict)}, 失败: {len(failed)}")
    
    # 生成标签
    print("\n" + "=" * 60)
    print("Step 3: 生成训练标签")
    print("=" * 60)
    
    labeled_df = create_labeled_dataset(annotations, price_dict)
    if labeled_df.empty:
        print("标签生成失败！")
        return
    
    print(f"总样本: {len(labeled_df)}, 正样本率: {labeled_df['label'].mean():.1%}")
    
    # 计算因子
    print("\n" + "=" * 60)
    print("Step 4: 计算因子")
    print("=" * 60)
    
    all_with_factors = []
    for code, price_df in price_dict.items():
        factors = calc_all_factors(price_df)
        if factors.empty:
            continue
        
        merged = price_df.copy()
        for col in factors.columns:
            merged[col] = factors[col].values
        
        # 标签
        ann_row = annotations[annotations['stock_code'].astype(str) == code]
        if ann_row.empty:
            continue
        
        from label_generator import generate_labels
        labels = generate_labels(price_df, str(ann_row.iloc[0]['start_date']), str(ann_row.iloc[0]['end_date']))
        merged['label'] = labels['label'].values
        merged['stock_code'] = code
        merged['stock_name'] = str(ann_row.iloc[0].get('note', ''))
        
        all_with_factors.append(merged)
    
    if not all_with_factors:
        print("因子计算失败！")
        return
    
    full_df = pd.concat(all_with_factors, ignore_index=True)
    full_df = full_df.dropna(subset=['label'])
    
    print(f"含因子的样本: {len(full_df)}")
    
    # 准备特征
    X, y, feature_names = prepare_features(full_df)
    print(f"特征数: {len(feature_names)}, 有效样本: {len(X)}")
    
    # 填充NaN
    X = X.fillna(0)
    
    # 训练
    print("\n" + "=" * 60)
    print("Step 5: 训练LightGBM模型")
    print("=" * 60)
    
    model, eval_results, feat_imp = train_model(X, y, feature_names=feature_names)
    
    print("\n训练完成！")


def run_predict():
    """预测流程: 加载模型→扫描市场→输出结果"""
    print("=" * 60)
    print("ML主升浪策略 - 市场扫描")
    print("=" * 60)
    
    # 直接调用扫描脚本
    from scan_stocks import scan_stocks
    import scan_stocks as ss
    
    # 用scan_stocks的main逻辑
    ss.main()


def run_full():
    """完整流程: 训练→扫描"""
    run_train()
    print("\n\n")
    run_predict()


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'train'
    
    if mode == 'train':
        run_train()
    elif mode == 'predict':
        run_predict()
    elif mode == 'full':
        run_full()
    else:
        print(f"用法: python main.py [train|predict|full]")
        print(f"  train   - 训练模型")
        print(f"  predict - 扫描市场")
        print(f"  full    - 训练+扫描")
