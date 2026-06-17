"""模型训练模块 - LightGBM + 时间序列切分"""
import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
with open(BASE_DIR / 'config.json', 'r', encoding='utf-8') as f:
    CONFIG = json.load(f)


def train_model(X, y, feature_names=None, model_dir=None):
    """训练LightGBM模型，使用时间序列切分
    
    Args:
        X: 特征矩阵
        y: 标签
        feature_names: 特征名列表
        model_dir: 模型保存目录
    
    Returns:
        model, eval_results
    """
    if model_dir is None:
        model_dir = BASE_DIR / 'models'
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    params = CONFIG['model']['params'].copy()
    train_ratio = CONFIG['model']['train_ratio']
    early_stopping_rounds = CONFIG['model']['early_stopping_rounds']
    num_boost_round = CONFIG['model']['num_boost_round']
    
    # 时间序列切分：按时间顺序，前70%训练，后30%测试
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    # y可能是ndarray(float32优化)或Series，统一处理
    if isinstance(y, np.ndarray):
        y_train, y_test = y[:split_idx], y[split_idx:]
    else:
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"训练集: {len(X_train)} 样本, 正样本率: {y_train.mean():.1%}")
    print(f"测试集: {len(X_test)} 样本, 正样本率: {y_test.mean():.1%}")
    
    # 创建LightGBM数据集
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names or X.columns.tolist())
    test_data = lgb.Dataset(X_test, label=y_test, feature_name=feature_names or X.columns.tolist(), reference=train_data)
    
    # 训练
    eval_results = {}
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'test'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds),
            lgb.log_evaluation(period=50),
            lgb.record_evaluation(eval_results)
        ]
    )
    
    # 评估
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    print(f"\n=== 模型评估 ===")
    print(f"AUC: {auc:.4f}")
    
    # 不同阈值的precision/recall
    for threshold in [0.40, 0.45, 0.60]:
        y_pred_label = (y_pred >= threshold).astype(int)
        labels_in_data = sorted(np.unique(y_test))
        target_names_map = {0: 'normal', 1: 'uptrend'}
        target_names = [target_names_map[l] for l in labels_in_data]
        
        if len(labels_in_data) > 1:
            cm = confusion_matrix(y_test, y_pred_label, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            print(f"  threshold={threshold:.2f}: precision={precision:.1%}, recall={recall:.1%}")
        else:
            print(f"  threshold={threshold:.2f}: 测试集只有1个类别，跳过")
    
    # 分类报告(默认阈值0.5)
    y_pred_label = (y_pred >= 0.5).astype(int)
    labels_in_data = sorted(np.unique(y_test))
    target_names_map = {0: 'normal', 1: 'uptrend'}
    target_names = [target_names_map[l] for l in labels_in_data]
    print(f"\n分类报告(threshold=0.5):")
    print(classification_report(y_test, y_pred_label, labels=labels_in_data, target_names=target_names, zero_division=0))
    
    # Top因子重要性
    importance = model.feature_importance(importance_type='gain')
    feat_imp = sorted(zip(feature_names or X.columns.tolist(), importance), key=lambda x: -x[1])
    print("\nTop 20 因子(gain):")
    for name, imp in feat_imp[:20]:
        print(f"  {name}: {imp:.0f}")
    
    # 保存模型
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    model_file = model_dir / f'uptrend_model_{timestamp}.txt'
    model.save_model(str(model_file))
    
    meta = {
        'timestamp': timestamp,
        'auc': float(auc),
        'n_features': len(feature_names) if feature_names else X.shape[1],
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'pos_rate_train': float(y_train.mean()),
        'pos_rate_test': float(y_test.mean()),
        'best_iteration': model.best_iteration,
        'top_features': [(name, int(imp)) for name, imp in feat_imp[:20]]
    }
    meta_file = model_dir / f'uptrend_model_{timestamp}_meta.json'
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    print(f"\n模型已保存: {model_file}")
    
    return model, eval_results, feat_imp


def load_latest_model(model_dir=None):
    """加载最新的训练模型"""
    if model_dir is None:
        model_dir = BASE_DIR / 'models'
    model_dir = Path(model_dir)
    
    model_files = sorted(model_dir.glob('uptrend_model_*.txt'))
    if not model_files:
        return None, None
    
    latest = model_files[-1]
    model = lgb.Booster(model_file=str(latest))
    
    # 加载meta
    meta_file = latest.with_name(latest.stem + '_meta.json')
    meta = None
    if meta_file.exists():
        with open(meta_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    
    return model, meta
