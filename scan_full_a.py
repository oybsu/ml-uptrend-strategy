"""全A市场扫描脚本 - 扫描全部A股，选出主升浪股票

优化策略：
1. 先批量下载数据（qlib离线转换）
2. 再逐只计算因子和预测
3. 已有缓存的跳过下载
"""
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

from data_loader import get_stock_data, qlib_get_stock_list
from factor_engine import calc_all_factors
from model_trainer import load_latest_model
from signal_generator import predict_stock


def get_all_a_stocks():
    """获取全部A股股票代码列表（排除ST、退市、北交所）
    
    从qlib数据获取，fallback到CSV缓存
    """
    # 优先CSV缓存
    csv_file = BASE_DIR / 'data' / 'full_a_stocks.csv'
    if csv_file.exists():
        df = pd.read_csv(csv_file, encoding='utf-8-sig', dtype={'code': str})
        df['code'] = df['code'].astype(str).str.zfill(6)
        name_map = dict(zip(df['code'].tolist(), df['name'].tolist()))
        print(f"从本地CSV读取: {len(name_map)} 只", flush=True)
        return name_map
    
    # 从qlib获取
    print("从qlib获取全A列表...", flush=True)
    name_map = qlib_get_stock_list()
    if name_map:
        df = pd.DataFrame(list(name_map.items()), columns=['code', 'name'])
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"qlib获取成功: {len(name_map)} 只", flush=True)
        return name_map
    
    print("获取全A列表失败！")
    return {}


def scan_single_stock(code, model, model_feature_names, cache_dir):
    """扫描单只股票"""
    try:
        price_df = get_stock_data(code, cache_dir=cache_dir)
        if price_df.empty or len(price_df) < 120:
            return None
        
        pred_df = predict_stock(model, price_df, feature_names=model_feature_names)
        if pred_df.empty:
            return None
        
        recent = pred_df.tail(5)
        avg_prob = recent['prob'].mean()
        latest_prob = pred_df.iloc[-1]['prob']
        
        if len(pred_df) >= 10:
            prob_5d_ago = pred_df.iloc[-6]['prob']
            prob_trend = latest_prob - prob_5d_ago
        else:
            prob_trend = 0
        
        latest_price = price_df.iloc[-1]['close']
        
        if len(price_df) >= 20:
            ret_20d = (price_df.iloc[-1]['close'] / price_df.iloc[-20]['close'] - 1)
        else:
            ret_20d = 0
        
        return {
            'code': code,
            'avg_prob_5d': round(float(avg_prob), 4),
            'latest_prob': round(float(latest_prob), 4),
            'prob_trend_5d': round(float(prob_trend), 4),
            'latest_price': round(float(latest_price), 2),
            'ret_20d': round(float(ret_20d), 4)
        }
    except Exception:
        return None


def scan_stocks_full_a(name_map, model=None, meta=None, exclude_codes=None,
                       top_n=50, upcoming_n=30, cache_dir=None):
    """全A扫描"""
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
    
    model_feature_names = model.feature_name()
    
    stock_codes = [c for c in name_map.keys() if c not in exclude_codes]
    total = len(stock_codes)
    print(f"待扫描: {total} 只", flush=True)
    
    # Step 1: 检查缓存（数据已通过run.py download下载）
    cached_count = len(list(cache_dir.glob('*.parquet')))
    print(f"\n=== 数据已就绪: {cached_count} 个缓存文件 ===", flush=True)
    
    # Step 2: 逐只计算因子和预测
    print("\n=== Step 2: 计算因子和预测 ===", flush=True)
    results = []
    success = 0
    fail = 0
    start_time = time.time()
    
    for idx, code in enumerate(stock_codes):
        if (idx + 1) % 200 == 0:
            elapsed = time.time() - start_time
            speed = (idx + 1) / elapsed
            eta = (total - idx - 1) / speed / 60
            print(f"扫描进度: {idx+1}/{total} | 成功: {success} | 失败: {fail} | {speed:.1f}/s | ETA: {eta:.0f}min", flush=True)
        
        result = scan_single_stock(code, model, model_feature_names, cache_dir)
        if result is not None:
            result['name'] = name_map.get(code, '')
            results.append(result)
            success += 1
        else:
            fail += 1
    
    if not results:
        return {}
    
    results_df = pd.DataFrame(results)
    
    # TOP N 主升浪概率最高
    top_uptrend = results_df.nlargest(top_n, 'latest_prob')
    
    # 即将启动：概率在0.40-0.55之间且趋势上升
    upcoming_mask = (results_df['latest_prob'] >= 0.40) & (results_df['latest_prob'] < 0.55) & (results_df['prob_trend_5d'] > 0)
    upcoming = results_df[upcoming_mask].nlargest(upcoming_n, 'prob_trend_5d')
    
    elapsed = time.time() - start_time
    print(f"\n扫描完成！共 {len(results)} 只成功, {fail} 只失败, 耗时 {elapsed/60:.1f} 分钟", flush=True)
    
    return {
        'top_uptrend': top_uptrend.to_dict('records'),
        'upcoming': upcoming.to_dict('records'),
        'total_scanned': len(results),
        'total_failed': fail
    }


def generate_report(results, output_path):
    """生成HTML报告"""
    top = results.get('top_uptrend', [])
    upcoming = results.get('upcoming', [])
    total_scanned = results.get('total_scanned', 0)
    
    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>ML主升浪策略 - 全A扫描报告</title>
<style>
body { font-family: 'Microsoft YaHei', sans-serif; margin: 20px; background: #f5f5f5; }
h1 { color: #1a5276; border-bottom: 2px solid #2980b9; padding-bottom: 10px; }
h2 { color: #2c3e50; margin-top: 30px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; background: white; font-size: 13px; }
th { background: #2980b9; color: white; padding: 8px 6px; text-align: center; }
td { padding: 7px 6px; text-align: center; border-bottom: 1px solid #ddd; }
tr:nth-child(even) { background: #f9f9f9; }
tr:hover { background: #eaf2f8; }
.up { color: #e74c3c; font-weight: bold; }
.down { color: #27ae60; }
.summary { background: white; padding: 15px; border-radius: 8px; margin: 10px 0; }
.badge { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 11px; margin: 1px; color: white; }
.badge-high { background: #e74c3c; }
.badge-mid { background: #f39c12; }
.badge-low { background: #3498db; }
.badge-strong { background: #8e44ad; }
</style>
</head>
<body>
<h1>ML主升浪策略 - 全A扫描报告</h1>
<div class="summary">
<p><strong>扫描范围:</strong> 全A股（排除ST/退市/北交所） | <strong>成功扫描:</strong> """ + str(total_scanned) + """ 只</p>
<p><strong>模型:</strong> LightGBM | <strong>因子数:</strong> 123 | <strong>训练样本:</strong> 41只真实标注</p>
<p><strong>筛选标准:</strong> TOP50主升浪概率 + 30只即将启动(概率0.40-0.55且趋势上升)</p>
</div>
""")
    
    # TOP主升浪
    html_parts.append("<h2>TOP 50 主升浪概率最高</h2><table><tr><th>排名</th><th>代码</th><th>名称</th><th>概率</th><th>5日趋势</th><th>最新价</th><th>20日涨幅</th><th>信号</th></tr>")
    for i, r in enumerate(top):
        prob = r['latest_prob']
        trend = r['prob_trend_5d']
        ret = r['ret_20d']
        
        signals = []
        if prob >= 0.60:
            signals.append('<span class="badge badge-high">高概率</span>')
        if trend > 0.1:
            signals.append('<span class="badge badge-mid">趋势加速</span>')
        if ret > 0.3:
            signals.append('<span class="badge badge-strong">强势</span>')
        elif ret > 0.15:
            signals.append('<span class="badge badge-low">走强</span>')
        
        trend_cls = 'up' if trend > 0 else 'down'
        ret_cls = 'up' if ret > 0 else 'down'
        
        html_parts.append(f"""<tr>
<td>{i+1}</td>
<td>{r['code']}</td>
<td>{r['name']}</td>
<td><strong>{prob*100:.1f}%</strong></td>
<td class="{trend_cls}">{trend:+.3f}</td>
<td>{r['latest_price']:.2f}</td>
<td class="{ret_cls}">{ret*100:+.1f}%</td>
<td>{' '.join(signals) if signals else '-'}</td>
</tr>""")
    html_parts.append("</table>")
    
    # 即将启动
    html_parts.append("<h2>即将启动 (概率0.40-0.55 + 趋势上升)</h2><table><tr><th>排名</th><th>代码</th><th>名称</th><th>概率</th><th>5日趋势</th><th>最新价</th><th>20日涨幅</th></tr>")
    for i, r in enumerate(upcoming):
        prob = r['latest_prob']
        trend = r['prob_trend_5d']
        ret = r['ret_20d']
        trend_cls = 'up' if trend > 0 else 'down'
        ret_cls = 'up' if ret > 0 else 'down'
        
        html_parts.append(f"""<tr>
<td>{i+1}</td>
<td>{r['code']}</td>
<td>{r['name']}</td>
<td><strong>{prob*100:.1f}%</strong></td>
<td class="{trend_cls}">{trend:+.3f}</td>
<td>{r['latest_price']:.2f}</td>
<td class="{ret_cls}">{ret*100:+.1f}%</td>
</tr>""")
    html_parts.append("</table>")
    
    html_parts.append("<p style='color:#7f8c8d; font-size:12px; margin-top:30px;'>报告生成时间: " + time.strftime('%Y-%m-%d %H:%M:%S') + "</p></body></html>")
    
    html = ''.join(html_parts)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"HTML报告已保存: {output_path}", flush=True)


if __name__ == '__main__':
    print("=" * 60)
    print("ML主升浪策略 - 全A市场扫描")
    print("=" * 60)
    
    # 加载标注文件
    annotations_file = BASE_DIR / 'data' / 'annotations.csv'
    exclude_codes = set()
    if annotations_file.exists():
        ann = pd.read_csv(annotations_file, encoding='utf-8-sig')
        exclude_codes = set(str(c).strip().zfill(6) for c in ann['stock_code'])
        print(f"排除已标注: {len(exclude_codes)} 只")
    
    # 获取全A股票
    name_map = get_all_a_stocks()
    if not name_map:
        print("获取全A股票列表失败！")
        sys.exit(1)
    
    # 确保code格式一致
    clean_map = {}
    for code, name in name_map.items():
        clean_map[str(code).zfill(6)] = name
    
    print(f"全A: {len(clean_map)} 只, 排除已标注 {len(exclude_codes)} 只")
    
    # 扫描
    results = scan_stocks_full_a(
        name_map=clean_map,
        exclude_codes=exclude_codes,
        top_n=50,
        upcoming_n=30
    )
    
    if not results:
        print("扫描失败！")
        sys.exit(1)
    
    # 保存结果
    signals_dir = BASE_DIR / 'signals'
    signals_dir.mkdir(parents=True, exist_ok=True)
    
    with open(signals_dir / 'scan_full_a_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    top_df = pd.DataFrame(results['top_uptrend'])
    up_df = pd.DataFrame(results.get('upcoming', []))
    all_results = pd.concat([top_df, up_df], ignore_index=True)
    all_results.to_csv(signals_dir / 'scan_full_a_results.csv', index=False, encoding='utf-8-sig')
    
    generate_report(results, signals_dir / 'scan_full_a_report.html')
    
    # 打印关键结果
    print("\n" + "=" * 60)
    print("TOP 20:")
    for i, r in enumerate(results['top_uptrend'][:20]):
        print(f"  {i+1}. {r['code']} | prob={r['latest_prob']*100:.1f}% | trend={r['prob_trend_5d']:+.3f} | ret20d={r['ret_20d']*100:+.1f}%")
    
    print("\nTOP 10:")
    for i, r in enumerate(results['upcoming'][:10]):
        print(f"  {i+1}. {r['code']} | prob={r['latest_prob']*100:.1f}% | trend={r['prob_trend_5d']:+.3f} | ret20d={r['ret_20d']*100:+.1f}%")
