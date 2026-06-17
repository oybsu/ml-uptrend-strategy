"""ML主升浪量化策略 - 统一入口

用法:
    python run.py train          # 用标注数据训练模型
    python run.py download       # 批量下载全A数据(qlib)
    python run.py scan           # 全A扫描(需先download)
    python run.py update         # 更新数据+扫描(一键运行)
    python run.py predict 000001 # 预测单只股票
    python run.py stocklist      # 更新全A股票列表

数据源: qlib (chenditc/investment_data, GitHub托管，纯离线读取)
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

from data_loader import get_stock_data, load_annotations, qlib_download_batch, qlib_get_stock_list


def get_end_date():
    """获取结束日期: config中end_date='auto'则取今天, 否则用配置值"""
    with open(BASE_DIR / 'config.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    ed = cfg['data'].get('end_date', 'auto')
    if ed == 'auto':
        return time.strftime('%Y%m%d')
    return ed


def get_all_a_stocks():
    """获取全部A股股票代码列表(排除ST/退市/北交所)
    
    从qlib数据获取股票列表(自动从CSV补充名称)
    """
    csv_file = BASE_DIR / 'data' / 'full_a_stocks.csv'
    
    # 从qlib获取(已内置CSV名称补充)
    print("从qlib获取全A列表...", flush=True)
    name_map = qlib_get_stock_list()
    if name_map:
        # 保存到CSV(保留名称)
        df = pd.DataFrame(list(name_map.items()), columns=['code', 'name'])
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"qlib获取成功: {len(name_map)} 只")
        return name_map
    
    # fallback: CSV缓存
    if csv_file.exists():
        df = pd.read_csv(csv_file, encoding='utf-8-sig', dtype={'code': str})
        df['code'] = df['code'].astype(str).str.zfill(6)
        name_map = dict(zip(df['code'].tolist(), df['name'].tolist()))
        print(f"从CSV缓存读取: {len(name_map)} 只")
        return name_map
    
    print("获取全A列表失败！")
    return {}


# ─── train ────────────────────────────────────────────────────────────

def cmd_train():
    """用标注数据训练模型"""
    from label_generator import create_labeled_dataset, generate_labels
    from factor_engine import calc_all_factors, prepare_features
    from model_trainer import train_model, load_latest_model
    print("=" * 60)
    print("ML主升浪策略 - 模型训练")
    print("=" * 60)

    annotations = load_annotations()
    print(f"标注: {len(annotations)} 只股票")

    # 下载数据
    # qlib模式: download步骤数据在scan_cache，train也用scan_cache
    with open(BASE_DIR / 'config.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    train_start_date = cfg['data'].get('start_date', '20240101')
    
    print("\n--- 下载标注股票数据 ---")
    train_cache_dir = BASE_DIR / 'data' / 'scan_cache'
    price_dict = {}
    failed = []
    for idx, row in annotations.iterrows():
        code = str(row['stock_code']).strip().zfill(6)
        name = str(row.get('note', '')).strip()
        print(f"  [{idx+1}/{len(annotations)}] {code} {name}...", end=' ', flush=True)
        df = get_stock_data(code, start_date=train_start_date, cache_dir=train_cache_dir)
        if df.empty:
            failed.append(code)
            print("FAIL")
        else:
            price_dict[code] = df
            print(f"OK ({len(df)} bars)")
    print(f"成功: {len(price_dict)}, 失败: {len(failed)}")

    # 生成标签+计算因子
    print("\n--- 计算因子和标签 ---")
    all_with_factors = []
    for code, price_df in price_dict.items():
        factors = calc_all_factors(price_df)
        if factors.empty:
            continue
        ann_row = annotations[annotations['stock_code'].astype(str).str.zfill(6) == code]
        if ann_row.empty:
            continue
        labels = generate_labels(price_df, str(ann_row.iloc[0]['start_date']), str(ann_row.iloc[0]['end_date']))
        # 只拼接因子+label，不拼接OHLCV
        factors = factors.copy()
        factors['label'] = labels['label'].values
        factors['stock_code'] = code
        # 转float32节省内存
        for c in factors.columns:
            if c not in ('label', 'stock_code') and factors[c].dtype == np.float64:
                factors[c] = factors[c].astype(np.float32)
        all_with_factors.append(factors)
        del factors

    full_df = pd.concat(all_with_factors, ignore_index=True)
    del all_with_factors
    full_df = full_df.dropna(subset=['label'])
    print(f"含因子的样本: {len(full_df)}")

    X, y, feature_names = prepare_features(full_df)
    X = X.fillna(0)
    print(f"特征数: {len(feature_names)}, 有效样本: {len(X)}")

    # 训练
    print("\n--- 训练LightGBM ---")
    model, eval_results, feat_imp = train_model(X, y, feature_names=feature_names)
    print("\n训练完成！")


# ─── download ─────────────────────────────────────────────────────────

def cmd_download():
    """批量下载全A数据(从qlib本地数据转换)"""
    cache_dir = BASE_DIR / 'data' / 'scan_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 全A列表
    name_map = get_all_a_stocks()
    if not name_map:
        print("获取全A列表失败！请先运行 python run.py stocklist")
        return

    # 排除已标注
    ann_file = BASE_DIR / 'data' / 'annotations.csv'
    ann_codes = set()
    if ann_file.exists():
        ann = pd.read_csv(ann_file, encoding='utf-8-sig')
        ann_codes = set(str(c).strip().zfill(6) for c in ann['stock_code'])

    stock_codes = [c for c in name_map if c not in ann_codes]
    
    end_date = get_end_date()
    with open(BASE_DIR / 'config.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    start_date = cfg['data'].get('start_date', '20240101')

    # qlib: GitHub托管数据，海外可访问，社区每日更新，纯离线读取
    print(f"\n--- qlib数据源下载 ---")
    print(f"数据源: qlib (chenditc/investment_data, GitHub托管)")
    print(f"日期范围: {start_date} ~ {end_date}")
    qlib_dir = os.environ.get('QLIB_DATA_DIR', None)
    ok, fail = qlib_download_batch(
        stock_codes=stock_codes,
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
        qlib_dir=qlib_dir,
        adjust='hfq'
    )


# ─── scan ─────────────────────────────────────────────────────────────

def cmd_scan():
    """全A扫描(需先download)"""
    from model_trainer import load_latest_model
    from signal_generator import predict_stock
    print("=" * 60)
    print("ML主升浪策略 - 全A市场扫描")
    print("=" * 60)

    # 排除已标注
    ann_file = BASE_DIR / 'data' / 'annotations.csv'
    exclude_codes = set()
    if ann_file.exists():
        ann = pd.read_csv(ann_file, encoding='utf-8-sig')
        exclude_codes = set(str(c).strip().zfill(6) for c in ann['stock_code'])

    name_map = get_all_a_stocks()
    if not name_map:
        print("获取全A列表失败！")
        return

    clean_map = {str(c).zfill(6): n for c, n in name_map.items()}
    stock_codes = [c for c in clean_map if c not in exclude_codes]

    cache_dir = BASE_DIR / 'data' / 'scan_cache'
    cached_count = len(list(cache_dir.glob('*.parquet'))) if cache_dir.exists() else 0
    print(f"全A: {len(clean_map)}, 排除标注: {len(exclude_codes)}, 待扫描: {len(stock_codes)}, 缓存: {cached_count}")

    # 加载模型
    model, meta = load_latest_model()
    if model is None:
        print("未找到模型！请先运行 python run.py train")
        return

    model_feature_names = model.feature_name()

    # 扫描
    with open(BASE_DIR / 'config.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    top_n = cfg.get('scan', {}).get('top_n', 50)
    upcoming_n = cfg.get('scan', {}).get('upcoming_n', 30)

    results = []
    ok = fail = 0
    t0 = time.time()

    for idx, code in enumerate(stock_codes):
        if (idx + 1) % 200 == 0:
            spd = (idx + 1) / (time.time() - t0)
            eta = (len(stock_codes) - idx - 1) / spd / 60
            print(f"进度: {idx+1}/{len(stock_codes)} | ok={ok} fail={fail} | {spd:.1f}/s | ETA: {eta:.0f}min", flush=True)

        try:
            price_df = get_stock_data(code, cache_dir=cache_dir)
            if price_df.empty or len(price_df) < 120:
                fail += 1
                continue
            pred_df = predict_stock(model, price_df, feature_names=model_feature_names)
            if pred_df.empty:
                fail += 1
                continue

            latest_prob = float(pred_df.iloc[-1]['prob'])
            prob_trend = float(latest_prob - pred_df.iloc[-6]['prob']) if len(pred_df) >= 10 else 0
            latest_price = float(price_df.iloc[-1]['close'])
            ret_20d = float(price_df.iloc[-1]['close'] / price_df.iloc[-20]['close'] - 1) if len(price_df) >= 20 else 0

            results.append({
                'code': code, 'name': clean_map.get(code, ''),
                'latest_prob': round(latest_prob, 4),
                'prob_trend_5d': round(prob_trend, 4),
                'latest_price': round(latest_price, 2),
                'ret_20d': round(ret_20d, 4)
            })
            ok += 1
        except Exception:
            fail += 1

    if not results:
        print("扫描失败！无有效结果")
        return

    df = pd.DataFrame(results)
    top = df.nlargest(top_n, 'latest_prob')
    upcoming_mask = (df['latest_prob'] >= 0.45) & (df['latest_prob'] < 0.60) & (df['prob_trend_5d'] > 0)
    upcoming = df[upcoming_mask].nlargest(upcoming_n, 'prob_trend_5d')

    elapsed = time.time() - t0
    print(f"\n扫描完成！{ok} 成功, {fail} 失败, 耗时 {elapsed/60:.1f}min")

    # 保存
    signals_dir = BASE_DIR / 'signals'
    signals_dir.mkdir(parents=True, exist_ok=True)

    res = {
        'top_uptrend': top.to_dict('records'),
        'upcoming': upcoming.to_dict('records'),
        'total_scanned': ok,
        'total_failed': fail,
        'scan_time': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(signals_dir / 'scan_full_a_results.json', 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    all_df = pd.concat([top, upcoming], ignore_index=True)
    all_df.to_csv(signals_dir / 'scan_full_a_results.csv', index=False, encoding='utf-8-sig')

    _generate_html_report(res, signals_dir / 'scan_full_a_report.html')

    # 控制台输出
    print("\n=== TOP 20 主升浪 ===")
    for i, r in enumerate(res['top_uptrend'][:20]):
        print(f"  {i+1:2d}. {r['code']} {r['name']} | {r['latest_prob']*100:.1f}% | trend={r['prob_trend_5d']:+.3f} | 20d={r['ret_20d']*100:+.1f}%")

    print("\n=== TOP 10 即将启动 ===")
    for i, r in enumerate(res['upcoming'][:10]):
        print(f"  {i+1:2d}. {r['code']} {r['name']} | {r['latest_prob']*100:.1f}% | trend={r['prob_trend_5d']:+.3f} | 20d={r['ret_20d']*100:+.1f}%")

    print(f"\nHTML报告: {signals_dir / 'scan_full_a_report.html'}")


def _generate_html_report(results, output_path):
    """生成HTML报告"""
    top = results.get('top_uptrend', [])
    upcoming = results.get('upcoming', [])
    total = results.get('total_scanned', 0)
    scan_time = results.get('scan_time', '')

    parts = [f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8">
<title>ML主升浪策略 - 全A扫描报告</title>
<style>
body{{font-family:'Microsoft YaHei',sans-serif;margin:20px;background:#f5f5f5}}
h1{{color:#1a5276;border-bottom:2px solid #2980b9;padding-bottom:10px}}
h2{{color:#2c3e50;margin-top:30px}}
table{{border-collapse:collapse;width:100%;margin:10px 0;background:white;font-size:13px}}
th{{background:#2980b9;color:white;padding:8px 6px;text-align:center}}
td{{padding:7px 6px;text-align:center;border-bottom:1px solid #ddd}}
tr:nth-child(even){{background:#f9f9f9}}tr:hover{{background:#eaf2f8}}
.up{{color:#e74c3c;font-weight:bold}}.down{{color:#27ae60}}
.summary{{background:white;padding:15px;border-radius:8px;margin:10px 0}}
.badge{{display:inline-block;padding:2px 6px;border-radius:3px;font-size:11px;margin:1px;color:white}}
.badge-high{{background:#e74c3c}}.badge-mid{{background:#f39c12}}.badge-low{{background:#3498db}}.badge-strong{{background:#8e44ad}}
</style></head><body>
<h1>ML主升浪策略 - 全A扫描报告</h1>
<div class="summary">
<p><b>扫描范围:</b> 全A股(排除ST/退市/北交所) | <b>成功:</b> {total} 只 | <b>时间:</b> {scan_time}</p>
<p><b>模型:</b> LightGBM | <b>因子:</b> 123 | <b>阈值:</b> buy=0.60 sell=0.50</p>
</div>"""]

    parts.append("<h2>TOP 50 主升浪概率最高</h2><table><tr><th>排名</th><th>代码</th><th>名称</th><th>概率</th><th>5日趋势</th><th>最新价</th><th>20日涨幅</th><th>信号</th></tr>")
    for i, r in enumerate(top):
        p, t, ret = r['latest_prob'], r['prob_trend_5d'], r['ret_20d']
        sgs = []
        if p >= 0.65: sgs.append('<span class="badge badge-high">高概率</span>')
        if t > 0.1: sgs.append('<span class="badge badge-mid">趋势加速</span>')
        if ret > 0.3: sgs.append('<span class="badge badge-strong">强势</span>')
        elif ret > 0.15: sgs.append('<span class="badge badge-low">走强</span>')
        tc = 'up' if t > 0 else 'down'
        rc = 'up' if ret > 0 else 'down'
        parts.append(f'<tr><td>{i+1}</td><td>{r["code"]}</td><td>{r["name"]}</td><td><b>{p*100:.1f}%</b></td><td class="{tc}">{t:+.3f}</td><td>{r["latest_price"]:.2f}</td><td class="{rc}">{ret*100:+.1f}%</td><td>{" ".join(sgs) or "-"}</td></tr>')
    parts.append('</table>')

    parts.append("<h2>即将启动 (概率0.45-0.60 + 趋势上升)</h2><table><tr><th>排名</th><th>代码</th><th>名称</th><th>概率</th><th>5日趋势</th><th>最新价</th><th>20日涨幅</th></tr>")
    for i, r in enumerate(upcoming):
        p, t, ret = r['latest_prob'], r['prob_trend_5d'], r['ret_20d']
        tc = 'up' if t > 0 else 'down'
        rc = 'up' if ret > 0 else 'down'
        parts.append(f'<tr><td>{i+1}</td><td>{r["code"]}</td><td>{r["name"]}</td><td><b>{p*100:.1f}%</b></td><td class="{tc}">{t:+.3f}</td><td>{r["latest_price"]:.2f}</td><td class="{rc}">{ret*100:+.1f}%</td></tr>')
    parts.append('</table>')

    parts.append(f"<p style='color:#7f8c8d;font-size:12px;margin-top:30px'>报告时间: {time.strftime('%Y-%m-%d %H:%M:%S')}</p></body></html>")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(''.join(parts))
    print(f"HTML报告: {output_path}")


# ─── update (一键运行) ────────────────────────────────────────────────

def cmd_update():
    """更新数据+扫描(一键运行)"""
    print("=" * 60)
    print("ML主升浪策略 - 一键更新")
    print("=" * 60)
    print("\n[1/2] 下载数据...")
    cmd_download()
    print("\n[2/2] 扫描市场...")
    cmd_scan()
    print("\n一键更新完成！")


# ─── predict (单只股票) ───────────────────────────────────────────────

def cmd_predict(code):
    """预测单只股票，无模型则自动训练"""
    from model_trainer import load_latest_model
    from signal_generator import predict_stock
    code = str(code).strip().zfill(6)
    print(f"预测 {code} ...")

    # 检查模型，无则训练
    model, meta = load_latest_model()
    if model is None:
        print("未找到模型，自动训练中...")
        cmd_train()
        model, meta = load_latest_model()
        if model is None:
            print("训练失败，无法预测")
            return
        print("模型训练完成，继续预测")

    # 获取股票名称
    csv_file = BASE_DIR / 'data' / 'full_a_stocks.csv'
    stock_name = ''
    if csv_file.exists():
        try:
            df = pd.read_csv(csv_file, encoding='utf-8-sig', dtype={'code': str})
            df['code'] = df['code'].astype(str).str.zfill(6)
            name_row = df[df['code'] == code]
            if not name_row.empty:
                stock_name = str(name_row.iloc[0]['name'])
        except Exception:
            pass

    cache_dir = BASE_DIR / 'data' / 'scan_cache'
    price_df = get_stock_data(code, cache_dir=cache_dir)
    if price_df.empty:
        print(f"无数据: {code}")
        return

    pred_df = predict_stock(model, price_df, feature_names=model.feature_name())
    if pred_df.empty:
        print("预测失败")
        return

    # 最近20天
    display_name = f"{code} {stock_name}" if stock_name else code
    recent = pred_df.tail(20)
    print(f"\n{display_name} 最近20日预测:")
    print(f"{'日期':>12s} | {'概率':>6s} | {'信号':>4s}")
    print("-" * 30)
    for _, row in recent.iterrows():
        sig = {1: '买入', -1: '卖出', 0: '观望', 2: '持有'}.get(int(row['signal']), '观望')
        print(f"{str(row['date'])[:10]:>12s} | {row['prob']*100:5.1f}% | {sig}")

    latest_prob = pred_df.iloc[-1]['prob']
    latest_signal = int(pred_df.iloc[-1]['signal'])
    signal_text = {1: '买入', -1: '卖出', 0: '观望', 2: '持有'}.get(latest_signal, '观望')

    if len(pred_df) >= 6:
        prob_trend = latest_prob - pred_df.iloc[-6]['prob']
    else:
        prob_trend = 0

    print(f"\n=== {display_name} 预测结果 ===")
    print(f"最新概率: {latest_prob*100:.1f}%")
    print(f"5日趋势: {prob_trend:+.3f}")
    print(f"最新信号: {signal_text}")
    if not price_df.empty:
        print(f"最新价格: {price_df.iloc[-1]['close']:.2f}")
    print("=" * 30)


# ─── stocklist ────────────────────────────────────────────────────────

def cmd_stocklist():
    """更新全A股票列表(从qlib获取)"""
    # 强制删除旧缓存，重新获取
    csv_file = BASE_DIR / 'data' / 'full_a_stocks.csv'
    if csv_file.exists():
        os.remove(csv_file)
    
    name_map = get_all_a_stocks()
    if name_map:
        print(f"已保存 {len(name_map)} 只到 {csv_file}")
    else:
        print("获取全A列表失败！")


# ─── main ─────────────────────────────────────────────────────────────

USAGE = """
ML主升浪量化策略 v4.0 (qlib数据源)
==================================

用法: python run.py <命令> [参数]

命令:
  train          用标注数据训练模型
  download       批量下载全A数据(qlib)
  scan           全A扫描(需先download)
  update         一键更新(download + scan)
  predict <代码> 预测单只股票, 如 python run.py predict 000001
  stocklist      更新全A股票列表

数据源:
  qlib           chenditc/investment_data, GitHub托管，纯离线读取

工作流:
  python run.py train -> python run.py download -> python run.py scan
  日常: python run.py update

数据目录:
  data/annotations.csv   标注文件(主升浪区间)
  data/full_a_stocks.csv 全A股票列表
  data/scan_cache/       日线数据缓存(~5000个parquet)
  models/                LightGBM模型
  signals/               扫描结果(JSON/CSV/HTML)
"""


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(0)

    cmd = sys.argv[1].lower()

    if cmd == 'train':
        cmd_train()
    elif cmd == 'download':
        cmd_download()
    elif cmd == 'scan':
        cmd_scan()
    elif cmd == 'update':
        cmd_update()
    elif cmd == 'predict':
        if len(sys.argv) < 3:
            print("用法: python run.py predict <股票代码>")
            print("示例: python run.py predict 000001")
        else:
            cmd_predict(sys.argv[2])
    elif cmd == 'stocklist':
        cmd_stocklist()
    else:
        print(USAGE)
