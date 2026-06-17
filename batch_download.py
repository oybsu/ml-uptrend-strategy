"""批量下载全A股数据 - 使用baostock多进程"""
import sys
import os
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Process, Queue

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent


def get_need_download_codes():
    """获取需要下载的股票代码列表"""
    cache_dir = BASE_DIR / 'data' / 'scan_cache'
    cached = set()
    if cache_dir.exists():
        for f in os.listdir(cache_dir):
            if f.endswith('.parquet'):
                cached.add(f.split('_')[0])
    
    # 全A列表
    csv_file = BASE_DIR / 'data' / 'full_a_stocks.csv'
    df = pd.read_csv(csv_file, encoding='utf-8-sig', dtype={'code': str})
    df['code'] = df['code'].astype(str).str.zfill(6)
    all_codes = df['code'].tolist()
    
    # 标注代码
    ann_file = BASE_DIR / 'data' / 'annotations.csv'
    ann = pd.read_csv(ann_file, encoding='utf-8-sig')
    ann_codes = set(str(c).strip().zfill(6) for c in ann['stock_code'])
    
    need = [c for c in all_codes if c not in cached and c not in ann_codes]
    return need


def download_worker(codes, cache_dir, worker_id, result_queue):
    """下载工作进程"""
    import baostock as bs
    
    lg = bs.login()
    if lg.error_code != '0':
        result_queue.put(('login_fail', worker_id, 0, 0))
        return
    
    success = 0
    fail = 0
    
    for code in codes:
        cache_file = Path(cache_dir) / f"{code}_hfq.parquet"
        if cache_file.exists():
            success += 1
            continue
        
        # 转换代码格式
        if code.startswith('6'):
            bs_code = f'sh.{code}'
        else:
            bs_code = f'sz.{code}'
        
        try:
            rs = bs.query_history_k_data_plus(
                bs_code,
                'date,open,high,low,close,volume,amount',
                start_date='2024-01-01',
                end_date='2026-06-16',
                frequency='d',
                adjustflag='1'  # 后复权
            )
            data = []
            while (rs.error_code == '0') and rs.next():
                data.append(rs.get_row_data())
            
            if data:
                df = pd.DataFrame(data, columns=rs.fields)
                for c in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                df['date'] = pd.to_datetime(df['date'])
                df = df.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
                if len(df) >= 120:
                    df.to_parquet(cache_file, index=False)
                    success += 1
                else:
                    fail += 1
            else:
                fail += 1
        except Exception:
            fail += 1
    
    bs.logout()
    result_queue.put(('done', worker_id, success, fail))


if __name__ == '__main__':
    print("=" * 60)
    print("批量下载全A股数据 (baostock)")
    print("=" * 60)
    
    need_codes = get_need_download_codes()
    print(f"需下载: {len(need_codes)} 只")
    
    if not need_codes:
        print("无需下载！")
        sys.exit(0)
    
    cache_dir = BASE_DIR / 'data' / 'scan_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 分成N个worker
    num_workers = 4
    chunk_size = len(need_codes) // num_workers + 1
    chunks = [need_codes[i:i+chunk_size] for i in range(0, len(need_codes), chunk_size)]
    
    print(f"启动 {len(chunks)} 个worker, 每个 {chunk_size} 只")
    
    result_queue = Queue()
    processes = []
    
    start_time = time.time()
    
    for i, chunk in enumerate(chunks):
        p = Process(target=download_worker, args=(chunk, str(cache_dir), i, result_queue))
        p.start()
        processes.append(p)
    
    # 等待完成
    total_success = 0
    total_fail = 0
    completed = 0
    
    while completed < len(processes):
        try:
            status, wid, s, f = result_queue.get(timeout=10)
            if status == 'done':
                total_success += s
                total_fail += f
                completed += 1
                elapsed = time.time() - start_time
                print(f"Worker {wid} done: success={s}, fail={f} | Total: {total_success}+{total_fail}/{len(need_codes)} | {elapsed/60:.1f}min", flush=True)
            elif status == 'login_fail':
                completed += 1
                print(f"Worker {wid} login failed!", flush=True)
        except Exception:
            # 检查进程是否还活着
            alive = sum(1 for p in processes if p.is_alive())
            if alive == 0:
                break
    
    for p in processes:
        p.join(timeout=5)
    
    elapsed = time.time() - start_time
    print(f"\n下载完成！成功: {total_success}, 失败: {total_fail}, 耗时: {elapsed/60:.1f}min")
