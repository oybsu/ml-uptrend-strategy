"""数据加载模块 - 基于qlib数据源，parquet缓存

数据源: qlib (chenditc/investment_data, GitHub托管，海外可访问，纯离线)
"""
import os
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
with open(BASE_DIR / 'config.json', 'r', encoding='utf-8') as f:
    CONFIG = json.load(f)


def _get_end_date():
    """解析end_date配置: 'auto'取今天，否则用配置值"""
    with open(BASE_DIR / 'config.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    ed = cfg['data'].get('end_date', 'auto')
    if ed == 'auto':
        import time as _time
        return _time.strftime('%Y%m%d')
    return ed


# ─── qlib: 社区维护数据源(GitHub托管, 海外可访问) ────────────────────────

QLIB_DATA_URL = "https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz"
QLIB_DATA_DIR = Path.home() / '.qlib' / 'qlib_data' / 'cn_data'

_qlib_initialized = False


def qlib_ensure_data(qlib_dir=None):
    """确保qlib数据已下载并解压
    
    从 chenditc/investment_data Release 下载 qlib_bin.tar.gz,
    解压到 ~/.qlib/qlib_data/cn_data/ (或指定目录)
    
    Returns:
        qlib_data_dir: Path
    """
    if qlib_dir is None:
        qlib_dir = QLIB_DATA_DIR
    qlib_dir = Path(qlib_dir)
    
    # 检查是否已有数据(看 calendars 目录是否存在)
    if (qlib_dir / 'calendars').exists():
        print(f"qlib数据已存在: {qlib_dir}")
        return qlib_dir
    
    print("qlib数据不存在，开始下载...")
    import tempfile
    import subprocess
    
    qlib_dir.mkdir(parents=True, exist_ok=True)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = Path(tmpdir) / 'qlib_bin.tar.gz'
        
        # 下载
        print(f"  下载: {QLIB_DATA_URL}")
        print(f"  约530MB，请稍候...")
        try:
            subprocess.run(
                ['curl', '-L', '-o', str(tar_path), QLIB_DATA_URL],
                check=True, timeout=600
            )
        except FileNotFoundError:
            # curl 不存在，用 Python 下载
            import urllib.request
            urllib.request.urlretrieve(QLIB_DATA_URL, str(tar_path))
        except subprocess.TimeoutExpired:
            print("  下载超时！")
            return qlib_dir
        
        size_mb = tar_path.stat().st_size / 1024 / 1024
        print(f"  下载完成: {size_mb:.1f} MB")
        
        # 解压
        print(f"  解压到: {qlib_dir}")
        subprocess.run(
            ['tar', '-zxf', str(tar_path), '-C', str(qlib_dir), '--strip-components=1'],
            check=True, timeout=300
        )
    
    print(f"  qlib数据就绪!")
    return qlib_dir


def _qlib_ensure_init(qlib_dir=None):
    """确保qlib已初始化(只初始化一次)"""
    global _qlib_initialized
    if _qlib_initialized:
        return
    
    import qlib
    from qlib.config import REG_CN
    
    qlib_data_dir = qlib_ensure_data(qlib_dir)
    
    try:
        qlib.init(provider_uri=str(qlib_data_dir), region=REG_CN)
    except Exception:
        pass
    _qlib_initialized = True


def _qlib_code_to_raw(qlib_code):
    """SH600519 → 600519"""
    return qlib_code[2:]


def _raw_code_to_qlib(code):
    """600519 → SH600519, 000001 → SZ000001"""
    code = str(code).strip().zfill(6)
    if code.startswith('6'):
        return f'SH{code}'
    else:
        return f'SZ{code}'


def _download_qlib_single(stock_code, start_date, end_date, adjust='hfq'):
    """从qlib本地数据读取单只股票日线(不联网)
    
    Args:
        stock_code: 6位代码如 '000001'
        start_date: YYYYMMDD
        end_date: YYYYMMDD
        adjust: 'hfq'
    
    Returns:
        DataFrame or None
    """
    from qlib.data import D
    
    _qlib_ensure_init()
    
    qlib_code = _raw_code_to_qlib(stock_code)
    qlib_start = f'{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}'
    qlib_end = f'{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}'
    
    try:
        df = D.features([qlib_code], ['$open', '$high', '$low', '$close', '$volume', '$factor'],
                       start_time=qlib_start, end_time=qlib_end, freq='day')
        if df is None or df.empty:
            return None
        
        stock_df = df.loc[qlib_code].copy()
        stock_df = stock_df.dropna(subset=['$close'])
        
        if len(stock_df) < 120:
            return None
        
        result = pd.DataFrame({
            'date': stock_df.index,
            'open': stock_df['$open'].values,
            'high': stock_df['$high'].values,
            'low': stock_df['$low'].values,
            'close': stock_df['$close'].values,
            'volume': stock_df['$volume'].values,
            'amount': 0
        })
        
        # 后复权
        if '$factor' in stock_df.columns and adjust == 'hfq':
            factor = stock_df['$factor'].values
            last_factor = factor[-1] if len(factor) > 0 else 1.0
            if last_factor > 0:
                hfq_ratio = factor / last_factor
                result['open'] = result['open'] / hfq_ratio
                result['high'] = result['high'] / hfq_ratio
                result['low'] = result['low'] / hfq_ratio
                result['close'] = result['close'] / hfq_ratio
                result['volume'] = result['volume'] * hfq_ratio
        
        result = result.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
        return result if len(result) >= 120 else None
    except Exception:
        return None


def get_stock_data(stock_code, start_date=None, end_date=None, adjust='hfq', cache_dir=None, max_retries=3):
    """获取单只股票日线数据
    
    从缓存读取，缓存未命中则从qlib本地数据获取(不联网)
    
    Args:
        stock_code: 股票代码(6位字符串)，如 '000001'
        start_date: 起始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD
        adjust: 复权方式 hfq(后复权)/qfq(前复权)
        cache_dir: 缓存目录
        max_retries: 重试次数(保留参数，qlib无需重试)
    
    Returns:
        DataFrame with columns: date, open, high, low, close, volume, amount
    """
    stock_code = str(stock_code).strip().zfill(6)
    
    if start_date is None:
        start_date = CONFIG['data']['start_date']
    if end_date is None:
        end_date = _get_end_date()
    if cache_dir is None:
        cache_dir = BASE_DIR / CONFIG['data']['cache_dir']
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_dir / f"{stock_code}_{adjust}.parquet"
    
    # 尝试读取缓存
    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            if len(df) > 0 and (df['date'].iloc[-1] >= pd.Timestamp(end_date) - pd.Timedelta(days=2)):
                mask = (df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= df['date'].iloc[-1])
                return df[mask].reset_index(drop=True)
        except Exception:
            pass
    
    # 从qlib本地数据读取
    df = _download_qlib_single(stock_code, start_date, end_date, adjust)
    if df is not None and len(df) >= 120:
        df.to_parquet(cache_file, index=False)
        mask = (df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))
        return df[mask].reset_index(drop=True)
    
    return pd.DataFrame()


def qlib_download_batch(stock_codes, start_date='20240101', end_date=None,
                        cache_dir=None, qlib_dir=None, adjust='hfq'):
    """用qlib数据源批量转换为parquet缓存
    
    流程:
        1. 确保qlib二进制数据已下载(约530MB tar.gz)
        2. 用qlib D.features() API读取每只股票数据
        3. 转换为项目标准的parquet格式缓存
    
    优势:
        - 数据托管在GitHub Release，海外服务器可访问
        - 社区每日自动更新(chenditc/investment_data)
        - 多源融合数据(质量高于单源)
    
    Args:
        stock_codes: 股票代码列表(6位字符串)
        start_date: YYYYMMDD
        end_date: YYYYMMDD or None
        cache_dir: parquet缓存目录
        qlib_dir: qlib数据目录(默认 ~/.qlib/qlib_data/cn_data)
        adjust: 复权方式(hfq, qlib数据已含复权因子)
    
    Returns:
        (success_count, fail_count)
    """
    import qlib
    from qlib.config import REG_CN
    from qlib.data import D
    
    if end_date is None:
        end_date = _get_end_date()
    if cache_dir is None:
        cache_dir = BASE_DIR / CONFIG['data']['scan_cache_dir']
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 确保qlib数据已下载
    qlib_data_dir = qlib_ensure_data(qlib_dir)
    
    # 初始化qlib
    try:
        qlib.init(provider_uri=str(qlib_data_dir), region=REG_CN)
    except Exception:
        # 可能已经初始化过
        pass
    
    # 排除已缓存
    cached = set()
    for f in os.listdir(cache_dir):
        if f.endswith('.parquet'):
            cached.add(f.split('_')[0])
    
    need = [c for c in stock_codes if c not in cached]
    
    if not need:
        print(f"全部 {len(stock_codes)} 只已缓存，无需下载！")
        return len(cached & set(stock_codes)), 0
    
    print(f"总计: {len(stock_codes)}, 已缓存: {len(cached & set(stock_codes))}, 需转换: {len(need)}")
    print(f"数据源: qlib (chenditc/investment_data)")
    print(f"日期范围: {start_date} ~ {end_date}")
    
    # qlib日期格式: YYYY-MM-DD
    qlib_start = f'{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}'
    qlib_end = f'{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}'
    
    ok = fail = 0
    t0 = time.time()
    
    # 批量读取: qlib支持一次读取多只股票
    batch_size = 50
    qlib_fields = ['$open', '$high', '$low', '$close', '$volume', '$factor']
    
    for batch_start in range(0, len(need), batch_size):
        batch = need[batch_start:batch_start + batch_size]
        qlib_codes = [_raw_code_to_qlib(c) for c in batch]
        
        try:
            # 批量读取
            df = D.features(qlib_codes, qlib_fields,
                          start_time=qlib_start, end_time=qlib_end,
                          freq='day')
            
            if df is None or df.empty:
                fail += len(batch)
                continue
            
            # 处理每只股票
            for code in batch:
                qlib_code = _raw_code_to_qlib(code)
                try:
                    if qlib_code not in df.index.get_level_values('instrument'):
                        fail += 1
                        continue
                    
                    stock_df = df.loc[qlib_code].copy()
                    stock_df = stock_df.dropna(subset=['$close'])
                    
                    if len(stock_df) < 120:
                        fail += 1
                        continue
                    
                    # 重命名列
                    result = pd.DataFrame({
                        'date': stock_df.index,
                        'open': stock_df['$open'].values,
                        'high': stock_df['$high'].values,
                        'low': stock_df['$low'].values,
                        'close': stock_df['$close'].values,
                        'volume': stock_df['$volume'].values,
                        'amount': 0  # qlib不含成交额，填充0
                    })
                    
                    # 后复权: 用factor调整ohlcv
                    if '$factor' in stock_df.columns and adjust == 'hfq':
                        factor = stock_df['$factor'].values
                        # factor是前复权因子，后复权需要用最后一个factor做基准
                        last_factor = factor[-1] if len(factor) > 0 else 1.0
                        if last_factor > 0:
                            hfq_ratio = factor / last_factor
                            result['open'] = result['open'] / hfq_ratio
                            result['high'] = result['high'] / hfq_ratio
                            result['low'] = result['low'] / hfq_ratio
                            result['close'] = result['close'] / hfq_ratio
                            result['volume'] = result['volume'] * hfq_ratio
                    
                    result = result.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
                    
                    if len(result) >= 120:
                        cf = cache_dir / f"{code}_{adjust}.parquet"
                        result.to_parquet(cf, index=False)
                        ok += 1
                    else:
                        fail += 1
                        
                except Exception:
                    fail += 1
            
        except Exception as e:
            fail += len(batch)
            if batch_start % 500 == 0:
                print(f"  批次 {batch_start} 异常: {e}")
        
        # 进度
        done = min(batch_start + batch_size, len(need))
        if done % 500 < batch_size or done == len(need):
            elapsed = time.time() - t0
            spd = done / elapsed if elapsed > 0 else 0
            eta = (len(need) - done) / spd / 60 if spd > 0 else 0
            print(f"  进度: {done}/{len(need)} | ok={ok} fail={fail} | {spd:.1f}/s | ETA: {eta:.0f}min", flush=True)
    
    elapsed = time.time() - t0
    print(f"qlib转换完成: 成功={ok}, 失败={fail}, 耗时={elapsed/60:.1f}min")
    return ok, fail


def qlib_get_stock_list(qlib_dir=None):
    """从qlib数据获取全A股票列表，并从本地CSV补充股票名称
    
    Returns:
        dict: {code: name} 如 {'600519': '贵州茅台'}
    """
    import qlib
    from qlib.config import REG_CN
    from qlib.data import D
    
    qlib_data_dir = qlib_ensure_data(qlib_dir)
    
    try:
        qlib.init(provider_uri=str(qlib_data_dir), region=REG_CN)
    except Exception:
        pass
    
    try:
        instruments = D.list_instruments(D.instruments('all'), as_list=True)
        name_map = {}
        for inst in instruments:
            code = _qlib_code_to_raw(inst)
            # 过滤: 只保留6位数字的主板+创业板+科创板
            if len(code) == 6 and code[0] in '0136':
                name_map[code] = ''
        
        # 从本地CSV补充股票名称
        csv_file = BASE_DIR / 'data' / 'full_a_stocks.csv'
        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file, encoding='utf-8-sig', dtype={'code': str})
                df['code'] = df['code'].astype(str).str.zfill(6)
                csv_names = dict(zip(df['code'].tolist(), df['name'].tolist()))
                filled = 0
                for code in name_map:
                    if csv_names.get(code):
                        name_map[code] = csv_names[code]
                        filled += 1
                print(f"从CSV补充名称: {filled}/{len(name_map)} 只")
            except Exception as e:
                print(f"读取股票名称CSV失败: {e}")
        
        print(f"qlib获取全A列表: {len(name_map)} 只")
        return name_map
    except Exception as e:
        print(f"qlib获取股票列表失败: {e}")
        return {}


def load_annotations(annotations_file=None):
    """加载标注文件，自动补零股票代码"""
    if annotations_file is None:
        annotations_file = BASE_DIR / CONFIG['data']['annotations_file']
    annotations_file = Path(annotations_file)
    if not annotations_file.exists():
        raise FileNotFoundError(f"标注文件不存在: {annotations_file}")
    df = pd.read_csv(annotations_file, encoding='utf-8-sig')
    df['stock_code'] = df['stock_code'].astype(str).str.strip().str.zfill(6)
    return df
