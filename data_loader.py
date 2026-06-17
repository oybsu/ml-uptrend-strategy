"""数据加载模块 - 支持多数据源，parquet缓存

DATA_SOURCE=qlib:  从本地qlib二进制数据读取(GitHub托管，海外可访问，纯离线)
默认(pytdx):       直连通达信服务器，速度极快（全A约5-10分钟），需自行计算复权
fallback:          baostock(已复权数据) + efinance(东方财富兜底)
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

# ─── 通达信服务器列表 ────────────────────────────────────────────────

TDX_SERVERS = [
    ('119.147.212.81', 7709),   # 华泰证券
    ('112.74.214.43', 7709),    # 中投证券
    ('221.231.141.60', 7709),   # 华泰证券南京
    ('101.227.73.20', 7709),    # 上海电信
    ('101.227.77.254', 7709),   # 上海电信
    ('14.215.128.18', 7709),    # 广州电信
    ('59.173.18.140', 7709),    # 武汉电信
    ('180.153.18.170', 7709),   # 上海电信
    ('47.103.48.45', 7709),     # 阿里云
    ('218.75.126.9', 7709),     # 浙江电信
    ('115.238.56.198', 7709),   # 浙江电信
    ('218.108.98.244', 7709),   # 杭州
    ('124.160.88.183', 7709),   # 杭州
]


def _get_end_date():
    """解析end_date配置: 'auto'取今天，否则用配置值"""
    with open(BASE_DIR / 'config.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    ed = cfg['data'].get('end_date', 'auto')
    if ed == 'auto':
        import time as _time
        return _time.strftime('%Y%m%d')
    return ed


def _stock_code_to_market(code):
    """6位股票代码 → (market, code)
    
    market: 0=深圳, 1=上海
    """
    code = str(code).strip().zfill(6)
    if code.startswith(('6', '9')):  # 上海: 6xxxxx主板, 9xxxxxB股
        return 1, code
    elif code.startswith(('0', '1', '2', '3')):  # 深圳: 0xxxxx主板, 1xxxxxB股, 2xxxxx中小板, 3xxxxx创业板
        return 0, code
    else:
        return 0, code  # 默认深圳


# ─── pytdx: 直连通达信服务器 ─────────────────────────────────────────

def _safe_disconnect(api):
    """安全断开pytdx连接，确保底层socket被关闭

    pytdx的disconnect()内部先shutdown()再close()，如果连接已断(如服务器超时)，
    shutdown()抛异常后close()就不会执行，导致socket泄漏(ResourceWarning: unclosed)。
    此函数先尝试正常disconnect，失败则手动关闭底层socket。
    """
    try:
        api.disconnect()
    except Exception:
        pass
    if api.client is not None:
        import socket as _socket
        try:
            api.client.shutdown(_socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            api.client.close()
        except Exception:
            pass
        api.client = None


def _tdx_connect(time_out=5):
    """连接通达信服务器，返回 (api, server_info) 或 (None, None)"""
    from pytdx.hq import TdxHq_API
    import random
    
    api = TdxHq_API()
    # 随机打乱服务器顺序，避免所有进程都连同一台
    servers = TDX_SERVERS.copy()
    random.shuffle(servers)
    
    for host, port in servers:
        try:
            connected = api.connect(host, port, time_out=time_out)
            if connected:
                return api, f'{host}:{port}'
            # connect返回False: 新socket已创建但连接未建立，需关闭
            _safe_disconnect(api)
        except Exception:
            _safe_disconnect(api)
    return None, None


def _tdx_get_xdxr(code, market=None):
    """获取除权除息信息
    
    Returns:
        DataFrame with columns: year, month, day, category, fenhong, peigu, peigujia, songgu, zhuanzeng, youpei, suogu
        category: 1=除权, 2=除息
    """
    if market is None:
        market, code = _stock_code_to_market(code)
    
    api, _ = _tdx_connect()
    if api is None:
        return pd.DataFrame()
    
    try:
        xdxr = api.get_xdxr_info(market, code)
        if xdxr:
            df = pd.DataFrame(xdxr)
            return df
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()
    finally:
        _safe_disconnect(api)


def _tdx_apply_hfq(df, xdxr_df):
    """对不复权数据应用后复权因子
    
    后复权: 保持历史价格不变，调整最近的价格(向后复权)
    公式: 复权价 = 原价 × 复权因子
    复权因子从除权日开始向前累积
    
    Args:
        df: 不复权日线数据 (必须有 date, open, high, low, close, volume, amount)
        xdxr_df: 除权除息信息
    
    Returns:
        后复权后的 DataFrame
    """
    if xdxr_df.empty or df.empty:
        return df
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # 解析除权除息事件
    events = []
    for _, row in xdxr_df.iterrows():
        try:
            year = int(row.get('year', 0))
            month = int(row.get('month', 0))
            day = int(row.get('day', 0))
            if year == 0:
                continue
            ex_date = pd.Timestamp(year=year, month=month, day=day)
            
            category = int(row.get('category', 0))
            
            # 送股、转增(每10股)
            songgu = float(row.get('songgu', 0) or 0)
            zhuanzeng = float(row.get('zhuanzeng', 0) or 0)
            # 配股(每10股)
            peigu = float(row.get('peigu', 0) or 0)
            peigujia = float(row.get('peigujia', 0) or 0)  # 配股价
            # 分红(每10股)
            fenhong = float(row.get('fenhong', 0) or 0)
            # 缩股(每10股)
            suogu = float(row.get('suogu', 0) or 0)
            
            if category in (1, 2):  # 除权或除息
                events.append({
                    'ex_date': ex_date,
                    'songgu': songgu,
                    'zhuanzeng': zhuanzeng,
                    'peigu': peigu,
                    'peigujia': peigujia,
                    'fenhong': fenhong,
                    'suogu': suogu,
                })
        except Exception:
            continue
    
    if not events:
        return df
    
    # 按日期排序(从早到晚)
    events.sort(key=lambda x: x['ex_date'])
    
    # 计算后复权因子
    # 后复权: 从最新日期往回累积调整因子
    # 除权日当天及之前的价格需要调整
    # 调整因子 = (前收盘价 - 分红 + 配股价×配股比例) / (前收盘价 × (1 + 送转比例 + 配股比例))
    # 简化: 复权因子按除权事件累积
    
    # 后复权: 保持最老数据不变，新数据逐次调整
    # 累积因子从第一次除权开始
    cumulative_factor = 1.0
    factors = []  # (ex_date, factor) 从旧到新
    
    for evt in events:
        # 每10股的送转配比例
        sg = evt['songgu'] / 10.0   # 每股送股
        zz = evt['zhuanzeng'] / 10.0  # 每股转增
        pg = evt['peigu'] / 10.0    # 每股配股
        fh = evt['fenhong'] / 10.0  # 每股分红(税前)
        pgj = evt['peigujia']       # 配股价
        
        # 后复权因子: 除权日前收盘价调整
        # 送股+转增使股数增加, 配股使股数增加但需要钱, 分红使价格下降
        # factor = (1 + sg + zz + pg) / (1 - fh/close + pg*pgj/close)
        # 简化(近似, 不依赖前收盘价):
        # 对于送转: factor *= (1 + sg + zz + pg)
        # 对于分红: factor需要前收盘价, 近似处理
        
        # 更准确的计算:
        # 除权价 = (前收盘 - 每股分红 + 每股配股价×每股配股比例) / (1 + 每股送转 + 每股配股)
        # 后复权因子(乘到除权日之前的价格上):
        # hfq_factor = 1 / 除权价比例 = (1 + 每股送转 + 每股配股) * (前收盘 / (前收盘 - 每股分红 + 每股配股价×每股配股比例))
        # 但我们没有前收盘价...
        
        # pytdx标准后复权计算方式:
        # factor_cumulative: 从最新往历史方向累积
        # 每次除权: factor_new = factor_old * (1 + sg + zz + pg)
        # 分红单独处理(需除以前收盘)
        
        # 简化方案: 只考虑送转配，分红影响小(除息日价格不变只是分红到账)
        # 后复权: 老价格不动，新价格乘以调整因子
        # 所以除权日之后的价格 = 原价 × (1 + sg + zz + pg) / 1 (如果只有送转)
        # 但配股要减去成本...
        
        # 最简方案: 送转配按比例调整
        adjust_ratio = 1.0 + sg + zz + pg
        if adjust_ratio > 0:
            cumulative_factor *= adjust_ratio
        
        factors.append((evt['ex_date'], cumulative_factor))
    
    if not factors:
        return df
    
    # 应用后复权因子
    # 后复权: 除权日之前的价格需要调整(因为现在的价格包含了送转)
    # 最新数据不变，往历史方向乘以累积因子
    # 第i次除权前的数据 × factor[i]
    # 第i次和第i+1次除权之间的数据 × factor[i]
    
    # 从最新到最老排列除权事件
    factors.reverse()  # 现在从新到旧
    
    # 最新的数据(最后一次除权之后)不变
    # 每次跨越一个除权日，价格乘以该事件的因子
    
    # 构建每条数据的复权因子
    df['hfq_factor'] = 1.0
    
    for i, (ex_date, factor) in enumerate(factors):
        # 除权日之前的所有数据，乘以 factor
        mask = df['date'] < ex_date
        # 注意: factors已经reverse, factor是从新到旧累积的
        # 但我们的factor是cumulative的，需要仔细处理
        
        # 重新思考: cumulative_factor 是从第一次除权开始累积的
        # 后复权: 最新价格不变，历史价格需要向上调整
        # 最后一次除权之后: factor = 1 (不变)
        # 最后一次除权之前: factor = 最后一次除权的adjust_ratio
        # 倒数第二次除权之前: factor *= 倒数第二次的adjust_ratio
        # ...
        
        # 所以从新到旧: factor依次递增
        pass
    
    # 重新用更清晰的方法
    # 1. 按时间从旧到新排列除权事件
    # 2. 计算每个事件对"之前"价格的累积调整倍数
    # 3. 后复权: 最新不变，往历史方向每次除权乘以调整比
    
    # 清除之前的临时列
    df = df.drop(columns=['hfq_factor'], errors='ignore')
    
    # 重新计算
    # adjust_ratios: 从旧到新，每个除权事件的调整比
    adjust_events = []
    for evt in events:
        sg = evt['songgu'] / 10.0
        zz = evt['zhuanzeng'] / 10.0
        pg = evt['peigu'] / 10.0
        fh = evt['fenhong'] / 10.0
        adjust_ratio = 1.0 + sg + zz + pg
        if adjust_ratio > 0.01:  # 避免异常值
            adjust_events.append((evt['ex_date'], adjust_ratio, fh))
    
    if not adjust_events:
        return df
    
    # 从最新到最老，累积因子
    # 后复权: 每次遇到除权日，之前的数据要乘以adjust_ratio
    cumulative = 1.0
    df['_hfq_factor'] = 1.0
    
    for ex_date, ratio, fh in reversed(adjust_events):
        cumulative *= ratio
        # 除权日之前的数据应用累积因子
        mask = df['date'] < ex_date
        df.loc[mask, '_hfq_factor'] = cumulative
    
    # 应用因子到OHLC
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col] * df['_hfq_factor']
    # volume: 送转会增加股本，后复权volume也要调整
    # 但习惯上volume不复权，保持原值
    # amount = close * volume, 后复权后 close变了，amount也该变
    df['amount'] = df['amount'] * df['_hfq_factor']
    
    df = df.drop(columns=['_hfq_factor'])
    return df


def _download_pytdx(stock_code, start_date, end_date, adjust='hfq'):
    """用pytdx下载日线数据
    
    pytdx获取的是不复权数据，需要自行计算复权
    
    Args:
        stock_code: 6位代码如 '000001'
        start_date: YYYYMMDD
        end_date: YYYYMMDD
        adjust: 'hfq'后复权 / 'qfq'前复权
    
    Returns:
        DataFrame or None
    """
    from pytdx.hq import TdxHq_API
    
    market, code = _stock_code_to_market(stock_code)
    
    api, _ = _tdx_connect()
    if api is None:
        return None
    
    try:
        # pytdx每次最多获取800条，需要分页
        all_bars = []
        start_pos = 0
        page_size = 800
        
        while True:
            bars = api.get_security_bars(4, market, code, start_pos, page_size)
            if not bars:
                break
            all_bars.extend(bars)
            if len(bars) < page_size:
                break
            start_pos += page_size
        
        if not all_bars:
            return None
        
        df = api.to_df(all_bars)
        # pytdx返回的列: open, close, high, low, vol, amount, year, month, day, hour, minute
        # 需要统一为我们的格式
        df = df.rename(columns={'vol': 'volume'})
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']].copy()
        
        for c in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        
        df = df.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
        
        if len(df) == 0:
            return None
        
        # 日期过滤
        start_ts = pd.Timestamp(start_date) if len(start_date) == 8 else pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date) if len(end_date) == 8 else pd.Timestamp(end_date)
        df = df[(df['date'] >= start_ts) & (df['date'] <= end_ts)].reset_index(drop=True)
        
        # 复权处理
        if adjust in ('hfq', 'qfq'):
            xdxr_df = _tdx_get_xdxr(code, market)
            if not xdxr_df.empty:
                if adjust == 'hfq':
                    df = _tdx_apply_hfq(df, xdxr_df)
                else:  # qfq
                    df = _tdx_apply_qfq(df, xdxr_df)
        
        return df if len(df) >= 120 else None
    
    except Exception:
        return None
    finally:
        _safe_disconnect(api)


def _tdx_apply_qfq(df, xdxr_df):
    """前复权: 最新价格不变，历史价格向下调整"""
    if xdxr_df.empty or df.empty:
        return df
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # 解析除权事件
    events = []
    for _, row in xdxr_df.iterrows():
        try:
            year = int(row.get('year', 0))
            month = int(row.get('month', 0))
            day = int(row.get('day', 0))
            if year == 0:
                continue
            ex_date = pd.Timestamp(year=year, month=month, day=day)
            category = int(row.get('category', 0))
            sg = float(row.get('songgu', 0) or 0) / 10.0
            zz = float(row.get('zhuanzeng', 0) or 0) / 10.0
            pg = float(row.get('peigu', 0) or 0) / 10.0
            fh = float(row.get('fenhong', 0) or 0) / 10.0
            if category in (1, 2):
                ratio = 1.0 + sg + zz + pg
                if ratio > 0.01:
                    events.append((ex_date, ratio, fh))
        except Exception:
            continue
    
    if not events:
        return df
    
    # 前复权: 除权日之后的数据不变，之前的数据除以adjust_ratio
    cumulative = 1.0
    df['_qfq_factor'] = 1.0
    
    for ex_date, ratio, fh in events:  # 从旧到新
        cumulative *= ratio
        mask = df['date'] < ex_date
        df.loc[mask, '_qfq_factor'] = 1.0 / cumulative
    
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col] * df['_qfq_factor']
    df['amount'] = df['amount'] * df['_qfq_factor']
    df = df.drop(columns=['_qfq_factor'])
    return df


# ─── pytdx批量下载(多线程并行, 替代多进程避免内存爆炸) ────────────

def _pytdx_download_thread(codes, cache_dir, wid, result_dict, start_dt, end_dt, adj):
    """pytdx下载线程 - 每个线程独立连接一个通达信服务器
    
    使用threading替代multiprocessing:
    - 共享进程内存，不会spawn子进程重导run.py(避免8×500MB内存爆炸)
    - 不存在pickle/序列化问题
    - pytdx是网络I/O密集型，threading完全胜任(GIL不影响网络等待)
    """
    from pytdx.hq import TdxHq_API
    import random
    
    ok = fail = 0
    api = TdxHq_API()
    connected = False
    
    servers = TDX_SERVERS.copy()
    random.shuffle(servers)
    
    for host, port in servers:
        try:
            if api.connect(host, port, time_out=5):
                connected = True
                break
            # connect返回False: 新socket已创建但连接失败，需关闭
            _safe_disconnect(api)
        except Exception:
            _safe_disconnect(api)
    
    if not connected:
        result_dict[wid] = {'status': 'login_fail', 'ok': 0, 'fail': 0}
        return
    
    xdxr_cache = {}
    
    for code in codes:
        cf = Path(cache_dir) / f"{code}_hfq.parquet"
        if cf.exists():
            ok += 1
            continue
        
        try:
            market, scode = _stock_code_to_market(code)
            
            # 分页获取日线(category=4)
            all_bars = []
            start_pos = 0
            page_size = 800
            
            while True:
                bars = api.get_security_bars(4, market, scode, start_pos, page_size)
                if not bars:
                    break
                all_bars.extend(bars)
                if len(bars) < page_size:
                    break
                start_pos += page_size
            
            if not all_bars:
                fail += 1
                continue
            
            df = api.to_df(all_bars)
            df = df.rename(columns={'vol': 'volume'})
            df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
            df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']].copy()
            for c in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            df = df.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
            
            # 日期过滤
            start_ts = pd.Timestamp(start_dt)
            end_ts = pd.Timestamp(end_dt)
            df = df[(df['date'] >= start_ts) & (df['date'] <= end_ts)].reset_index(drop=True)
            
            if len(df) < 120:
                fail += 1
                continue
            
            # 后复权
            if adj == 'hfq':
                xdxr_key = f'{market}_{scode}'
                if xdxr_key not in xdxr_cache:
                    try:
                        xdxr_data = api.get_xdxr_info(market, scode)
                        xdxr_cache[xdxr_key] = pd.DataFrame(xdxr_data) if xdxr_data else pd.DataFrame()
                    except Exception:
                        xdxr_cache[xdxr_key] = pd.DataFrame()
                
                xdxr_df = xdxr_cache[xdxr_key]
                if not xdxr_df.empty:
                    df = _tdx_apply_hfq(df, xdxr_df)
            
            if len(df) >= 120:
                df.to_parquet(cf, index=False)
                ok += 1
            else:
                fail += 1
            
        except Exception:
            fail += 1
            # 安全重连
            _safe_disconnect(api)
            connected = False
            for host, port in servers:
                try:
                    if api.connect(host, port, time_out=5):
                        connected = True
                        break
                    # connect返回False也需关闭socket
                    _safe_disconnect(api)
                except Exception:
                    _safe_disconnect(api)
            if not connected:
                break
    
    _safe_disconnect(api)
    result_dict[wid] = {'status': 'done', 'ok': ok, 'fail': fail}


def pytdx_download_batch(stock_codes, start_date='20240101', end_date=None, 
                          cache_dir=None, num_connections=8, adjust='hfq'):
    """用pytdx多线程并行下载全A数据
    
    使用threading替代multiprocessing:
    - Windows spawn模式下multiprocessing会重新导入run.py
    - 8个子进程 × sklearn/scipy/lightgbm(~500MB/进程) = 4GB → MemoryError
    - threading共享进程内存，不存在此问题
    
    Args:
        stock_codes: 股票代码列表
        start_date: YYYYMMDD
        end_date: YYYYMMDD or None(auto)
        cache_dir: 缓存目录
        num_connections: 并行线程数(建议4-10)
        adjust: 复权方式
    
    Returns:
        (success_count, fail_count)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    
    if end_date is None:
        end_date = _get_end_date()
    if cache_dir is None:
        cache_dir = BASE_DIR / CONFIG['data']['cache_dir']
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 排除已缓存
    cached = set()
    for f in os.listdir(cache_dir):
        if f.endswith('.parquet'):
            cached.add(f.split('_')[0])
    
    need = [c for c in stock_codes if c not in cached]
    
    if not need:
        print(f"全部 {len(stock_codes)} 只已缓存，无需下载！")
        return len(cached & set(stock_codes)), 0
    
    print(f"总计: {len(stock_codes)}, 已缓存: {len(cached & set(stock_codes))}, 需下载: {len(need)}")
    
    # 分批给每个线程
    chunk_size = len(need) // num_connections + 1
    chunks = [need[i:i + chunk_size] for i in range(0, len(need), chunk_size)]
    print(f"启动 {len(chunks)} 个pytdx连接(线程), 每组约 {chunk_size} 只")
    
    # 共享结果字典(线程安全: 每个线程写不同的key)
    result_dict = {}
    t0 = time.time()
    
    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = {}
        for i, chunk in enumerate(chunks):
            future = executor.submit(_pytdx_download_thread, chunk, str(cache_dir), i, result_dict, start_date, end_date, adjust)
            futures[future] = i
        
        total_ok = total_fail = 0
        for future in as_completed(futures):
            wid = futures[future]
            try:
                future.result()  # 等待完成，触发异常(如有)
            except Exception as e:
                print(f"  Worker {wid} exception: {e}", flush=True)
            
            if wid in result_dict:
                r = result_dict[wid]
                if r['status'] == 'done':
                    total_ok += r['ok']
                    total_fail += r['fail']
                    elapsed = time.time() - t0
                    spd = (total_ok + total_fail) / elapsed if elapsed > 0 else 0
                    print(f"  Worker {wid} done: ok={r['ok']} fail={r['fail']} | Total: {total_ok}+{total_fail}/{len(need)} | {spd:.1f}/s | {elapsed/60:.1f}min", flush=True)
                else:
                    print(f"  Worker {wid} connection failed!", flush=True)
    
    elapsed = time.time() - t0
    print(f"\npytdx下载完成！成功: {total_ok}, 失败: {total_fail}, 耗时: {elapsed/60:.1f}min")
    return total_ok, total_fail


# ─── pytdx: 获取全A股票列表 ──────────────────────────────────────────

def pytdx_get_stock_list():
    """从通达信服务器获取全部A股列表(排除ST/退市/北交所)
    
    Returns:
        dict {code: name}
    """
    from pytdx.hq import TdxHq_API
    
    api, _ = _tdx_connect()
    if api is None:
        return {}
    
    try:
        all_stocks = {}
        
        # 深圳市场(market=0)
        start = 0
        while True:
            stocks = api.get_security_list(0, start)
            if not stocks:
                break
            for s in stocks:
                code = s.get('code', '')
                name = s.get('name', '')
                if len(code) != 6:
                    continue
                if code.startswith(('8', '4', '9')):  # 排除北交所/老三板
                    continue
                if 'ST' in str(name).upper() or '退' in str(name):
                    continue
                all_stocks[code] = name
            start += len(stocks)
            if len(stocks) < 1000:
                break
        
        # 上海市场(market=1)
        start = 0
        while True:
            stocks = api.get_security_list(1, start)
            if not stocks:
                break
            for s in stocks:
                code = s.get('code', '')
                name = s.get('name', '')
                if len(code) != 6:
                    continue
                if code.startswith(('8', '4', '9')):
                    continue
                if 'ST' in str(name).upper() or '退' in str(name):
                    continue
                all_stocks[code] = name
            start += len(stocks)
            if len(stocks) < 1000:
                break
        
        return all_stocks
    
    except Exception:
        return {}
    finally:
        _safe_disconnect(api)


# ─── baostock: 备用数据源 ────────────────────────────────────────────

def _download_baostock(stock_code, start_date, end_date, adjust='hfq'):
    """用baostock下载日线数据(备用)"""
    import baostock as bs
    
    if stock_code.startswith('6'):
        bs_code = f'sh.{stock_code}'
    else:
        bs_code = f'sz.{stock_code}'
    
    bs_start = f'{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}' if len(start_date) == 8 and '-' not in start_date else start_date
    bs_end = f'{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}' if len(end_date) == 8 and '-' not in end_date else end_date
    
    adjustflag = '1' if adjust == 'hfq' else '2'
    
    try:
        lg = bs.login()
        if lg.error_code != '0':
            return None
        
        rs = bs.query_history_k_data_plus(
            bs_code,
            'date,open,high,low,close,volume,amount',
            start_date=bs_start,
            end_date=bs_end,
            frequency='d',
            adjustflag=adjustflag
        )
        data = []
        while (rs.error_code == '0') and rs.next():
            data.append(rs.get_row_data())
        bs.logout()
        
        if not data:
            return None
        
        df = pd.DataFrame(data, columns=rs.fields)
        for c in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
        return df if len(df) > 0 else None
    except Exception:
        try:
            bs.logout()
        except Exception:
            pass
        return None


# ─── efinance: 兜底数据源 ────────────────────────────────────────────

def _download_efinance(stock_code, start_date, end_date, adjust='hfq', max_retries=3):
    """用efinance下载日线数据(兜底)"""
    import efinance as ef
    
    for attempt in range(max_retries):
        try:
            raw = ef.stock.get_quote_history(stock_code, klt=101, fqt=1 if adjust == 'qfq' else 2)
            if raw is not None and len(raw) > 0:
                raw = raw.copy()
                col_map = {
                    '股票名称': 'name', '股票代码': 'code',
                    '日期': 'date', '开盘': 'open', '收盘': 'close',
                    '最高': 'high', '最低': 'low', '成交量': 'volume',
                    '成交额': 'amount', '振幅': 'amplitude',
                    '涨跌幅': 'pct_change', '涨跌额': 'change',
                    '换手率': 'turnover'
                }
                raw.rename(columns=col_map, inplace=True)
                raw['date'] = pd.to_datetime(raw['date'])
                df = raw[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']].copy()
                for c in ['open', 'high', 'low', 'close', 'volume', 'amount']:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                df = df.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
                return df if len(df) > 0 else None
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 + np.random.random() * 3)
    return None


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
    import qlib
    from qlib.config import REG_CN
    from qlib.data import D
    
    qlib_dir = os.environ.get('QLIB_DATA_DIR', None)
    qlib_data_dir = qlib_ensure_data(qlib_dir)
    
    try:
        qlib.init(provider_uri=str(qlib_data_dir), region=REG_CN)
    except Exception:
        pass
    
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


# ─── 统一接口 ────────────────────────────────────────────────────────

def get_stock_data(stock_code, start_date=None, end_date=None, adjust='hfq', cache_dir=None, max_retries=3):
    """获取单只股票日线数据
    
    DATA_SOURCE=qlib: 缓存 → qlib本地数据(不联网)
    默认: 缓存 → pytdx → baostock → efinance
    
    Args:
        stock_code: 股票代码(6位字符串)，如 '000001'
        start_date: 起始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD
        adjust: 复权方式 hfq(后复权)/qfq(前复权)
        cache_dir: 缓存目录
        max_retries: 重试次数
    
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
            
            # 检测旧缓存是否是分钟线(category=7的bug): 日线2024年后约600条，如果>5000说明是分钟线
            df_since_2024 = df[df['date'] >= pd.Timestamp('2024-01-01')]
            if len(df_since_2024) > 5000:
                # 分钟线缓存，删除并重新下载
                import os
                os.remove(cache_file)
            elif len(df) > 0 and (df['date'].iloc[-1] >= pd.Timestamp(end_date) - pd.Timedelta(days=2)):
                mask = (df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= df['date'].iloc[-1])
                return df[mask].reset_index(drop=True)
        except Exception:
            pass
    
    # 根据数据源选择获取方式
    data_source = os.environ.get('DATA_SOURCE', 'pytdx').lower()
    
    if data_source == 'qlib':
        # qlib模式: 从本地qlib数据读取(不联网)
        df = _download_qlib_single(stock_code, start_date, end_date, adjust)
        if df is not None and len(df) >= 120:
            df.to_parquet(cache_file, index=False)
            mask = (df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))
            return df[mask].reset_index(drop=True)
        return pd.DataFrame()
    
    # 默认模式: pytdx → baostock → efinance
    # 1. 尝试pytdx
    df = _download_pytdx(stock_code, start_date, end_date, adjust)
    if df is not None and len(df) >= 120:
        df.to_parquet(cache_file, index=False)
        mask = (df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))
        return df[mask].reset_index(drop=True)
    
    # 2. 尝试baostock
    df = _download_baostock(stock_code, start_date, end_date, adjust)
    if df is not None and len(df) >= 120:
        df.to_parquet(cache_file, index=False)
        mask = (df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))
        return df[mask].reset_index(drop=True)
    
    # 3. 尝试efinance
    df = _download_efinance(stock_code, start_date, end_date, adjust, max_retries)
    if df is not None and len(df) >= 120:
        df.to_parquet(cache_file, index=False)
        mask = (df['date'] >= pd.Timestamp(start_date)) & (df['date'] <= pd.Timestamp(end_date))
        return df[mask].reset_index(drop=True)
    
    return pd.DataFrame()


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


# ─── qlib: 社区维护数据源(GitHub托管, 海外可访问) ────────────────────────

QLIB_DATA_URL = "https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz"
QLIB_DATA_DIR = Path.home() / '.qlib' / 'qlib_data' / 'cn_data'


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
    """从qlib数据获取全A股票列表
    
    Returns:
        dict: {code: name} 如 {'600519': '贵州茅台'}
              如果qlib没有股票名称，name为空字符串
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
        
        print(f"qlib获取全A列表: {len(name_map)} 只")
        return name_map
    except Exception as e:
        print(f"qlib获取股票列表失败: {e}")
        return {}
