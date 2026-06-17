# ML主升浪量化策略

用机器学习(LightGBM)挖掘A股主升浪因子的量化策略。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 首次使用

```bash
# Step 1: 训练模型（需标注文件 data/annotations.csv）
python run.py train

# Step 2: 下载全A数据（pytdx直连通达信，8连接并行，约5-10分钟）
python run.py download

# Step 3: 全A扫描（约30分钟，输出TOP50+即将启动30只）
python run.py scan
```

### 3. 日常使用

```bash
# 一键更新（下载新数据 + 扫描）
python run.py update
```

### 4. 其他命令

```bash
# 预测单只股票
python run.py predict 000001

# 更新全A股票列表
python run.py stocklist

# 查看帮助
python run.py
```

## 命令一览

| 命令 | 说明 | 耗时 |
|------|------|------|
| `train` | 用标注数据训练LightGBM模型 | 2-5min |
| `download` | 批量下载全A日线数据(pytdx) | 5-10min(首次) |
| `scan` | 全A扫描，输出主升浪排名 | 30min |
| `update` | 一键更新(download+scan) | 35-40min |
| `predict <代码>` | 预测单只股票 | 1-3s |
| `stocklist` | 更新全A股票列表 | 10-30s |

## 数据源

优先级: **pytdx**(快) → **baostock**(稳) → **efinance**(兜底)

| 数据源 | 速度 | 特点 |
|--------|------|------|
| pytdx | 极快(全A 5-10min) | 直连通达信服务器，无需注册，需自行计算复权 |
| baostock | 慢(全A约40min) | 已复权数据，稳定可靠，每日17:00后更新 |
| efinance | 不稳定 | 东方财富接口，仅作兜底 |

## 目录结构

```
ml_uptrend_strategy/
  run.py                  统一入口（日常只用这个）
  config.json             全局配置
  requirements.txt        Python依赖
  data_loader.py          数据加载(pytdx+baostock+efinance+缓存)
  factor_engine.py        123个技术因子(6大类)
  label_generator.py      标签生成
  model_trainer.py        LightGBM训练
  signal_generator.py     信号生成(概率→买卖)
  backtester.py           简易回测
  data/
    annotations.csv       标注文件(主升浪区间)
    full_a_stocks.csv     全A股票列表
    cache/                训练用数据缓存
    scan_cache/           全A扫描数据缓存(~5000个parquet)
  models/
    uptrend_model_*.txt   LightGBM模型文件
    uptrend_model_*_meta.json  模型元信息
  signals/
    scan_full_a_results.json   扫描结果JSON
    scan_full_a_results.csv    扫描结果CSV
    scan_full_a_report.html    扫描报告HTML
```

## 标注文件格式

`data/annotations.csv` 需包含以下列：

```
stock_code,start_date,end_date,note
002371,2025-09-10,2025-11-20,北方华创
601939,2025-10-08,2025-12-15,建设银行
...
```

- `stock_code`: 6位股票代码
- `start_date`: 主升浪开始日期(YYYY-MM-DD)
- `end_date`: 主升浪结束日期
- `note`: 股票名称/备注

## 配置说明

`config.json` 关键参数：

- `data.source`: 数据源(pytq/baostock/efinance)
- `data.end_date`: 设为 `"auto"` 自动取当天，或指定如 `"20260630"`
- `signal.buy_threshold`: 买入概率阈值(默认0.45)
- `signal.sell_threshold`: 卖出概率阈值(默认0.30)
- `scan.top_n`: 输出TOP N主升浪股票(默认50)
- `scan.download_workers`: 并行连接数(默认8，pytdx推荐8-10)

## 模型信息

- 算法: LightGBM (GBDT)
- 因子: 123个技术因子，6大类(趋势/动量/波动/量价/形态/统计)
- 训练样本: 41只股票真实标注
- AUC: ~0.80
- Top因子: ma120, price_position_120, obv_ma20, close_to_ma120, macd_dea

## 注意事项

1. pytdx直连通达信服务器，国内网络一般都能连上，无需注册
2. 首次download约5-10分钟(pytdx)，后续增量下载更快
3. pytdx下载失败的股票会自动用baostock补充(慢但稳)
4. scan需要先download完成，否则大量股票无数据会失败
5. Windows终端中文可能乱码，HTML报告中文正常，建议看HTML
6. 如遇网络超时，重新运行同一命令即可（有缓存不会重复下载）
7. 复权方式默认后复权(hfq)，与模型训练一致，不建议更改

## GitHub Actions 自动化

项目已配置 GitHub Actions，支持手动触发和定时执行：

- **手动触发**：在 GitHub 仓库 → Actions → 选择 "ML Uptrend Strategy Pipeline" → Run workflow，可选 download/train/scan 步骤
- **定时执行**：每交易日 15:30 CST 自动运行完整流水线
- **结果下载**：信号和模型文件上传为 Artifacts，保留30天

配置文件：`.github/workflows/strategy.yml`
