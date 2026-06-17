"""上传数据到 GitHub Release 供 CI 使用

用法:
    python upload_data.py          # 打包并上传 scan_cache + labels
    python upload_data.py --dry-run # 只打包不上传

流程:
    1. 本地 python run.py download (pytdx下载，快)
    2. python upload_data.py (打包上传到 GitHub Release)
    3. CI 自动从 Release 下载数据

注意: 需要 gh CLI 已登录 (gh auth login)
"""
import os
import sys
import subprocess
import shutil
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).parent
REPO = "oybsu/ml-uptrend-strategy"
RELEASE_TAG = "data-latest"


def run_cmd(cmd, check=True):
    """执行命令并打印输出"""
    print(f"  > {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout.strip():
        print(f"    {result.stdout.strip()}")
    if result.returncode != 0 and check:
        print(f"    ERROR: {result.stderr.strip()}")
        if check:
            sys.exit(1)
    return result


def main():
    dry_run = "--dry-run" in sys.argv

    scan_cache = BASE_DIR / "data" / "scan_cache"
    labels_dir = BASE_DIR / "data" / "labels"

    # 检查数据目录
    if not scan_cache.exists():
        print("ERROR: data/scan_cache 不存在，请先运行 python run.py download")
        sys.exit(1)

    parquet_count = len(list(scan_cache.glob("*.parquet")))
    print(f"scan_cache: {parquet_count} 个 parquet 文件")

    if parquet_count == 0:
        print("ERROR: 没有数据文件，请先运行 python run.py download")
        sys.exit(1)

    # 创建临时目录打包
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. 打包 scan_cache
        print("\n[1/3] 打包 scan_cache...")
        scan_zip = Path(tmpdir) / "scan_cache.zip"
        run_cmd(f'cd "{scan_cache}" && "{sys.executable}" -m zipfile -c "{scan_zip}" *.parquet')
        size_mb = scan_zip.stat().st_size / 1024 / 1024
        print(f"    scan_cache.zip: {size_mb:.1f} MB")

        # 2. 打包 labels (如果存在)
        labels_zip = None
        if labels_dir.exists():
            label_files = list(labels_dir.glob("*"))
            if label_files:
                print("\n[2/3] 打包 labels...")
                labels_zip = Path(tmpdir) / "labels.zip"
                run_cmd(f'cd "{labels_dir}" && "{sys.executable}" -m zipfile -c "{labels_zip}" *')
                size_mb = labels_zip.stat().st_size / 1024 / 1024
                print(f"    labels.zip: {size_mb:.1f} MB")
        else:
            print("\n[2/3] labels 目录不存在，跳过")

        if dry_run:
            print("\n[DRY RUN] 打包完成，跳过上传")
            print(f"  scan_cache.zip: {scan_zip}")
            if labels_zip:
                print(f"  labels.zip: {labels_zip}")
            return

        # 3. 上传到 GitHub Release
        print("\n[3/3] 上传到 GitHub Release...")

        # 检查 gh 是否可用
        gh_result = run_cmd("gh --version", check=False)
        if gh_result.returncode != 0:
            print("ERROR: gh CLI 未安装或未登录")
            print("  安装: https://cli.github.com/")
            print("  登录: gh auth login")
            sys.exit(1)

        # 删除旧的 release（如果存在）
        run_cmd(f'gh release delete {RELEASE_TAG} --repo {REPO} --yes', check=False)
        # 删除旧的 tag
        run_cmd(f'git push origin :refs/tags/{RELEASE_TAG}', check=False)

        # 创建新的 release
        assets = f'"{scan_zip}"'
        if labels_zip:
            assets += f' "{labels_zip}"'

        date_str = subprocess.run(
            "python -c \"import time; print(time.strftime('%Y-%m-%d'))\"",
            shell=True, capture_output=True, text=True
        ).stdout.strip()

        run_cmd(
            f'gh release create {RELEASE_TAG} {assets} '
            f'--repo {REPO} '
            f'--title "数据包 {date_str}" '
            f'--notes "全A日线数据(pytdx下载)，{parquet_count} 只股票，更新于 {date_str}"'
        )

        print(f"\n✓ 上传完成！Release: {RELEASE_TAG}")
        print(f"  CI 将自动从此 Release 下载数据")


if __name__ == "__main__":
    main()
