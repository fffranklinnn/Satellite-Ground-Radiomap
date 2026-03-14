import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor

# 配置参数
YEAR = 2025
BASE_URL = "https://cddis.nasa.gov/archive/gnss/products/ionex"
SAVE_DIR = "./data/l1_space/data/cddis_data_2025"
MAX_WORKERS = 4  # 网络不稳定时，建议降低并发数，4-5 比较稳妥

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def get_session():
    """创建一个带有自动重试能力的 Session"""
    session = requests.Session()
    # 定义重试策略：如果失败，自动重试 3 次，间隔时间递增
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

def download_file(doy):
    doy_str = f"{doy:03d}"
    file_name = f"UPC0OPSRAP_{YEAR}{doy_str}0000_01D_15M_GIM.INX.gz"
    url = f"{BASE_URL}/{YEAR}/{doy_str}/{file_name}"
    target_path = os.path.join(SAVE_DIR, file_name)
    
    # --- 增加：跳过已成功下载的文件 ---
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
        print(f"⏭️  跳过已存在: {file_name}")
        return

    session = get_session()
    try:
        # 增加 timeout 防止死读，auth 依然依赖你的 .netrc
        with session.get(url, stream=True, timeout=(10, 30)) as r:
            if r.status_code == 200:
                with open(target_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=16384):
                        f.write(chunk)
                print(f"✅ 成功: {file_name}")
            else:
                print(f"❌ 失败: {file_name} (状态码: {r.status_code})")
    except Exception as e:
        print(f"🔥 异常: {file_name} -> {e}")

# 执行下载
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    executor.map(download_file, range(1, 366))
