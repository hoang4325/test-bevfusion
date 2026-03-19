import os
import json
import shutil
import tarfile
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# =========================================================
# CẤU HÌNH - CHỈ CẦN SỬA PHẦN NÀY
# =========================================================
USER_EMAIL = "your_email"
PASSWORD = "your_password"

# Thư mục lưu dataset
OUTPUT_DIR = "/path/to/save"

# Region: "asia" hoặc "us"
REGION = "asia"

# Số file tải song song
MAX_WORKERS = 3

# Thiết lập aria2c cho mỗi file
ARIA2_SPLIT = 8
ARIA2_CONN_PER_SERVER = 8
ARIA2_MIN_SPLIT_SIZE = "16M"

# Tải xong thì giải nén luôn
AUTO_EXTRACT = True

# Giải nén xong thì xóa file nén luôn để tiết kiệm dung lượng
DELETE_ARCHIVE_AFTER_EXTRACT = True

# Chỉ tải trainval để train. Đặt True nếu muốn tải thêm test.
DOWNLOAD_TEST_ARCHIVES = False

# =========================================================
# DANH SÁCH FILE
# =========================================================
ALL_DOWNLOAD_FILES: Dict[str, str] = {
    "v1.0-test_meta.tgz": "b0263f5c41b780a5a10ede2da99539eb",
    "v1.0-test_blobs.tgz": "e065445b6019ecc15c70ad9d99c47b33",
    "v1.0-trainval01_blobs.tgz": "cbf32d2ea6996fc599b32f724e7ce8f2",
    "v1.0-trainval02_blobs.tgz": "aeecea4878ec3831d316b382bb2f72da",
    "v1.0-trainval03_blobs.tgz": "595c29528351060f94c935e3aaf7b995",
    "v1.0-trainval04_blobs.tgz": "b55eae9b4aa786b478858a3fc92fb72d",
    "v1.0-trainval05_blobs.tgz": "1c815ed607a11be7446dcd4ba0e71ed0",
    "v1.0-trainval06_blobs.tgz": "7273eeea36e712be290472859063a678",
    "v1.0-trainval07_blobs.tgz": "46674d2b2b852b7a857d2c9a87fc755f",
    "v1.0-trainval08_blobs.tgz": "37524bd4edee2ab99678909334313adf",
    "v1.0-trainval09_blobs.tgz": "a7fcd6d9c0934e4052005aa0b84615c0",
    "v1.0-trainval10_blobs.tgz": "31e795f2c13f62533c727119b822d739",
    "v1.0-trainval_meta.tgz": "537d3954ec34e5bcb89a35d4f6fb0d4a",
}

if DOWNLOAD_TEST_ARCHIVES:
    DOWNLOAD_FILES = ALL_DOWNLOAD_FILES
else:
    DOWNLOAD_FILES = {
        k: v for k, v in ALL_DOWNLOAD_FILES.items()
        if not k.startswith("v1.0-test")
    }


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "nuscenes-aria2-downloader/1.0"})
    return session


def login(username: str, password: str) -> str:
    session = make_session()
    headers = {
        "Content-Type": "application/x-amz-json-1.1",
        "X-Amz-Target": "AWSCognitoIdentityProviderService.InitiateAuth",
    }
    payload = {
        "AuthFlow": "USER_PASSWORD_AUTH",
        "ClientId": "7fq5jvs5ffs1c50hd3toobb3b9",
        "AuthParameters": {
            "USERNAME": username,
            "PASSWORD": password,
        },
        "ClientMetadata": {},
    }

    resp = session.post(
        "https://cognito-idp.us-east-1.amazonaws.com/",
        headers=headers,
        data=json.dumps(payload),
        timeout=60,
    )
    resp.raise_for_status()

    data = resp.json()
    if "AuthenticationResult" not in data:
        raise RuntimeError(f"Đăng nhập thất bại: {data}")
    return data["AuthenticationResult"]["IdToken"]


def get_signed_url(token: str, filename: str) -> str:
    session = make_session()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    api_url = (
        f"https://o9k5xn5546.execute-api.us-east-1.amazonaws.com/"
        f"v1/archives/v1.0/{filename}?region={REGION}&project=nuScenes"
    )
    resp = session.get(api_url, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()["url"]


def resolve_output_name(url: str, default_name: str) -> str:
    """
    Một số file tên .tgz nhưng server có thể trả về tar.
    """
    session = make_session()
    try:
        resp = session.get(url, stream=True, timeout=60)
        content_type = resp.headers.get("Content-Type", "")
        resp.close()
        if default_name.endswith(".tgz") and content_type == "application/x-tar":
            return default_name[:-4] + ".tar"
    except Exception:
        pass
    return default_name


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_aria2(url: str, out_dir: str, out_name: str, md5: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        "aria2c",
        "--continue=true",
        "--allow-overwrite=false",
        "--auto-file-renaming=false",
        "--file-allocation=none",
        "--check-integrity=true",
        f"--checksum=md5={md5}",
        f"--dir={out_dir}",
        f"--out={out_name}",
        f"--split={ARIA2_SPLIT}",
        f"--max-connection-per-server={ARIA2_CONN_PER_SERVER}",
        f"--min-split-size={ARIA2_MIN_SPLIT_SIZE}",
        "--max-tries=10",
        "--retry-wait=5",
        "--connect-timeout=30",
        "--timeout=60",
        "--summary-interval=10",
        "--console-log-level=warn",
        "--download-result=full",
        url,
    ]

    print(f"[TẢI] {out_name}")
    subprocess.run(cmd, check=True)


def safe_extract_tar(tar: tarfile.TarFile, path: str) -> None:
    base_path = os.path.abspath(path)
    for member in tar.getmembers():
        member_path = os.path.abspath(os.path.join(path, member.name))
        if not member_path.startswith(base_path):
            raise RuntimeError(f"Phát hiện đường dẫn không an toàn trong tar: {member.name}")
    tar.extractall(path)


def extract_archive(path: str) -> None:
    print(f"[GIẢI NÉN] {path}")
    out_dir = os.path.dirname(path)
    with tarfile.open(path, "r:*") as tar:
        safe_extract_tar(tar, out_dir)
    print(f"[OK] Giải nén xong: {path}")

    if DELETE_ARCHIVE_AFTER_EXTRACT:
        os.remove(path)
        print(f"[XÓA ARCHIVE] {path}")


def worker(token: str, filename: str, md5: str) -> Tuple[str, str]:
    default_path = os.path.join(OUTPUT_DIR, filename)

    if os.path.exists(default_path):
        try:
            if md5_file(default_path) == md5:
                return filename, f"[SKIP] {filename} đã có sẵn, md5 đúng."
        except Exception:
            pass

    url = get_signed_url(token, filename)
    resolved_name = resolve_output_name(url, filename)
    resolved_path = os.path.join(OUTPUT_DIR, resolved_name)

    if os.path.exists(resolved_path):
        try:
            if md5_file(resolved_path) == md5:
                if AUTO_EXTRACT:
                    return filename, f"[SKIP] {resolved_name} đã có sẵn, md5 đúng. Có thể giải nén trực tiếp."
                return filename, f"[SKIP] {resolved_name} đã có sẵn, md5 đúng."
        except Exception:
            pass

    run_aria2(url=url, out_dir=OUTPUT_DIR, out_name=resolved_name, md5=md5)

    got_md5 = md5_file(resolved_path)
    if got_md5 != md5:
        raise RuntimeError(
            f"{resolved_name}: md5 sai, expected={md5}, got={got_md5}"
        )

    if AUTO_EXTRACT:
        extract_archive(resolved_path)

    return filename, f"[XONG] {resolved_name}"


def main() -> None:
    if shutil.which("aria2c") is None:
        raise RuntimeError("Không tìm thấy aria2c. Hãy cài aria2 trước.")

    if USER_EMAIL == "your_email" or PASSWORD == "your_password":
        raise RuntimeError("Hãy sửa USER_EMAIL và PASSWORD trong file trước khi chạy.")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("===== CẤU HÌNH HIỆN TẠI =====")
    print(f"OUTPUT_DIR = {OUTPUT_DIR}")
    print(f"REGION = {REGION}")
    print(f"MAX_WORKERS = {MAX_WORKERS}")
    print(f"AUTO_EXTRACT = {AUTO_EXTRACT}")
    print(f"DELETE_ARCHIVE_AFTER_EXTRACT = {DELETE_ARCHIVE_AFTER_EXTRACT}")
    print(f"DOWNLOAD_TEST_ARCHIVES = {DOWNLOAD_TEST_ARCHIVES}")
    print(f"Số file sẽ tải = {len(DOWNLOAD_FILES)}")
    print("=============================\n")

    print("[1/3] Đăng nhập nuScenes...")
    token = login(USER_EMAIL, PASSWORD)

    print(f"[2/3] Bắt đầu tải với {MAX_WORKERS} worker...\n")
    errors = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(worker, token, filename, md5): filename
            for filename, md5 in DOWNLOAD_FILES.items()
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                _, msg = fut.result()
                print(msg)
            except Exception as e:
                errors.append((name, str(e)))
                print(f"[LỖI] {name}: {e}")

    print("\n[3/3] Hoàn tất.")
    if errors:
        print("\nCác file lỗi:")
        for name, err in errors:
            print(f"- {name}: {err}")
        print("\nBạn có thể chạy lại script, aria2 sẽ resume phần đã tải dở.")
    else:
        print("Tất cả file đã tải xong.")


if __name__ == "__main__":
    main()
    
    
# pip install requests
# sudo apt-get update && sudo apt-get install -y aria2
# python download_nuscenes_aria2.py