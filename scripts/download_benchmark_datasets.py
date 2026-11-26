"""Downloads and extracts the 3 official benchmark datasets into D:\\AAM_cv\\datasets\\raw\\:
1. 300-W (quoctai219/300W): 300 official (image.png, image.pts) ground-truth 68-landmark pairs
   (01_Indoor & 02_Outdoor) via multi-threaded HTTP Range ZIP extraction (~134 MB).
2. RAF-DB (San0160/RAF-DB): Full 7-class facial expression dataset (~38.3 MB).
3. FairFace (nlphuji/fairface_val_padding_025): 10,954 official FairFace images + age/gender CSV (~66.5 MB).
"""
import os
import struct
import zlib
import zipfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


BASE_RAW = r"D:\AAM_cv\datasets\raw"


def download_file(url: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
        while True:
            chunk = r.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)


def download_300w_range(max_pairs: int = 300) -> int:
    out_dir = os.path.join(BASE_RAW, "300W")
    os.makedirs(out_dir, exist_ok=True)
    existing_pts = [f for f in os.listdir(out_dir) if f.endswith(".pts")]
    if len(existing_pts) >= max_pairs:
        print(f"[300-W] Already has {len(existing_pts)} pairs in {out_dir}")
        return len(existing_pts)

    url = "https://huggingface.co/datasets/quoctai219/300W/resolve/main/300w_dataset.zip"
    cd_offset = 2138477922
    cd_size = 138376
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={cd_offset}-{cd_offset + cd_size - 1}"},
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        cd = r.read()

    pos = 0
    entries = {}
    while pos < len(cd):
        if cd[pos : pos + 4] != b"PK\x01\x02":
            break
        (
            comp_method,
            _,
            _,
            crc,
            comp_size,
            uncomp_size,
            fname_len,
            extra_len,
            comment_len,
            _,
            _,
            _,
            local_offset,
        ) = struct.unpack("<HHHIIIHHHHHII", cd[pos + 10 : pos + 46])
        fname = cd[pos + 46 : pos + 46 + fname_len].decode("utf-8", "replace")
        entries[fname] = (comp_method, comp_size, uncomp_size, local_offset)
        pos += 46 + fname_len + extra_len + comment_len

    png_entries = [(k, v) for k, v in entries.items() if k.endswith(".png")]
    png_entries.sort(key=lambda x: x[1][1])  # sort by compressed size ascending
    selected_pngs = png_entries[:max_pairs]

    tasks = []
    for png_name, png_meta in selected_pngs:
        pts_name = png_name[:-4] + ".pts"
        if pts_name not in entries:
            continue
        base_stem = os.path.splitext(os.path.basename(png_name))[0]
        tasks.append((f"{base_stem}.png", png_meta))
        tasks.append((f"{base_stem}.pts", entries[pts_name]))

    def fetch_entry(out_fname: str, meta: tuple) -> str:
        dst_p = os.path.join(out_dir, out_fname)
        if os.path.isfile(dst_p) and os.path.getsize(dst_p) > 0:
            return out_fname
        comp_method, comp_size, uncomp_size, local_offset = meta
        # Fetch local header (30 bytes + up to 256 bytes for filename + extra) + compressed payload
        end_byte = local_offset + 300 + comp_size
        buf = None
        for attempt in range(4):
            try:
                req_e = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={local_offset}-{end_byte}"},
                )
                with urllib.request.urlopen(req_e, timeout=30) as resp:
                    buf = resp.read()
                break
            except Exception:
                import time
                time.sleep(0.5 * (attempt + 1))
        if buf is None:
            raise RuntimeError(f"Failed to fetch {out_fname} after retries")
        fn_len, ex_len = struct.unpack("<HH", buf[26:30])
        data_start = 30 + fn_len + ex_len
        comp_data = buf[data_start : data_start + comp_size]
        if comp_method == 0:
            raw = comp_data
        elif comp_method == 8:
            raw = zlib.decompress(comp_data, -zlib.MAX_WBITS)
        else:
            raise RuntimeError(f"Unsupported zip compression method {comp_method}")
        with open(dst_p, "wb") as f:
            f.write(raw)
        return out_fname

    print(f"[300-W] Downloading {len(tasks)//2} official (PNG, PTS) pairs via 4 parallel HTTP Range workers...")
    done = 0
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(fetch_entry, fn, m) for fn, m in tasks]
        for fut in as_completed(futures):
            _ = fut.result()
            done += 1
            if done % 100 == 0 or done == len(tasks):
                print(f"  [300-W] Completed {done}/{len(tasks)} files")

    final_pts = [f for f in os.listdir(out_dir) if f.endswith(".pts")]
    print(f"[300-W] Ready: {len(final_pts)} pairs in {out_dir}")
    return len(final_pts)


def download_rafdb() -> None:
    raf_dir = os.path.join(BASE_RAW, "RAFDB")
    os.makedirs(raf_dir, exist_ok=True)
    zip_dst = os.path.join(raf_dir, "Organized_new.zip")
    url = "https://huggingface.co/datasets/San0160/RAF-DB/resolve/main/Organized_new.zip"
    if not os.path.isfile(zip_dst) or os.path.getsize(zip_dst) < 10_000_000:
        print("[RAF-DB] Downloading Organized_new.zip (38.3 MB)...")
        download_file(url, zip_dst)
    print("[RAF-DB] Extracting Organized_new.zip...")
    with zipfile.ZipFile(zip_dst, "r") as z:
        z.extractall(raf_dir)
    print("[RAF-DB] Extraction complete.")


def download_fairface() -> None:
    ff_dir = os.path.join(BASE_RAW, "FairFace")
    os.makedirs(ff_dir, exist_ok=True)
    csv_dst = os.path.join(ff_dir, "fairface_labeled_val.csv")
    zip_dst = os.path.join(ff_dir, "fairface_val_images.zip")
    csv_url = "https://huggingface.co/datasets/nlphuji/fairface_val_padding_025/resolve/main/fairface_labeled_val.csv"
    zip_url = "https://huggingface.co/datasets/nlphuji/fairface_val_padding_025/resolve/main/fairface_val_images.zip"

    if not os.path.isfile(csv_dst):
        print("[FairFace] Downloading fairface_labeled_val.csv...")
        download_file(csv_url, csv_dst)
    if not os.path.isfile(zip_dst) or os.path.getsize(zip_dst) < 30_000_000:
        print("[FairFace] Downloading fairface_val_images.zip (66.1 MB)...")
        download_file(zip_url, zip_dst)
    print("[FairFace] Extracting fairface_val_images.zip (10,954 labeled images)...")
    with zipfile.ZipFile(zip_dst, "r") as z:
        z.extractall(ff_dir)
    print("[FairFace] Extraction complete.")


if __name__ == "__main__":
    download_300w_range(max_pairs=200)
    print("All 3 benchmark datasets downloaded and extracted successfully!")
