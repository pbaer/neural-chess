"""Step 1: Download raw PGN sources.

Sources (per data/v2/README.md):
  - Lichess Elite (nikonoel) -- all monthly archives (.zip)
  - Lichess monthly (database.lichess.org) -- 3 most recent (.pgn.zst)
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

UA = "Mozilla/5.0 (compatible; neural-chess-research/1.0; +mailto:pbaer@outlook.com)"
ROOT = Path("D:/dev/neural-chess/data/v2/raw")
PROGRESS_PATH = Path("D:/dev/neural-chess/data/v2/_acquire/download_progress.json")


def _open(url: str, method: str = "GET", timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": UA}, method=method)
    return urllib.request.urlopen(req, timeout=timeout)


def head_size(url: str) -> int | None:
    try:
        with _open(url, method="HEAD", timeout=30) as r:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl else None
    except Exception:
        return None


def download_to(url: str, dest: Path, expected_size: int | None = None) -> tuple[int, str]:
    """Download url -> dest, return (bytes_written, sha256_hex). Skip if already complete.
    Supports HTTP Range-based resume from existing .part files."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        sz = dest.stat().st_size
        if expected_size is not None and sz == expected_size:
            h = hashlib.sha256()
            with open(dest, "rb") as f:
                while True:
                    chunk = f.read(1 << 20)
                    if not chunk:
                        break
                    h.update(chunk)
            return sz, h.hexdigest()
        elif expected_size is None and sz > 0:
            h = hashlib.sha256()
            with open(dest, "rb") as f:
                while True:
                    chunk = f.read(1 << 20)
                    if not chunk:
                        break
                    h.update(chunk)
            return sz, h.hexdigest()
        else:
            dest.unlink()

    tmp = dest.with_suffix(dest.suffix + ".part")
    resume_from = 0
    h = hashlib.sha256()
    if tmp.exists():
        resume_from = tmp.stat().st_size
        if resume_from > 0:
            # Hash existing portion so the final sha256 is valid
            with open(tmp, "rb") as f:
                while True:
                    chunk = f.read(1 << 20)
                    if not chunk:
                        break
                    h.update(chunk)
            print(f"  resuming {dest.name} from {resume_from/1e9:.2f} GB", flush=True)

    headers = {"User-Agent": UA}
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"
    req = urllib.request.Request(url, headers=headers)
    total = resume_from
    last_print = time.time()
    started = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # Range Not Satisfiable -> server has exactly expected_size; we already have it
            print(f"  416 Range Not Satisfiable; treating .part as complete", flush=True)
            tmp.rename(dest)
            return total, h.hexdigest()
        raise
    mode = "ab" if (resume_from > 0 and r.status == 206) else "wb"
    if mode == "wb":
        # Server didn't honor Range; start fresh
        if resume_from > 0:
            print(f"  server did not honor Range; starting fresh", flush=True)
            h = hashlib.sha256()
            total = 0
            resume_from = 0
    with r, open(tmp, mode) as f:
        cl = r.headers.get("Content-Length")
        cl_int = int(cl) if cl else None
        total_size = (cl_int + resume_from) if cl_int and mode == "ab" else cl_int
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
            total += len(chunk)
            now = time.time()
            if now - last_print > 5:
                pct = f"{100 * total / total_size:.1f}%" if total_size else f"{total/1e6:.0f} MB"
                rate = (total - resume_from) / max(now - started, 0.001) / 1e6
                print(f"  ... {dest.name}: {pct} @ {rate:.1f} MB/s", flush=True)
                last_print = now
    tmp.rename(dest)
    return total, h.hexdigest()


def list_elite_archives() -> list[str]:
    with _open("https://database.nikonoel.fr/", timeout=30) as r:
        html = r.read().decode("utf-8", errors="replace")
    links = re.findall(r'https?://[^"\'<>\s]+', html)
    return sorted(set(l for l in links if "lichess_elite" in l.lower() and l.lower().endswith(".zip")))


def list_lichess_monthly(n_recent: int) -> list[str]:
    with _open("https://database.lichess.org/standard/list.txt", timeout=30) as r:
        txt = r.read().decode("utf-8", errors="replace")
    lines = [l.strip() for l in txt.strip().split("\n") if l.strip()]
    return lines[:n_recent]


def load_progress() -> dict:
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text())
    return {"elite": {}, "lichess_monthly": {}}


def save_progress(p: dict) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text(json.dumps(p, indent=2))


def download_elite(progress: dict) -> None:
    print("\n=== Lichess Elite (nikonoel) ===", flush=True)
    urls = list_elite_archives()
    print(f"Found {len(urls)} Elite archives", flush=True)
    for i, url in enumerate(urls, 1):
        name = url.rsplit("/", 1)[-1]
        dest = ROOT / "lichess_elite" / name
        if name in progress["elite"] and dest.exists() and dest.stat().st_size == progress["elite"][name]["size"]:
            print(f"[{i:2}/{len(urls)}] SKIP (cached) {name}", flush=True)
            continue
        print(f"[{i:2}/{len(urls)}] DOWNLOAD {name}", flush=True)
        exp = head_size(url)
        sz, sha = download_to(url, dest, exp)
        progress["elite"][name] = {"url": url, "size": sz, "sha256": sha}
        save_progress(progress)


def download_lichess_monthly(progress: dict, n_recent: int = 3) -> None:
    print(f"\n=== Lichess monthly (last {n_recent}) ===", flush=True)
    urls = list_lichess_monthly(n_recent)
    print(f"Downloading {urls}", flush=True)
    for i, url in enumerate(urls, 1):
        name = url.rsplit("/", 1)[-1]
        dest = ROOT / "lichess_standard" / name
        if name in progress["lichess_monthly"] and dest.exists() and dest.stat().st_size == progress["lichess_monthly"][name]["size"]:
            print(f"[{i}/{len(urls)}] SKIP (cached) {name}", flush=True)
            continue
        print(f"[{i}/{len(urls)}] DOWNLOAD {name}", flush=True)
        exp = head_size(url)
        if exp:
            print(f"  expected size: {exp/1e9:.2f} GB", flush=True)
        sz, sha = download_to(url, dest, exp)
        progress["lichess_monthly"][name] = {"url": url, "size": sz, "sha256": sha}
        save_progress(progress)


def main() -> int:
    progress = load_progress()
    args = sys.argv[1:]
    do_all = not args or "all" in args
    if do_all or "elite" in args:
        download_elite(progress)
    if do_all or "monthly" in args:
        # Pragmatic cap: each monthly is ~29 GB (vs README's ~800MB estimate).
        # 1 month yields ~10M games before sampling — vastly more than the
        # 490K mid+low target, even after Elo filtering.
        download_lichess_monthly(progress, n_recent=1)
    print("\nDONE.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
