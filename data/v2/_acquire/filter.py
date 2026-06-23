"""Step 2: Filter, dedup, index.

Streams raw archives in dedup priority order (Elite > Lichess monthly),
applies filters from data/v2/README.md, strips annotations, writes:

  filtered/tier_top_2400plus.pgn
  filtered/tier_mid_1900-2400.pgn
  filtered/tier_low_1600-1900.pgn
  games_index.parquet
  manifest.json
"""
from __future__ import annotations
import hashlib
import io
import json
import os
import random
import re
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import argparse
import pickle

import chess
import chess.pgn
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd

# ----- Paths -----
ROOT = Path("D:/dev/neural-chess/data/v2")
RAW = ROOT / "raw"
FILTERED = ROOT / "filtered"
INDEX_PATH = ROOT / "games_index.parquet"
MANIFEST_PATH = ROOT / "manifest.json"
PROGRESS_PATH = ROOT / "_acquire" / "filter_progress.json"
DOWNLOAD_PROGRESS = ROOT / "_acquire" / "download_progress.json"

# ----- Filter criteria (mirror of README) -----
MIN_INITIAL_TIME_SECONDS = 180
ELO_DIFF_MAX = 400
MIN_PLIES = 10
ALLOWED_TERMINATIONS = {"normal", ""}  # "" treated as normal
TIER_BANDS = {"top": (2400, None), "mid": (1900, 2400), "low": (1600, 1900)}
TARGET_GAMES = {"top": 910_000, "mid": 350_000, "low": 140_000}
# Sampling rate to keep monthly archive processing tractable.
# Each monthly has ~100M games, we need ~150K each from 3 months for mid+low.
# Apply random sampling at game level before deep parsing.
MONTHLY_SAMPLE_RATE = 0.10  # process ~10% of games for filter eval

# ----- Annotation strip regexes -----
RE_COMMENT = re.compile(r"\{[^{}]*\}")  # non-nested {...} (PGN spec; we iterate)
RE_VARIATION = re.compile(r"\([^()]*\)")  # variations (one level at a time)
RE_NAG = re.compile(r"\$\d+")
RE_SUFFIX_NAG = re.compile(r"(?<=\S)([!?]{1,3})(?=\s|$)")  # !, ??, !?, etc. after move


def strip_annotations(movetext: str) -> str:
    """Remove comments {}, variations (), NAGs $N, and suffix annotations !? ??."""
    s = movetext
    # Strip {...} comments (may contain %eval, %clk, etc.); iterate in case of any oddities
    for _ in range(5):
        new = RE_COMMENT.sub("", s)
        if new == s:
            break
        s = new
    # Strip variations (....) iteratively to handle nesting
    for _ in range(10):
        new = RE_VARIATION.sub("", s)
        if new == s:
            break
        s = new
    # Strip $N NAGs
    s = RE_NAG.sub("", s)
    # Strip !, ??, !?, etc.
    s = RE_SUFFIX_NAG.sub("", s)
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_initial_time_seconds(tc: str) -> int | None:
    """Lichess TimeControl is e.g. '600+0', '180+2', '-' for correspondence."""
    if not tc or tc == "-":
        return None
    # Could be '40/9000+30' for tournament; take first segment
    m = re.match(r"(?:\d+/)?(\d+)(?:\+\d+)?", tc.strip())
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


# ----- Tag parser -----
RE_TAG = re.compile(r'^\[([A-Za-z0-9_]+)\s+"((?:[^"\\]|\\.)*)"\]\s*$')


def parse_tags(header: str) -> dict[str, str]:
    tags = {}
    for line in header.split("\n"):
        line = line.strip()
        if not line.startswith("["):
            continue
        m = RE_TAG.match(line)
        if m:
            tags[m.group(1)] = m.group(2).replace('\\"', '"').replace("\\\\", "\\")
    return tags


# ----- PGN stream reader -----
def iter_pgn_games_from_text(stream: io.TextIOBase) -> Iterator[tuple[str, str]]:
    """Yield (header_block, movetext_block) for each game in the text stream.

    Uses the standard PGN structure: header lines start with '[', followed by a
    blank line, then movetext (possibly multi-line), then a blank line.
    """
    header_lines: list[str] = []
    movetext_lines: list[str] = []
    in_header = True
    line = stream.readline()
    while line:
        if in_header:
            if line.startswith("["):
                header_lines.append(line)
                line = stream.readline()
                continue
            if line.strip() == "":
                if header_lines:
                    in_header = False
                line = stream.readline()
                continue
            # Movetext begins (no blank line). Push as movetext.
            in_header = False
            movetext_lines.append(line)
            line = stream.readline()
            continue
        else:
            if line.startswith("["):
                # Start of next game; yield current
                if header_lines:
                    yield ("".join(header_lines), "".join(movetext_lines))
                header_lines = [line]
                movetext_lines = []
                in_header = True
                line = stream.readline()
                continue
            movetext_lines.append(line)
            line = stream.readline()
    if header_lines:
        yield ("".join(header_lines), "".join(movetext_lines))


# ----- Source iterators -----
def open_zst_text(path: Path) -> io.TextIOWrapper:
    dctx = zstd.ZstdDecompressor(max_window_size=2**31)
    fh = open(path, "rb")
    rdr = dctx.stream_reader(fh, read_size=1 << 20)
    return io.TextIOWrapper(rdr, encoding="utf-8", errors="replace", newline="\n")


def iter_lichess_elite(zip_path: Path) -> Iterator[tuple[str, str, str]]:
    """Yield (header, movetext, source_file_basename) from a Lichess Elite zip."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if not info.filename.lower().endswith(".pgn"):
                continue
            with zf.open(info) as fh_bin:
                tr = io.TextIOWrapper(fh_bin, encoding="utf-8", errors="replace", newline="\n")
                for header, movetext in iter_pgn_games_from_text(tr):
                    yield header, movetext, zip_path.name


def iter_lichess_monthly(zst_path: Path) -> Iterator[tuple[str, str, str]]:
    tr = open_zst_text(zst_path)
    try:
        for header, movetext in iter_pgn_games_from_text(tr):
            yield header, movetext, zst_path.name
    finally:
        try:
            tr.close()
        except Exception:
            pass


# ----- Filter -----
def classify_tier(white_elo: int, black_elo: int) -> str | None:
    weaker = min(white_elo, black_elo)
    if weaker >= 2400:
        return "top"
    if weaker >= 1900:
        return "mid"
    if weaker >= 1600:
        return "low"
    return None


def is_acceptable_termination(tags: dict[str, str]) -> bool:
    term = tags.get("Termination", "").strip().lower()
    if term in ALLOWED_TERMINATIONS:
        return True
    return False


def is_standard_variant(tags: dict[str, str]) -> bool:
    v = tags.get("Variant", "").strip().lower()
    return v in ("", "standard")


RE_MOVE_TOKEN = re.compile(
    r"(?<!\w)(?:O-O-O|O-O|"  # castles
    r"[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=?[QRBNqrbn])?[+#]?"  # piece/pawn move
    r")"
)
RE_MOVE_NUM = re.compile(r"\d+\.+")
RE_RESULT = re.compile(r"(?:1-0|0-1|1/2-1/2|\*)")


def count_plies_and_extract_first20_san(cleaned_movetext: str) -> tuple[int, str] | None:
    """Cheap ply count + first-20-plies-SAN extraction without python-chess.

    Tokenizes SAN moves directly from cleaned movetext. Sufficient for both
    the min-plies filter and the dedup key (same game text -> same SAN sequence).
    """
    # Strip move numbers and result markers, then extract move tokens
    s = RE_RESULT.sub(" ", cleaned_movetext)
    s = RE_MOVE_NUM.sub(" ", s)
    # Tokenize by whitespace, then filter to valid-looking SAN
    tokens = s.split()
    moves = [t for t in tokens if RE_MOVE_TOKEN.fullmatch(t)]
    if not moves:
        return None
    return len(moves), " ".join(moves[:20])


def evaluate_game(tags: dict[str, str], movetext: str) -> dict | None:
    """Apply all filters; return dict of metadata if game passes, else None."""
    # Both Elos
    try:
        white_elo = int(tags.get("WhiteElo", "").strip())
        black_elo = int(tags.get("BlackElo", "").strip())
    except (ValueError, TypeError):
        return None
    if white_elo <= 0 or black_elo <= 0:
        return None
    # Elo diff
    if abs(white_elo - black_elo) > ELO_DIFF_MAX:
        return None
    # Result
    result = tags.get("Result", "*").strip()
    if result not in {"1-0", "0-1", "1/2-1/2"}:
        return None
    # Variant
    if not is_standard_variant(tags):
        return None
    # Termination
    if not is_acceptable_termination(tags):
        return None
    # Time control: only enforce when present (some games lack the tag)
    tc = tags.get("TimeControl", "").strip()
    if tc:
        secs = parse_initial_time_seconds(tc)
        if secs is not None and secs < MIN_INITIAL_TIME_SECONDS:
            return None
    # Tier
    tier = classify_tier(white_elo, black_elo)
    if tier is None:
        return None
    # Strip annotations + count plies
    cleaned = strip_annotations(movetext)
    res = count_plies_and_extract_first20_san(cleaned)
    if res is None:
        return None
    n_plies, first20 = res
    if n_plies < MIN_PLIES:
        return None
    # Date
    date_raw = tags.get("Date", tags.get("UTCDate", "")).strip()
    date_iso = ""
    m = re.match(r"(\d{4})\.(\d{2})\.(\d{2})", date_raw)
    if m:
        date_iso = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    else:
        m = re.match(r"(\d{4})\.(\d{2})\.\?\?", date_raw)
        if m:
            date_iso = f"{m.group(1)}-{m.group(2)}-01"
        else:
            m = re.match(r"(\d{4})\.\?\?\.\?\?", date_raw)
            if m:
                date_iso = f"{m.group(1)}-01-01"
    return {
        "tier": tier,
        "white": tags.get("White", "").strip(),
        "black": tags.get("Black", "").strip(),
        "white_elo": white_elo,
        "black_elo": black_elo,
        "result": result,
        "date": date_iso,
        "date_raw": date_raw,
        "eco": tags.get("ECO", "").strip()[:4],
        "time_control": tc,
        "n_plies": n_plies,
        "first20_san": first20,
        "cleaned_movetext": cleaned,
    }


def dedup_key(meta: dict) -> str:
    s = f"{meta['white']}|{meta['black']}|{meta['date']}|{meta['result']}|{meta['first20_san']}"
    return hashlib.sha1(s.encode("utf-8", errors="replace")).hexdigest()


# ----- Output writers -----
class TierWriter:
    def __init__(self, name: str, path: Path):
        self.name = name
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "wb")
        self.bytes_written = 0
        self.games = 0

    def write_game(self, header_lines: list[str], movetext: str) -> tuple[int, int]:
        """Write game; return (offset, length) in bytes."""
        offset = self.bytes_written
        out = ""
        for line in header_lines:
            out += line if line.endswith("\n") else line + "\n"
        out += "\n"
        out += movetext.rstrip() + "\n\n"
        data = out.encode("utf-8", errors="replace")
        self.fh.write(data)
        self.bytes_written += len(data)
        self.games += 1
        return offset, len(data)

    def close(self):
        self.fh.close()


def build_header_lines(meta: dict, source: str) -> list[str]:
    """Build a minimal seven-tag-roster header for the cleaned game."""
    return [
        f'[Event "{source}"]',
        f'[Site "?"]',
        f'[Date "{meta["date_raw"] or meta["date"]}"]',
        f'[Round "?"]',
        f'[White "{meta["white"]}"]',
        f'[Black "{meta["black"]}"]',
        f'[Result "{meta["result"]}"]',
        f'[WhiteElo "{meta["white_elo"]}"]',
        f'[BlackElo "{meta["black_elo"]}"]',
        f'[ECO "{meta["eco"]}"]',
        f'[TimeControl "{meta["time_control"]}"]',
    ]


# ----- Manifest helpers -----
def load_download_progress() -> dict:
    if DOWNLOAD_PROGRESS.exists():
        return json.loads(DOWNLOAD_PROGRESS.read_text())
    return {"elite": {}, "lichess_monthly": {}}


CHECKPOINT_PATH = ROOT / "_acquire" / "filter_checkpoint.pkl"


def save_checkpoint(seen: set, rows: list, stats: dict, writers: dict) -> None:
    """Persist dedup state + accumulated rows so a second phase can resume."""
    # Flush writers so byte sizes/offsets are accurate
    for w in writers.values():
        w.fh.flush()
    # defaultdict-with-lambda is not picklable; flatten to plain dict
    stats_serializable = dict(stats)
    bs = stats_serializable.get("by_source", {})
    stats_serializable["by_source"] = {k: dict(v) for k, v in bs.items()}
    payload = {
        "seen": seen,
        "rows": rows,
        "stats": stats_serializable,
        "writers": {
            t: {"games": w.games, "bytes_written": w.bytes_written, "path": str(w.path)}
            for t, w in writers.items()
        },
    }
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Checkpoint saved -> {CHECKPOINT_PATH} (seen={len(seen):,}, rows={len(rows):,})", flush=True)


def load_checkpoint() -> tuple[set, list, dict, dict] | None:
    if not CHECKPOINT_PATH.exists():
        return None
    with open(CHECKPOINT_PATH, "rb") as f:
        p = pickle.load(f)
    print(f"Checkpoint loaded: seen={len(p['seen']):,}, rows={len(p['rows']):,}", flush=True)
    return p["seen"], p["rows"], p["stats"], p["writers"]


# ----- Main pipeline -----
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", default="all",
        choices=["all", "elite", "monthly"],
        help="Run all phases (default), or just Elite (writes checkpoint), "
             "or just monthly (loads checkpoint).",
    )
    args = parser.parse_args()

    if not RAW.exists():
        print("ERROR: raw/ directory missing")
        return 1

    FILTERED.mkdir(parents=True, exist_ok=True)

    if args.phase == "monthly":
        cp = load_checkpoint()
        if cp is None:
            print("ERROR: --phase monthly requires checkpoint from prior --phase elite")
            return 1
        seen, rows, stats, writer_state = cp
        # Reopen writers in append mode at the recorded byte offset
        writers = {}
        for t, st in writer_state.items():
            path = Path(st["path"])
            w = TierWriter.__new__(TierWriter)
            w.name = t
            w.path = path
            w.fh = open(path, "ab")
            w.bytes_written = st["bytes_written"]
            w.games = st["games"]
            writers[t] = w
        # Convert stats by_source back to defaultdict
        bs = defaultdict(lambda: {"seen": 0, "kept": 0, "dup": 0, "rejected": 0})
        for k, v in stats.get("by_source", {}).items():
            bs[k] = v
        stats["by_source"] = bs
    else:
        writers = {
            "top": TierWriter("top", FILTERED / "tier_top_2400plus.pgn"),
            "mid": TierWriter("mid", FILTERED / "tier_mid_1900-2400.pgn"),
            "low": TierWriter("low", FILTERED / "tier_low_1600-1900.pgn"),
        }
        rows = []
        seen = set()
        stats = {
            "raw_games_seen": 0,
            "duplicates_removed": 0,
            "rejected_filter": 0,
            "kept": 0,
            "by_source": defaultdict(lambda: {"seen": 0, "kept": 0, "dup": 0, "rejected": 0}),
        }

    rng = random.Random(20260516)
    started = time.time()
    last_print = started

    do_elite = args.phase in ("all", "elite")
    do_monthly = args.phase in ("all", "monthly")

    # Per-tier overshoot caps (applies across all phases).
    # Spec says: "If a source over-delivers, we down-sample to fit, preferring
    # more recent dates and higher Elo within tier." We process archives newest-first
    # so the natural "stop at overshoot" behavior keeps recent games.
    target_with_overshoot = {t: int(TARGET_GAMES[t] * 1.1) for t in TARGET_GAMES}

    if not do_elite:
        elite_files = []
    else:
        # ---- Phase 1: Lichess Elite (highest priority) ----
        # Process newest first to preserve recent games on tier overshoot.
        elite_dir = RAW / "lichess_elite"
        elite_files = sorted(elite_dir.glob("*.zip"), reverse=True) if elite_dir.exists() else []
        print(f"\n=== Phase 1: Lichess Elite ({len(elite_files)} archives, newest-first) ===", flush=True)
        print(f"  per-tier cap (110% of target): {target_with_overshoot}", flush=True)
    for fp in elite_files:
        print(f"-- {fp.name}", flush=True)
        # Skip whole archive if all 3 tiers full (early exit for speed)
        if all(writers[t].games >= target_with_overshoot[t] for t in writers):
            print(f"  all tiers at cap, skipping remaining Elite", flush=True)
            break
        for header, movetext, src in iter_lichess_elite(fp):
            stats["raw_games_seen"] += 1
            stats["by_source"]["lichess_elite"]["seen"] += 1
            tags = parse_tags(header)
            meta = evaluate_game(tags, movetext)
            if meta is None:
                stats["rejected_filter"] += 1
                stats["by_source"]["lichess_elite"]["rejected"] += 1
                continue
            # Per-tier cap
            if writers[meta["tier"]].games >= target_with_overshoot[meta["tier"]]:
                continue
            key = dedup_key(meta)
            if key in seen:
                stats["duplicates_removed"] += 1
                stats["by_source"]["lichess_elite"]["dup"] += 1
            else:
                seen.add(key)
                w = writers[meta["tier"]]
                hlines = build_header_lines(meta, "lichess_elite")
                offset, length = w.write_game(hlines, meta["cleaned_movetext"])
                rows.append({
                    "game_id": key,
                    "tier": meta["tier"],
                    "source": "lichess_elite",
                    "file_path": w.path.name,
                    "pgn_offset_bytes": offset,
                    "pgn_length_bytes": length,
                    "date": meta["date"] or None,
                    "white": meta["white"],
                    "black": meta["black"],
                    "white_elo": meta["white_elo"],
                    "black_elo": meta["black_elo"],
                    "result": meta["result"],
                    "eco": meta["eco"],
                    "time_control": meta["time_control"],
                    "n_plies": meta["n_plies"],
                })
                stats["kept"] += 1
                stats["by_source"]["lichess_elite"]["kept"] += 1
            now = time.time()
            if now - last_print > 15:
                _print_progress(stats, now - started)
                last_print = now
        _print_progress(stats, time.time() - started)

    if args.phase == "elite":
        # Save checkpoint and exit before monthly phase
        for w in writers.values():
            w.fh.flush()
        save_checkpoint(seen, rows, stats, writers)
        # Close writers (will be reopened on monthly run)
        for w in writers.values():
            w.close()
        elapsed = time.time() - started
        print(f"\nElite phase DONE in {elapsed/60:.1f} min. "
              f"Kept {stats['kept']:,} games. Checkpoint saved.", flush=True)
        return 0

    if not do_monthly:
        monthly_files = []
    else:
        # ---- Phase 2: Lichess monthly (only need mid + low; newest-first) ----
        monthly_dir = RAW / "lichess_standard"
        monthly_files = sorted(monthly_dir.glob("*.pgn.zst"), reverse=True) if monthly_dir.exists() else []
        print(f"\n=== Phase 2: Lichess monthly ({len(monthly_files)} archives, newest-first) ===", flush=True)
    # Track how many mid/low we have; stop early once both exceed (target * 1.1)
    target_with_overshoot = {t: int(TARGET_GAMES[t] * 1.1) for t in TARGET_GAMES}
    for fp in monthly_files:
        print(f"-- {fp.name} (sample rate {MONTHLY_SAMPLE_RATE})", flush=True)
        for header, movetext, src in iter_lichess_monthly(fp):
            stats["raw_games_seen"] += 1
            stats["by_source"]["lichess_monthly"]["seen"] += 1
            # Cheap pre-filter using tags only (no python-chess parse) for speed
            tags = parse_tags(header)
            # Sampling rate: skip most games for speed
            if rng.random() > MONTHLY_SAMPLE_RATE:
                continue
            meta = evaluate_game(tags, movetext)
            if meta is None:
                stats["rejected_filter"] += 1
                stats["by_source"]["lichess_monthly"]["rejected"] += 1
                continue
            # Only keep mid/low from monthly (top covered by Elite)
            if meta["tier"] == "top":
                continue
            # Early stop: if this tier has overshoot, skip further additions
            tier_writer = writers[meta["tier"]]
            if tier_writer.games >= target_with_overshoot[meta["tier"]]:
                continue
            key = dedup_key(meta)
            if key in seen:
                stats["duplicates_removed"] += 1
                stats["by_source"]["lichess_monthly"]["dup"] += 1
                continue
            seen.add(key)
            hlines = build_header_lines(meta, "lichess_monthly")
            offset, length = tier_writer.write_game(hlines, meta["cleaned_movetext"])
            rows.append({
                "game_id": key,
                "tier": meta["tier"],
                "source": "lichess_monthly",
                "file_path": tier_writer.path.name,
                "pgn_offset_bytes": offset,
                "pgn_length_bytes": length,
                "date": meta["date"] or None,
                "white": meta["white"],
                "black": meta["black"],
                "white_elo": meta["white_elo"],
                "black_elo": meta["black_elo"],
                "result": meta["result"],
                "eco": meta["eco"],
                "time_control": meta["time_control"],
                "n_plies": meta["n_plies"],
            })
            stats["kept"] += 1
            stats["by_source"]["lichess_monthly"]["kept"] += 1
            now = time.time()
            if now - last_print > 15:
                _print_progress(stats, now - started)
                last_print = now
            # Check stop condition (whole job done?)
            if (writers["mid"].games >= target_with_overshoot["mid"]
                    and writers["low"].games >= target_with_overshoot["low"]):
                print("  Both mid and low tiers at target+10%; stopping monthly", flush=True)
                break
        _print_progress(stats, time.time() - started)
        if (writers["mid"].games >= target_with_overshoot["mid"]
                and writers["low"].games >= target_with_overshoot["low"]):
            break

    # ---- Close writers ----
    for w in writers.values():
        w.close()

    # ---- Write parquet index ----
    print(f"\nWriting {len(rows)} rows -> {INDEX_PATH}", flush=True)
    schema = pa.schema([
        ("game_id", pa.string()),
        ("tier", pa.string()),
        ("source", pa.string()),
        ("file_path", pa.string()),
        ("pgn_offset_bytes", pa.int64()),
        ("pgn_length_bytes", pa.int32()),
        ("date", pa.string()),
        ("white", pa.string()),
        ("black", pa.string()),
        ("white_elo", pa.int16()),
        ("black_elo", pa.int16()),
        ("result", pa.string()),
        ("eco", pa.string()),
        ("time_control", pa.string()),
        ("n_plies", pa.int16()),
    ])
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, INDEX_PATH, compression="zstd")

    # ---- Write manifest ----
    write_manifest(stats, writers)

    elapsed = time.time() - started
    print(f"\nDONE in {elapsed/60:.1f} min. Kept {stats['kept']:,} games "
          f"(top={writers['top'].games:,}, mid={writers['mid'].games:,}, low={writers['low'].games:,})", flush=True)
    return 0


def _print_progress(stats: dict, elapsed_s: float) -> None:
    rate = stats["raw_games_seen"] / max(elapsed_s, 0.001)
    by = stats["by_source"]
    print(f"  [{elapsed_s/60:5.1f}m] seen={stats['raw_games_seen']:>10,} "
          f"kept={stats['kept']:>8,} rej={stats['rejected_filter']:>10,} "
          f"dup={stats['duplicates_removed']:>7,}  rate={rate:>6.0f} g/s | "
          f"elite kept={by['lichess_elite']['kept']:,} | "
          f"monthly kept={by['lichess_monthly']['kept']:,}", flush=True)


def write_manifest(stats: dict, writers: dict) -> None:
    dp = load_download_progress()
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                             cwd=str(ROOT.parent.parent)).decode().strip()
    except Exception:
        git_commit = "unknown"
    # Build sources block
    sources = []
    for src_key, raw_dir, license_str, url in [
        ("lichess_elite", RAW / "lichess_elite", "CC0",
         "https://database.nikonoel.fr"),
        ("lichess_monthly", RAW / "lichess_standard", "CC0",
         "https://database.lichess.org/standard/"),
    ]:
        prog_key = {"lichess_elite": "elite",
                    "lichess_monthly": "lichess_monthly"}[src_key]
        files_meta = []
        total_size = 0
        for name, info in sorted(dp.get(prog_key, {}).items()):
            files_meta.append({"name": name, "sha256": info["sha256"], "size_bytes": info["size"]})
            total_size += info["size"]
        sources.append({
            "name": src_key,
            "url": url,
            "license": license_str,
            "raw_files": files_meta,
            "raw_files_count": len(files_meta),
            "raw_size_bytes": total_size,
            "filter_stats": {
                "seen": stats["by_source"][src_key]["seen"],
                "kept": stats["by_source"][src_key]["kept"],
                "duplicates_removed": stats["by_source"][src_key]["dup"],
                "rejected_filter": stats["by_source"][src_key]["rejected"],
            },
        })

    manifest = {
        "spec_version": "v2.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "sources": sources,
        "filter_criteria": {
            "min_initial_time_seconds": MIN_INITIAL_TIME_SECONDS,
            "elo_diff_max": ELO_DIFF_MAX,
            "min_plies": MIN_PLIES,
            "allowed_terminations": ["Normal", ""],
            "tier_bands": {
                "top": [2400, None],
                "mid": [1900, 2400],
                "low": [1600, 1900],
            },
            "monthly_sample_rate": MONTHLY_SAMPLE_RATE,
            "annotations_stripped": ["{...}", "(...)", "$N", "!", "?", "!!", "??", "!?", "?!"],
        },
        "tiers": {
            t: {
                "games": writers[t].games,
                "size_bytes": writers[t].path.stat().st_size if writers[t].path.exists() else 0,
                "file_path": str(writers[t].path.relative_to(ROOT)),
            } for t in writers
        },
        "totals": {
            "games": sum(w.games for w in writers.values()),
            "size_bytes": sum(w.path.stat().st_size for w in writers.values() if w.path.exists()),
            "approx_positions_if_all_moves": "filled in by validation step",
        },
        "dedup_stats": {
            "raw_games_seen": stats["raw_games_seen"],
            "duplicates_removed": stats["duplicates_removed"],
            "rejected_filter": stats["rejected_filter"],
            "kept": stats["kept"],
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest -> {MANIFEST_PATH}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
