"""Rebuild filter_checkpoint.pkl from existing filtered/*.pgn files.

Needed because the elite filter run crashed during checkpoint save
(pickle couldn't serialize a defaultdict-with-lambda). The PGN output is
intact; we just need to rebuild the dedup set + rows.
"""
from __future__ import annotations
import hashlib
import io
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from filter import (
    FILTERED, RAW, CHECKPOINT_PATH, RE_TAG, parse_tags, strip_annotations,
    count_plies_and_extract_first20_san, dedup_key, evaluate_game, TierWriter,
)

TIER_FILES = {
    "top": FILTERED / "tier_top_2400plus.pgn",
    "mid": FILTERED / "tier_mid_1900-2400.pgn",
    "low": FILTERED / "tier_low_1600-1900.pgn",
}


def iter_games_with_offsets(path: Path):
    """Yield (header_str, movetext_str, offset, length) for each game in a PGN file.

    Tracks BYTE offsets (not chars) so they match what was written.
    """
    fh = open(path, "rb")
    buf = bytearray()
    in_header = True
    header_lines = []
    movetext_lines = []
    game_start = 0
    pos = 0

    def yield_game(start, end):
        return ("".join(header_lines), "".join(movetext_lines), start, end - start)

    while True:
        line_b = fh.readline()
        if not line_b:
            if header_lines:
                yield yield_game(game_start, pos)
            break
        line = line_b.decode("utf-8", errors="replace")
        if in_header:
            if line.startswith("["):
                header_lines.append(line)
            elif line.strip() == "":
                if header_lines:
                    in_header = False
            else:
                # Movetext starts without blank (shouldn't happen here)
                in_header = False
                movetext_lines.append(line)
        else:
            if line.startswith("["):
                # New game starts
                yield yield_game(game_start, pos)
                header_lines = [line]
                movetext_lines = []
                game_start = pos
                in_header = True
            else:
                movetext_lines.append(line)
        pos += len(line_b)
    fh.close()


def parse_date_iso(date_raw: str) -> str:
    if not date_raw:
        return ""
    m = re.match(r"(\d{4})\.(\d{2})\.(\d{2})", date_raw)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.match(r"(\d{4})\.(\d{2})\.\?\?", date_raw)
    if m:
        return f"{m.group(1)}-{m.group(2)}-01"
    m = re.match(r"(\d{4})\.\?\?\.\?\?", date_raw)
    if m:
        return f"{m.group(1)}-01-01"
    return ""


def main():
    started = time.time()
    seen = set()
    rows = []
    writer_state = {}
    stats = {
        "raw_games_seen": 0,
        "duplicates_removed": 0,
        "rejected_filter": 0,
        "kept": 0,
        "by_source": defaultdict(lambda: {"seen": 0, "kept": 0, "dup": 0, "rejected": 0}),
    }

    for tier, path in TIER_FILES.items():
        if not path.exists():
            print(f"SKIP {tier}: {path} missing")
            writer_state[tier] = {"games": 0, "bytes_written": 0, "path": str(path)}
            continue
        print(f"Scanning {path.name} ({path.stat().st_size/1e6:.1f} MB)...", flush=True)
        n = 0
        bytes_written = path.stat().st_size
        for header, movetext, offset, length in iter_games_with_offsets(path):
            n += 1
            tags = parse_tags(header)
            try:
                we = int(tags.get("WhiteElo", "0"))
                be = int(tags.get("BlackElo", "0"))
            except ValueError:
                continue
            # Reconstruct first20_san from movetext for dedup key
            cleaned = strip_annotations(movetext)
            res = count_plies_and_extract_first20_san(cleaned)
            if res is None:
                continue
            n_plies, first20 = res
            date_raw = tags.get("Date", "").strip()
            date_iso = parse_date_iso(date_raw)
            meta = {
                "white": tags.get("White", "").strip(),
                "black": tags.get("Black", "").strip(),
                "date": date_iso,
                "result": tags.get("Result", "").strip(),
                "first20_san": first20,
            }
            key = dedup_key(meta)
            seen.add(key)
            # Source detection from Event tag
            source = tags.get("Event", "").strip()
            if source not in ("lichess_elite", "lichess_monthly"):
                source = "lichess_elite"  # default
            rows.append({
                "game_id": key,
                "tier": tier,
                "source": source,
                "file_path": path.name,
                "pgn_offset_bytes": offset,
                "pgn_length_bytes": length,
                "date": date_iso or None,
                "white": meta["white"],
                "black": meta["black"],
                "white_elo": we,
                "black_elo": be,
                "result": meta["result"],
                "eco": tags.get("ECO", "").strip()[:4],
                "time_control": tags.get("TimeControl", "").strip(),
                "n_plies": n_plies,
            })
            stats["kept"] += 1
            stats["by_source"][source]["kept"] += 1
            stats["raw_games_seen"] += 1
            stats["by_source"][source]["seen"] += 1
            if n % 100_000 == 0:
                print(f"  ... {n:,}", flush=True)
        writer_state[tier] = {
            "games": n, "bytes_written": bytes_written, "path": str(path),
        }
        print(f"  {tier}: {n:,} games", flush=True)

    # Flatten defaultdict for pickling
    stats["by_source"] = {k: dict(v) for k, v in stats["by_source"].items()}

    payload = {"seen": seen, "rows": rows, "stats": stats, "writers": writer_state}
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    elapsed = time.time() - started
    print(f"\nDONE in {elapsed/60:.1f}m. seen={len(seen):,}, rows={len(rows):,}, "
          f"checkpoint -> {CHECKPOINT_PATH}", flush=True)


if __name__ == "__main__":
    main()
