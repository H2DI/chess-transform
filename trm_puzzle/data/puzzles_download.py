import requests
import csv
import io
import json
import re

URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"


def stream_csv_from_url(url):
    """Stream-decompress a remote compressed CSV and return a text file-like object.

    Supports .zst (zstandard) and .bz2. Raises RuntimeError with install hints
    if required decompressor isn't available.
    """
    print(f"Opening stream {url}...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    raw = resp.raw

    if url.endswith(".zst"):
        try:
            import zstandard as zstd
        except Exception:
            raise RuntimeError(
                "The file is a .zst stream but the 'zstandard' package is not installed. "
                "Install it with: pip install zstandard"
            )

        dctx = zstd.ZstdDecompressor()
        # stream_reader accepts a file-like raw stream and returns decompressed bytes
        reader = dctx.stream_reader(raw)
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")
        return text_stream

    if url.endswith(".bz2"):
        import bz2

        # bz2.open can read from a file-like object
        return bz2.open(raw, mode="rt")

    # Unknown compression — try to treat as plain text
    return io.TextIOWrapper(raw, encoding="utf-8")


def is_mate_in_1(themes_value, moves_value):
    """Heuristically determine whether a puzzle is mate-in-1.

    We prefer checking the Themes column but fall back to inspecting the moves
    (a single move that results in checkmate or contains #).
    """
    if not themes_value and not moves_value:
        return False

    if themes_value:
        if re.search(r"mate\s*in\s*1|matein1|mateIn1|mate1", themes_value, re.I):
            return True

    if moves_value:
        # If the move list contains a '#' it's a mate; often mate-in-1 puzzles have one move
        if "#" in moves_value:
            # quick check: count moves (split on spaces / dots)
            # Moves could be like 'Qh5#' or '1. Qh5#'
            moves = re.findall(r"\S+", moves_value)
            if len(moves) <= 4:
                return True

    return False


def main(url=URL, out_path="mate_in_1.json"):
    f = stream_csv_from_url(url)
    reader = csv.DictReader(f)

    mate_in_1 = []

    for row in reader:
        # be robust to capitalization of header names
        row_l = {k.lower(): (v or "") for k, v in row.items()}

        themes = row_l.get("themes") or row_l.get("theme")
        moves = row_l.get("moves") or row_l.get("move") or row_l.get("san")

        if is_mate_in_1(themes, moves):
            puzzle = {
                "PuzzleId": row.get("PuzzleId")
                or row.get("puzzleid")
                or row_l.get("puzzleid"),
                "FEN": row.get("FEN") or row.get("fen") or row_l.get("fen"),
                "Moves": row.get("Moves") or row.get("moves") or moves,
            }
            mate_in_1.append(puzzle)

    print(f"Collected {len(mate_in_1)} puzzles")

    with open(out_path, "w") as out:
        json.dump(mate_in_1, out, indent=2)

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print("ERROR:", e)
        print("You can install missing dependencies with: pip install zstandard")
