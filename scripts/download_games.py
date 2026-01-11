import os
import io
import urllib.request
import zipfile

from chess_seq.datasets.preprocessing import run_through_folder
from chess_seq import MoveEncoder

URL = "https://www.pgnmentor.com/players/Carlsen.zip"
OUT_DIR = "data/carlsen_games"
OUT_FILE = "magnus_carlsen.pgn"
ENCODER_PATH = "checkpoints/gamba_rossa/id_to_token.json"

encoder = MoveEncoder().load(ENCODER_PATH)

os.makedirs(OUT_DIR, exist_ok=True)

zip_bytes = urllib.request.urlopen(URL).read()

with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
    pgn_name = next(n for n in z.namelist() if n.lower().endswith(".pgn"))
    pgn_bytes = z.read(pgn_name)

out_file_full_path = os.path.join(OUT_DIR, OUT_FILE)

with open(out_file_full_path, "wb") as f:
    f.write(pgn_bytes)

print("Saved:", out_file_full_path)

# Encode to token_ids in npz file
run_through_folder(out_file_full_path, OUT_DIR, encoder)
