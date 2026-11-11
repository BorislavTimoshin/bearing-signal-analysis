from pathlib import Path
import hashlib
import io
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records():
    return json.loads((ROOT / "configs/records.json").read_text(encoding="utf-8"))


def load(row):
    p = ROOT / row["data_path"]
    if p.suffix == ".npy":
        if sha(p) != row["sample_sha256"]:
            raise ValueError(f"Source checksum mismatch: {p}")
        return np.load(p, allow_pickle=False)
    if sha(p) != row["original_file_sha256_reported"]:
        raise ValueError("Full source checksum mismatch")
    a = np.loadtxt(io.StringIO(p.read_text().split("Data\n", 1)[1]), delimiter="\t")
    return a[:, {"X": 2, "Y": 3, "Z": 4}[row["channel"]]]
