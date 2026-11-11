import json
import platform
import numpy as np
import pandas as pd
import scipy
from .data import ROOT, records, load, sha
from .resampling import transform
from .time_grid import plan
from .analysis import check
from .plots import plot


def run(command):
    manifests = []
    checks = []
    for row in records():
        x = load(row)
        fs = int(row["original_sample_rate_hz"])
        g = plan(len(x), fs)
        dest = ROOT / "data/processed" / (row["id"] + ".npy")
        if command in ("all", "resample"):
            y, g = transform(x, fs)
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.save(dest, y)
        elif command == "audit":
            assert len(x) == int(row["n_samples_available"]) and np.isfinite(x).all()
            continue
        else:
            if not dest.exists():
                raise FileNotFoundError("Run python main.py resample first")
            y = np.load(dest, allow_pickle=False)
        c = check(x, y, g)
        c["id"] = row["id"]
        checks.append(c)
        if not c["passed"]:
            raise ValueError(f"Validation failed: {row['id']}")
        if command in ("all", "figures"):
            out = ROOT / "results/figures"
            out.mkdir(parents=True, exist_ok=True)
            plot(x, y, g, row, out / (row["id"] + ".png"))
        manifests.append(
            dict(
                row,
                **g,
                output_path=dest.relative_to(ROOT).as_posix(),
                input_file_sha256=sha(ROOT / row["data_path"]),
                output_sha256=sha(dest),
                algorithm="scipy.signal.resample",
                axis=0,
                domain="time",
                window="None",
                scipy_version=scipy.__version__,
                numpy_version=np.__version__,
                python_version=platform.python_version(),
                resampling_performed=True,
            )
        )
    if checks:
        out = ROOT / "results/tables"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(checks).to_csv(out / "resampling_checks.csv", index=False)
        pd.DataFrame(manifests).to_csv(out / "resampling_manifest.csv", index=False)
    print(
        json.dumps(
            {"command": command, "records": len(records()), "passed": len(checks)},
            ensure_ascii=False,
        )
    )
