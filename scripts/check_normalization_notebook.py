"""Sequential IPython fallback when this host cannot start a socket-based kernel.

Run in a new process. This is NOT a substitute for testing a Jupyter server.
Writes executed notebook outputs only when --write is explicitly supplied.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "module://matplotlib_inline.backend_inline")

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from matplotlib_inline.backend_inline import configure_inline_support


def execute(path, cwd):
    notebook = nbformat.read(path, as_version=4)
    shell = InteractiveShell.instance()
    configure_inline_support(shell, "module://matplotlib_inline.backend_inline")
    previous = Path.cwd()
    count = 0
    try:
        os.chdir(cwd)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            count += 1
            with capture_output(stdout=True, stderr=True, display=True) as captured:
                result = shell.run_cell(cell.source, store_history=True)
            if not result.success:
                raise RuntimeError(
                    f"Cell {index} failed: {captured.stdout}\n{captured.stderr}"
                ) from (result.error_in_exec or result.error_before_exec)
            cell.execution_count = count
            cell.outputs = []
            if captured.stdout:
                cell.outputs.append(
                    nbformat.v4.new_output(
                        "stream", name="stdout", text=captured.stdout
                    )
                )
            if captured.stderr:
                cell.outputs.append(
                    nbformat.v4.new_output(
                        "stream", name="stderr", text=captured.stderr
                    )
                )
            for output in captured.outputs:
                cell.outputs.append(
                    nbformat.v4.new_output(
                        "display_data", data=output.data, metadata=output.metadata
                    )
                )
            print(f"PASS cell {index + 1}", flush=True)
    finally:
        os.chdir(previous)
    notebook.metadata["cos1_validation"] = {
        "execution_mode": "Sequential IPython in a fresh process; external kernel unavailable on validation host",
        "code_cells_executed": count,
    }
    nbformat.validate(notebook)
    return notebook


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--cwd", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks/03_normalization.ipynb"
    result = execute(path, args.cwd or path.parent)
    if args.write:
        nbformat.write(result, path)
    images = sum(
        "image/png" in output.get("data", {})
        for cell in result.cells
        for output in cell.get("outputs", [])
    )
    print(f"Sequential execution PASS; {images} image outputs")
