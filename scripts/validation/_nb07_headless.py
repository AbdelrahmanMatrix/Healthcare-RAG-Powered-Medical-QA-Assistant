"""Headless NB07 launcher — runs the BioBERT fine-tuning notebook on GPU.

Executes notebooks/07_classification_model.ipynb via nbclient, streaming output
to scripts/validation/_nb07_run.log. Intended to run as a DETACHED process
across many tool-command windows (training is multi-hour).
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "scripts" / "validation" / "_nb07_run.log"


def log(msg: str) -> None:
    stamp = time.strftime("%H:%M:%S")
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {msg}\n")


def dump_outputs(nb) -> None:
    """Write every code cell's stream/result/error output to the log file."""
    with open(LOG, "a", encoding="utf-8") as f:
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            outs = cell.get("outputs", [])
            if not outs:
                continue
            f.write(f"\n===== CELL [{idx}] outputs =====\n")
            for out in outs:
                otype = out.get("output_type")
                if otype == "stream":
                    f.write(out.get("text", ""))
                elif otype in ("execute_result", "display_data"):
                    txt = out.get("data", {}).get("text/plain", "")
                    if txt:
                        f.write(txt + "\n")
                elif otype == "error":
                    f.write(f"ERROR: {out.get('ename')}: {out.get('evalue')}\n")
                    for line in out.get("traceback", []):
                        f.write(line.rstrip("\n") + "\n")
    log("cell outputs dumped to log")


def main() -> None:
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(str(ROOT / "notebooks" / "07_classification_model.ipynb"), as_version=4)
    # Strip ALL stored outputs first: the file still carries outputs from the
    # pre-repair historical run (stale split sizes/metrics). After this, every
    # output in memory is guaranteed to be from THIS execution.
    stripped = 0
    for cell in nb.cells:
        if cell.cell_type == "code":
            stripped += len(cell.get("outputs", []))
            cell["outputs"] = []
            cell["execution_count"] = None
    log(f"stripped {stripped} stale stored outputs from notebook object")

    # Windows c10.dll load-order fix: torch MUST be imported before pandas/numpy.
    # NB07 cell 2 imports pandas first, which kills torch in a fresh kernel
    # (WinError 1114). Inject an in-memory pre-cell that imports torch first;
    # it is REMOVED again before the notebook is written back, so the stored
    # .ipynb file is never modified by this workaround.
    INJECTION_MARK = "headless launcher: torch imported FIRST"
    import nbformat as _nbf
    pre_cell = _nbf.v4.new_code_cell(
        f"import torch  # ({INJECTION_MARK} - Windows c10.dll load-order fix)"
    )
    nb.cells.insert(0, pre_cell)
    log("injected in-memory torch-first pre-cell (stripped before writeback)")

    client = NotebookClient(
        nb,
        timeout=-1,          # no per-cell timeout; training cells run for hours
        kernel_name="healthcare-rag",  # registered spec -> .venv python (CUDA torch)
        resources={"metadata": {"path": str(ROOT / "notebooks")}},
        allow_errors=True,   # never crash the driver; failures land in the log
    )
    log("starting notebook execution")
    completed = False
    try:
        client.execute()
        completed = True
        log("notebook execution FINISHED")
    except Exception as exc:  # noqa: BLE001
        log(f"notebook execution RAISED: {type(exc).__name__}: {exc}")
        log("partial outputs preserved in notebook object; dumping them now")
    finally:
        dump_outputs(nb)
        # Write the executed notebook (fresh outputs) back over the stale file.
        backup = ROOT / "scripts" / "validation" / "_nb07_stale_backup.ipynb"
        target = ROOT / "notebooks" / "07_classification_model.ipynb"
        if not backup.exists():
            backup.write_bytes(target.read_bytes())
            log(f"backed up original (stale-output) notebook to {backup.name}")
        # Remove the in-memory torch-first pre-cell so the stored .ipynb
        # matches the original source exactly (workaround leaves no trace).
        nb.cells = [
            c for c in nb.cells
            if not (c.cell_type == "code" and INJECTION_MARK in "".join(c.get("source", [])))
        ]
        nbformat.write(nb, str(target))
        log(f"wrote executed notebook back to {target.name} (completed={completed})")

    with open(ROOT / "scripts" / "validation" / "_nb07_done.flag", "w", encoding="utf-8") as f:
        f.write("done\n")
    log("DONE — wrote _nb07_done.flag")


if __name__ == "__main__":
    sys.exit(main())
