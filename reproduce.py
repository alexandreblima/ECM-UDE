"""Portable entry points; original scientific sources and saved inputs stay byte-identical."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

BUNDLE = Path(__file__).resolve().parent
JSON_LOADS = json.loads


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_bundle() -> dict:
    manifest = JSON_LOADS((BUNDLE/"bundle_manifest.json").read_text())
    for relative, expected in manifest["files_sha256"].items():
        path = BUNDLE/relative
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"Missing or modified bundle input: {relative}")
    print(f"SHA-256 OK: {len(manifest['files_sha256'])} packaged files.")
    return manifest


def configure_portability(root: Path, datasets: Path, manifest: dict) -> None:
    original_root = manifest["original_repository"]
    original_data = manifest["original_datasets"]

    def relocate(value):
        if isinstance(value, str):
            if value == original_root or value.startswith(original_root+"/"):
                return str(root)+value[len(original_root):]
            if value == original_data or value.startswith(original_data+"/"):
                return str(datasets)+value[len(original_data):]
            return value
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, dict):
            return {relocate(key): relocate(item) for key, item in value.items()}
        return value

    def portable_loads(*args, **kwargs):
        return relocate(JSON_LOADS(*args, **kwargs))

    json.loads = portable_loads
    sys.path.insert(0, str(root))
    os.chdir(root)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MPLCONFIGDIR", str(root.parent/"matplotlib-cache"))

    import exp_param_ablation

    def verify_protocol():
        expected = manifest["reviewer_protocol_sha256"]
        if digest(root/"reviewer_round_protocol.json") != expected:
            raise RuntimeError("Reviewer protocol differs from preregistered d14935e bytes")

    # Archive exports have no Git object database. Verify the same frozen bytes by SHA-256.
    exp_param_ablation.verify = verify_protocol
    verify_protocol()

    import matplotlib.font_manager as fm
    families = {entry.name for entry in fm.fontManager.ttflist}
    if "Times New Roman" not in families:
        equivalent = next((name for name in ("Liberation Serif", "Nimbus Roman") if name in families), None)
        if equivalent is None:
            raise RuntimeError("Install Times New Roman, Liberation Serif or Nimbus Roman to render IEEE PDFs")
        filename = fm.findfont(equivalent, fallback_to_default=False)
        entry = copy.copy(next(e for e in fm.fontManager.ttflist if e.fname == filename))
        entry.name = "Times New Roman"
        fm.fontManager.ttflist.append(entry)
        fm.fontManager._findfont_cached.cache_clear()
        print(f"Typography: {equivalent}, Times-equivalent; no font redistributed.")
    import torch
    torch.set_num_threads(1)


def validate_science(root: Path) -> None:
    import unittest
    from test_reviewer_round import ReviewerRoundTests
    from figures_ieee import inspect_pdf
    methods = [name for name in unittest.defaultTestLoader.getTestCaseNames(ReviewerRoundTests)
               if name != "test_principal_tables_and_all_previous_bytes"]
    suite = unittest.TestSuite(ReviewerRoundTests(name) for name in methods)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("Scientific checks failed")
    for path in (root/"figures_ieee").glob("*.pdf"):
        manifest = JSON_LOADS((root/"figures_ieee/manifest.json").read_text())
        width = next(e["width_in"] for e in manifest["figures"] if e["file"] == path.name)
        inspect_pdf(path, width)
    print("Scientific checks and eight reference vector PDFs passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "validate", "analysis", "figure2",
                                           "ablation", "rescore", "tables-only"))
    parser.add_argument("--datasets", type=Path, default=BUNDLE/"datasets",
                        help="External datasets directory (never copied into the bundle)")
    parser.add_argument("--work", type=Path, help="New, nonexisting work directory")
    args = parser.parse_args()
    manifest = verify_bundle()
    if args.command == "verify":
        return
    work = (args.work or BUNDLE/"work"/f"{args.command}-{time.time_ns()}").resolve()
    if work.exists():
        raise FileExistsError(f"Choose a fresh work directory: {work}")
    datasets = args.datasets.expanduser().resolve()
    if args.command in ("ablation", "rescore") and not datasets.is_dir():
        raise FileNotFoundError(f"Raw datasets required, see README: {datasets}")
    work.mkdir(parents=True)
    root = work/"code"
    shutil.copytree(BUNDLE/"reference", root)
    if datasets.is_dir():
        (work/"datasets").symlink_to(datasets, target_is_directory=True)
    print(f"Working copy: {root}", flush=True)
    configure_portability(root, datasets, manifest)
    if args.command == "validate":
        validate_science(root)
    elif args.command == "analysis":
        import reviewer_saved_analysis
        reviewer_saved_analysis.main()
    elif args.command == "figure2":
        import figures_ieee
        sys.argv = ["figures_ieee.py", "--figures", "fig02_physics_vs_lstm"]
        figures_ieee.main()
    else:
        import exp_param_ablation
        option = {"ablation": [], "rescore": ["--reuse-fits"], "tables-only": ["--report-only"]}[args.command]
        sys.argv = ["exp_param_ablation.py", *option]
        exp_param_ablation.main()
    (work/"portable_run.json").write_text(json.dumps({
        "command": args.command, "python": sys.version, "datasets": str(datasets),
        "working_code": str(root), "source_commit": manifest["source_commit"],
        "no_neural_training": True, "reference_inputs_modified": False,
        "path_relocation": "In-memory JSON path prefixes only; original input bytes unchanged",
        "protocol_verification": "SHA-256 of exact d14935e preregistration, no Git required",
    }, indent=2)+"\n")
    verify_bundle()
    print(f"Done. Outputs: {root}\nRun metadata: {work/'portable_run.json'}")


if __name__ == "__main__":
    main()
