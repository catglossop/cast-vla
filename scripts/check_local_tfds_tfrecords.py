#!/usr/bin/env python3
"""
Scan a local tensorflow_datasets (or any) directory for corrupted TFRecord files.

Walks the tree for files whose names contain ".tfrecord" and reads every record
with tf.data.TFRecordDataset. TensorFlow errors here match training-time
DataLossError when a shard is truncated or corrupted.

Usage:
  python check_local_tfds_tfrecords.py
  python check_local_tfds_tfrecords.py --root /home/noam/tensorflow_datasets
  python check_local_tfds_tfrecords.py --root /home/noam/tensorflow_datasets \\
      --subdir bridge_dataset
  python check_local_tfds_tfrecords.py --root ... --max-files 5   # smoke test
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _compression_type(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".gz") or ".tfrecord.gz" in name:
        return "GZIP"
    return ""


def iter_tfrecord_paths(scan_root: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(scan_root, followlinks=False):
        for fn in filenames:
            if ".tfrecord" not in fn:
                continue
            p = Path(dirpath) / fn
            if p.is_file():
                out.append(p)
    return sorted(out)


def scan_tfrecord_file(
    path: Path,
    progress_every: int,
) -> tuple[bool, int, str | None]:
    """
    Read all records in one TFRecord file.

    Returns:
        (ok, num_records_read, error_message_or_none)
    """
    import tensorflow as tf

    comp = _compression_type(path)
    ds = tf.data.TFRecordDataset(str(path), compression_type=comp)
    n = 0
    try:
        for _ in ds:
            n += 1
            if progress_every > 0 and n % progress_every == 0:
                print(f"    ... {n} records", flush=True)
        return True, n, None
    except tf.errors.DataLossError as e:
        return False, n, f"DataLossError after {n} good records: {e}"
    except tf.errors.OpError as e:
        return False, n, f"{type(e).__name__} after {n} good records: {e}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/noam/tensorflow_datasets"),
        help="Top-level tensorflow_datasets directory (or any folder containing TFRecords)",
    )
    parser.add_argument(
        "--subdir",
        type=str,
        default="",
        help="If set, only scan root / subdir (e.g. a single dataset name)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="If > 0, stop after scanning this many files (for quick tests)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200_000,
        help="Print progress every N records per file (0 to disable)",
    )
    args = parser.parse_args()

    try:
        import tensorflow as tf  # noqa: F401
    except ImportError:
        print("TensorFlow is required: pip install tensorflow", file=sys.stderr)
        return 2

    scan_root = args.root.expanduser().resolve()
    if args.subdir:
        scan_root = scan_root / args.subdir
    if not scan_root.is_dir():
        print(f"Not a directory: {scan_root}", file=sys.stderr)
        return 2

    paths = iter_tfrecord_paths(scan_root)
    if not paths:
        print(f"No TFRecord files found under {scan_root}")
        return 1

    if args.max_files > 0:
        paths = paths[: args.max_files]

    print(f"Scanning {len(paths)} TFRecord file(s) under {scan_root}\n")

    bad: list[tuple[Path, str]] = []
    total_records = 0

    for i, path in enumerate(paths, start=1):
        try:
            rel = path.relative_to(scan_root)
        except ValueError:
            rel = path
        print(f"[{i}/{len(paths)}] {rel} ...", flush=True)

        ok, n, err = scan_tfrecord_file(path, args.progress_every)
        total_records += n
        if ok:
            print(f"    OK — {n} records", flush=True)
        else:
            assert err is not None
            print(f"    FAIL — {err}", flush=True)
            bad.append((path, err))

    print()
    print(f"Total records read across all files: {total_records}")
    if bad:
        print(f"\n{len(bad)} corrupted / unreadable file(s):")
        for p, msg in bad:
            print(f"  - {p}\n    {msg}")
        return 1

    print("All scanned TFRecord files read successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
