#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch SMPL fitting for Van Criekinge dataset using human_body_prior.
Parallelizes processing of multiple trials across subjects.
"""

import os
import argparse
import glob
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

# Import core fitting function from fit_markers
try:
    from .fit_markers import run_fitting, sanitize
except ImportError:
    from fit_markers import run_fitting, sanitize


def parse_subj_selection(selection_str):
    """
    Parses subject selection string.
    Supported formats: 'subj20-50', '20-50', '01,06,30', 'subj01,subj06'
    """
    if not selection_str:
        return None

    selected_nums = set()
    parts = [p.strip() for p in selection_str.split(",")]

    for part in parts:
        range_match = re.search(r"(\d+)-(\d+)", part)
        if range_match:
            start, end = map(int, range_match.groups())
            selected_nums.update(range(start, end + 1))
        else:
            num_match = re.search(r"\d+", part)
            if num_match:
                selected_nums.add(int(num_match.group()))

    return selected_nums


def get_subj_num(subj_name):
    """Extracts numeric index from subject name (e.g., 'SUBJ01' -> 1)"""
    match = re.search(r"\d+", subj_name)
    return int(match.group()) if match else None


def worker_task(task_args):
    """Task function for ProcessPoolExecutor"""
    input_path, output_path, models_dir, device, batch_size, vicon_up, vicon_forward = (
        task_args
    )
    try:
        run_fitting(
            input_path=input_path,
            output_path=output_path,
            models_dir=models_dir,
            device=device,
            batch_size=batch_size,
            vicon_up=vicon_up,
            vicon_forward=vicon_forward,
        )
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch fit SMPL parameters to Van Criekinge markers using parallel workers",
    )

    parser.add_argument(
        "--processed_dir",
        required=True,
        help="Path to processed markers (typically data/processed_markers_all_2/)",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for fitted SMPL parameters",
    )
    parser.add_argument(
        "--models_dir",
        required=True,
        help="Path to SMPL models directory (typically data/smpl/)",
    )
    parser.add_argument(
        "--subjects", help="Subject range to process (e.g., 'subj20-50' or '1-10')"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8, each needs ~1.5GB RAM)",
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cpu", "cuda"], help="Device to use"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Maximum frames per batch"
    )
    parser.add_argument("--vicon_up", default="Z", choices=["X", "Y", "Z"])
    parser.add_argument("--vicon_forward", default="Y", choices=["X", "Y", "Z"])

    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Filter subjects
    selected_nums = parse_subj_selection(args.subjects)
    if selected_nums:
        print(f"Filtering subjects by numeric IDs: {sorted(list(selected_nums))}")

    all_subjects = sorted([d for d in processed_dir.iterdir() if d.is_dir()])
    tasks = []

    for subj_dir in all_subjects:
        subj_name = subj_dir.name

        # Apply selection filter if specified
        if selected_nums:
            num = get_subj_num(subj_name)
            if num is None or num not in selected_nums:
                continue

        # Find marker NPZ files for this subject
        npz_files = sorted(glob.glob(str(subj_dir / "*_markers_positions.npz")))
        if not npz_files:
            continue

        subj_out_dir = out_root / subj_name

        for npz_path in npz_files:
            trial_name = Path(npz_path).stem.replace("_markers_positions", "")
            trial_safe = sanitize(trial_name)
            output_path = subj_out_dir / f"{trial_safe}_smpl_params.npz"

            if output_path.exists():
                # print(f"Skipping {trial_name}, already exists.")
                continue

            tasks.append(
                (
                    str(npz_path),
                    str(output_path),
                    args.models_dir,
                    args.device,
                    args.batch_size,
                    args.vicon_up,
                    args.vicon_forward,
                )
            )

    print(f"Found {len(tasks)} trials to process across {len(all_subjects)} subjects.")

    if not tasks:
        print("No tasks to perform. Exit.")
        return

    print(f"Starting batch processing with {args.workers} workers on {args.device}...")

    # Process tasks in parallel
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(worker_task, tasks))

    success_count = sum(results)
    print(f"Batch processing complete: {success_count}/{len(tasks)} trials succeeded.")


if __name__ == "__main__":
    main()
