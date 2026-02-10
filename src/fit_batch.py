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
import signal
import subprocess
import time
import tempfile
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


active_processes = {}
max_iters = 400


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


def sanitize(name: str):
    """
    Sanitize trial name for file naming.
    """
    return name.replace(" ", "_").replace("(", "").replace(")", "")


def signal_handler(sig, frame):
    """Kill all active subprocesses on Ctrl+C"""
    print("\nTermination signal received. Killing workers...")
    for pid, p in active_processes.items():
        if p.poll() is None:
            p.terminate()
    os._exit(1)


signal.signal(signal.SIGINT, signal_handler)


def update_individual_progress(log_file, task_id, pbar):
    """Monitor a log file and update the tqdm progress bar"""
    last_iter = 0
    while task_id in active_processes:
        try:
            if not log_file.exists():
                time.sleep(0.5)
                continue

            with open(log_file, "r") as f:
                content = f.read()
                matches = re.findall(r"it (\d+) --", content)
                if matches:
                    current_iter = int(matches[-1])
                    if current_iter > last_iter:
                        pbar.update(current_iter - last_iter)
                        last_iter = current_iter
        except Exception:
            pass
        time.sleep(1)

    # Final catch-up
    if last_iter < max_iters:
        pbar.update(max_iters - last_iter)


def worker_task(task_args):
    """Task function for running fit_markers as a subprocess"""
    (
        input_path,
        output_path,
        models_dir,
        device,
        batch_size,
        vicon_up,
        vicon_forward,
        log_dir,
        task_id,
        pbar,
    ) = task_args

    trial_name = Path(input_path).stem.replace("_markers_positions", "")
    log_file = Path(log_dir) / f"{trial_name}.log"

    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "src.fit_markers",
        "--input",
        input_path,
        "--output",
        output_path,
        "--models_dir",
        models_dir,
        "--device",
        device,
        "--batch_size",
        str(batch_size),
        "--vicon_up",
        vicon_up,
        "--vicon_forward",
        vicon_forward,
    ]

    process = None
    try:
        # Start monitoring thread for this task
        monitor_thread = threading.Thread(
            target=update_individual_progress, args=(log_file, task_id, pbar)
        )
        monitor_thread.daemon = True

        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd, stdout=f, stderr=subprocess.STDOUT, text=True
            )
            active_processes[task_id] = process
            monitor_thread.start()
            process.wait()

        return process.returncode == 0
    except Exception:
        return False
    finally:
        if task_id in active_processes:
            del active_processes[task_id]


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
    parser.add_argument(
        "--log_dir",
        default=None,
        help="Directory for worker logs. If None, a temp dir is used.",
    )

    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Setup log directory
    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        log_dir = Path(tempfile.mkdtemp(prefix="fit_batch_logs_"))

    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Worker logs will be saved to: {log_dir}")

    # Filter subjects
    selected_nums = parse_subj_selection(args.subjects)
    if selected_nums:
        print(f"Filtering subjects by numeric IDs: {sorted(list(selected_nums))}")

    all_subjects = sorted([d for d in processed_dir.iterdir() if d.is_dir()])
    tasks_data = []

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
                continue

            tasks_data.append(
                {
                    "input_path": str(npz_path),
                    "output_path": str(output_path),
                    "trial_name": trial_name,
                }
            )

    print(
        f"Found {len(tasks_data)} trials to process across {len(all_subjects)} subjects."
    )

    if not tasks_data:
        print("No tasks to perform. Exit.")
        return

    print(
        f"Starting batch processing with {args.workers} workers on {args.device}...\n"
    )

    # Use tqdm nested progress bars
    # Main bar at position 0
    main_pbar = tqdm(
        total=len(tasks_data), desc="Overall Progress", position=0, leave=True
    )

    # We maintain a pool of worker bars at positions 1 to num_workers
    available_positions = list(range(1, args.workers + 1))
    pos_lock = threading.Lock()
    success_count = 0

    def run_with_tqdm(task_idx):
        t = tasks_data[task_idx]

        with pos_lock:
            pos = available_positions.pop(0)
            pbar = tqdm(
                total=max_iters,
                desc=f"Worker {pos:2d}: {t['trial_name'][:15]}",
                position=pos,
                leave=False,
            )

        try:
            arg = (
                t["input_path"],
                t["output_path"],
                args.models_dir,
                args.device,
                args.batch_size,
                args.vicon_up,
                args.vicon_forward,
                str(log_dir),
                task_idx,
                pbar,
            )
            res = worker_task(arg)
            return res
        finally:
            pbar.close()
            with pos_lock:
                available_positions.append(pos)
                available_positions.sort()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_with_tqdm, i): i for i in range(len(tasks_data))}
        for future in as_completed(futures):
            if future.result():
                success_count += 1
            main_pbar.update(1)

    main_pbar.close()

    # Move cursor past all worker bars
    print("\n" * args.workers)
    print(
        f"Batch processing complete: {success_count}/{len(tasks_data)} trials succeeded."
    )
    print(f"Logs are available at: {log_dir}")


if __name__ == "__main__":
    main()
