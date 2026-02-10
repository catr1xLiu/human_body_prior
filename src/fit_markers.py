#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMPL fitting using human_body_prior IK_Engine for Van Criekinge dataset.

This script fits SMPL body model parameters to a single marker NPZ file.
It uses human_body_prior's IK_Engine with VPoser v2.05 prior.

Usage:
    python -m human_body_prior.src.fit_markers \
        --input path/to/markers.npz \
        --output path/to/output.npz \
        --models_dir data/smpl
"""

import os
import json
import argparse
import glob
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# Import human_body_prior components
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.models.ik_engine import IK_Engine
from human_body_prior.tools.omni_tools import get_support_data_dir

# Import marker mapping utilities from same package
from . import marker_vids
from . import labels_map


def vicon_to_smpl_coords(points, vicon_up="Z", vicon_forward="Y"):
    """
    Convert Vicon coordinate system to SMPL coordinate system.

    Args:
        points: numpy array of shape (..., 3) containing 3D points in Vicon coordinates
        vicon_up: Vicon up axis ('X', 'Y', or 'Z')
        vicon_forward: Vicon forward axis ('X', 'Y', or 'Z')

    Returns:
        numpy array of same shape as points in SMPL coordinates

    Notes:
        Vicon typically uses Z-up, Y-forward. SMPL uses Y-up, Z-forward.
        Supported conventions:
        - Z-up, Y-forward (Vicon default): rotate -90° around X-axis
        - Z-up, X-forward: rotate -90° around X, then -90° around Y
        - Y-up, Z-forward (SMPL default): identity
        - Y-up, X-forward: rotate 90° around Y-axis
    """
    if vicon_up == "Z" and vicon_forward == "Y":
        rot = R.from_euler("x", -90, degrees=True).as_matrix()
    elif vicon_up == "Z" and vicon_forward == "X":
        rot = R.from_euler("xy", [-90, -90], degrees=True).as_matrix()
    elif vicon_up == "Y" and vicon_forward == "Z":
        rot = np.eye(3)
    elif vicon_up == "Y" and vicon_forward == "X":
        rot = R.from_euler("y", 90, degrees=True).as_matrix()
    else:
        raise ValueError(
            f"Unsupported Vicon convention: up={vicon_up}, forward={vicon_forward}"
        )

    original_shape = points.shape
    points_flat = points.reshape(-1, 3)
    transformed = (rot @ points_flat.T).T
    return transformed.reshape(original_shape)


def load_markers_npz(npz_path):
    """
    Load processed marker data from NPZ file.

    Args:
        npz_path: Path to NPZ file containing marker data

    Returns:
        tuple: (markers_np, marker_names, fps)
            - markers_np: numpy array of shape (T, N, 3) with marker positions
            - marker_names: list of N marker names
            - fps: frame rate in Hz

    Raises:
        FileNotFoundError: If NPZ file does not exist
        KeyError: If required keys are missing from NPZ file
    """
    d = np.load(npz_path, allow_pickle=True)
    M = d["marker_data"]
    names = [str(x) for x in d["marker_names"].tolist()]
    fps = float(d["frame_rate"])
    return M, names, fps


def sanitize(name: str):
    """
    Sanitize trial name for file naming.

    Args:
        name: Original trial name string

    Returns:
        str: Sanitized name with spaces replaced by underscores and
             parentheses removed

    Example:
        >>> sanitize("SUBJ1_0 (walking)")
        "SUBJ1_0_walking"
    """
    return name.replace(" ", "_").replace("(", "").replace(")", "")


def map_markers_to_vertex_ids(marker_names):
    """
    Map Van Criekinge marker names to SMPL vertex IDs.

    This function performs a two-step mapping:
    1. Converts marker names to canonical names using labels_map.py
    2. Maps canonical names to SMPL vertex IDs using marker_vids.py

    Args:
        marker_names: List of marker names from processed NPZ file

    Returns:
        tuple: (vids, canonical_names, valid_indices)
            - vids: List of vertex IDs that have mappings
            - canonical_names: List of canonical marker names after labels_map
            - valid_indices: Indices in original marker_names that have mappings

    Note:
        Only about 35 out of 111 Van Criekinge markers map to SMPL vertices,
        corresponding to core Plug-in Gait markers.
    """
    vids = []
    canonical_names = []
    valid_indices = []

    for i, name in enumerate(marker_names):
        # Convert to uppercase for matching
        key = name.strip().upper()

        # Apply labels_map to get canonical name
        canonical = labels_map.general_labels_map.get(key, key)

        # Check if canonical name exists in SMPL vertex mapping
        if canonical in marker_vids.all_marker_vids["smpl"]:
            vertex_id = marker_vids.all_marker_vids["smpl"][canonical]
            vids.append(vertex_id)
            canonical_names.append(canonical)
            valid_indices.append(i)

    print(f"  Mapped {len(vids)}/{len(marker_names)} markers to SMPL vertices")
    if len(vids) < 10:
        print(f"  Warning: Few markers mapped. Available: {canonical_names}")

    return vids, canonical_names, valid_indices


def prepare_markers_for_fitting(
    markers_np, valid_indices, vicon_up="Z", vicon_forward="Y"
):
    """
    Prepare marker data for IK fitting.

    This function:
    1. Selects only markers with valid vertex mappings
    2. Converts coordinates from Vicon to SMPL system
    3. Handles NaN values with linear interpolation
    4. Converts to PyTorch tensor

    Args:
        markers_np: numpy array of shape (T, N, 3) with marker positions
        valid_indices: List of indices of markers that have vertex mappings
        vicon_up: Vicon up axis ('X', 'Y', or 'Z')
        vicon_forward: Vicon forward axis ('X', 'Y', or 'Z')

    Returns:
        torch.Tensor: Tensor of shape (T, K, 3) containing valid markers
                      in SMPL coordinate system, where K = len(valid_indices)

    Note:
        NaN values are interpolated linearly across time. If all values
        in a channel are NaN, they are set to 0.0.
    """
    # Select only markers with vertex mappings
    markers_subset = markers_np[:, valid_indices, :]

    # Convert to SMPL coordinate system
    markers_smpl = vicon_to_smpl_coords(markers_subset, vicon_up, vicon_forward)

    # Handle NaN values (linear interpolation)
    T, K, _ = markers_smpl.shape
    markers_clean = markers_smpl.copy()

    for k in range(K):
        for d in range(3):
            v = markers_clean[:, k, d]
            nans = np.isnan(v)
            if nans.any() and not nans.all():
                idx = np.arange(T)
                v[nans] = np.interp(idx[nans], idx[~nans], v[~nans])
                markers_clean[:, k, d] = v
            elif nans.all():
                markers_clean[:, k, d] = 0.0

    return torch.from_numpy(markers_clean).float()


def create_source_keypoints(bm_fname, vids, device):
    """
    Create SourceKeyPoints module for IK_Engine.

    This module defines how virtual markers are computed from SMPL parameters.
    It is adapted from ik_example_mocap.py to work with our markers.

    Args:
        bm_fname: Path to SMPL model file
        vids: List of vertex IDs corresponding to markers
        device: PyTorch device (cpu or cuda)

    Returns:
        SourceKeyPoints: PyTorch module that computes virtual marker positions
                         from SMPL parameters

    Note:
        Virtual markers are computed as vertex positions + 0.0095m offset
        along vertex normals, following the human_body_prior convention.
    """
    from colour import Color
    from torch import nn

    def compute_vertex_normal_batched(vertices, indices):
        from pytorch3d.structures import Meshes

        return (
            Meshes(verts=vertices, faces=indices.expand(len(vertices), -1, -1))
            .verts_normals_packed()
            .view(-1, vertices.shape[1], 3)
        )

    class SourceKeyPoints(nn.Module):
        def __init__(self, bm, vids, kpts_colors=None):
            super().__init__()
            self.bm = (
                BodyModel(bm, persistant_buffer=False) if isinstance(bm, str) else bm
            )
            self.bm_f = []  # self.bm.f
            self.vids = vids
            self.kpts_colors = (
                np.array([Color("grey").rgb for _ in vids])
                if kpts_colors is None
                else kpts_colors
            )

        def forward(self, body_parms):
            new_body = self.bm(**body_parms)

            # Compute vertex normals with offset (0.0095m as in example)
            vn = compute_vertex_normal_batched(new_body.v, new_body.f)
            virtual_markers = new_body.v[:, self.vids] + 0.0095 * vn[:, self.vids]

            return {"source_kpts": virtual_markers, "body": new_body}

    # Create color gradient for visualization
    red = Color("red")
    blue = Color("blue")
    kpts_colors = [c.rgb for c in list(red.range_to(blue, len(vids)))]

    return SourceKeyPoints(bm_fname, vids, kpts_colors).to(device)


def fit_sequence_with_ik_engine(markers_torch, vids, bm_fname, device, batch_size=128):
    """
    Fit SMPL parameters to markers using IK_Engine.

    This is the core optimization function that uses human_body_prior's
    IK_Engine with VPoser prior to fit SMPL parameters to marker data.

    Args:
        markers_torch: PyTorch tensor of shape (T, K, 3) with marker positions
        vids: List of vertex IDs corresponding to markers
        bm_fname: Path to SMPL model file
        device: PyTorch device (cpu or cuda)
        batch_size: Maximum frames per batch (sequences longer than this
                    are split into chunks)

    Returns:
        dict: Dictionary containing fitted SMPL parameters:
            - 'betas': numpy array of shape (10,) - shape parameters
            - 'root_orient': numpy array of shape (T, 3) - root orientation
            - 'pose_body': numpy array of shape (T, 63) - body pose
            - 'trans': numpy array of shape (T, 3) - global translation

    Note:
        Uses L-BFGS optimizer with VPoser regularization. Betas are averaged
        across chunks for consistency.
    """
    T = markers_torch.shape[0]

    # Chunk sequence if too long
    if T > batch_size:
        chunks = []
        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            chunks.append((start, end))
        print(
            f"  Splitting {T} frames into {len(chunks)} chunks of max {batch_size} frames"
        )
    else:
        chunks = [(0, T)]

    all_results = {"betas": [], "root_orient": [], "pose_body": [], "trans": []}

    # Configuration from ik_example_mocap.py
    data_loss = torch.nn.MSELoss(reduction="sum")
    stepwise_weights = [{"data": 10.0, "poZ_body": 0.03, "betas": 0.5}]
    optimizer_args = {
        "type": "LBFGS",
        "max_iter": 300,
        "lr": 1,
        "tolerance_change": 1e-4,
        "history_size": 200,
    }

    # Create IK_Engine
    support_dir = Path(get_support_data_dir()) / "dowloads"
    vposer_expr_dir = str(support_dir / "V02_05")

    ik_engine = IK_Engine(
        vposer_expr_dir=vposer_expr_dir,
        verbosity=1,  # Reduced verbosity for batch processing
        display_rc=(1, 1),
        data_loss=data_loss,
        stepwise_weights=stepwise_weights,
        optimizer_args=optimizer_args,
        num_betas=10,  # SMPL uses 10 betas
    ).to(device)

    # Process each chunk
    for chunk_idx, (start, end) in enumerate(chunks):
        chunk_markers = markers_torch[start:end].to(device)
        print(f"  Chunk {chunk_idx + 1}/{len(chunks)}: frames {start}:{end}")

        # Create source keypoints for this chunk
        source_pts = create_source_keypoints(bm_fname, vids, device)

        # Run IK optimization
        ik_res = ik_engine(source_pts, chunk_markers, initial_body_params={})

        # Extract results
        all_results["betas"].append(ik_res["betas"].detach().cpu().numpy())
        all_results["root_orient"].append(ik_res["root_orient"].detach().cpu().numpy())
        all_results["pose_body"].append(ik_res["pose_body"].detach().cpu().numpy())
        all_results["trans"].append(ik_res["trans"].detach().cpu().numpy())

    # Combine results across chunks
    combined = {}
    for key in all_results:
        if key == "betas":
            # Average betas across chunks (should be similar)
            combined[key] = np.mean(np.vstack(all_results[key]), axis=0)
        else:
            combined[key] = np.vstack(all_results[key])

    return combined


def build_smpl_model(model_dir, gender, device):
    """
    Load SMPL model for given gender using human_body_prior BodyModel.

    Args:
        model_dir: Directory containing SMPL model files
        gender: Gender string ('male', 'female', or 'neutral')
        device: PyTorch device (cpu or cuda)

    Returns:
        tuple: (model, bm_fname)
            - model: BodyModel instance
            - bm_fname: Path to the SMPL model file

    Raises:
        FileNotFoundError: If SMPL model file for specified gender is not found
    """
    # Get model file path for BodyModel
    if gender.lower() == "male":
        bm_fname = str(Path(model_dir) / "SMPL_MALE.npz")
    elif gender.lower() == "female":
        bm_fname = str(Path(model_dir) / "SMPL_FEMALE.npz")
    else:
        bm_fname = str(Path(model_dir) / "SMPL_NEUTRAL.npz")

    # Create BodyModel instance
    model = BodyModel(bm_fname=bm_fname, num_betas=10, model_type="smpl").to(device)

    return model, bm_fname


def pick_gender(subject_meta):
    """
    Extract gender from subject metadata.

    Args:
        subject_meta: Dictionary containing subject metadata

    Returns:
        str: Gender string ('male', 'female', or 'neutral')

    Note:
        Falls back to 'neutral' if gender is not specified or invalid.
    """
    g = subject_meta.get("gender", "neutral").lower()
    return g if g in ("male", "female") else "neutral"


def run_fitting(
    input_path,
    output_path,
    models_dir,
    device="cuda",
    batch_size=128,
    vicon_up="Z",
    vicon_forward="Y",
):
    """
    Fit SMPL parameters to a single marker NPZ file.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = output_path.with_suffix(".json")

    # Setup device
    if device == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
    else:
        dev = torch.device("cpu")

    subj_dir = input_path.parent
    subj = subj_dir.name
    trial_name = input_path.stem.replace("_markers_positions", "")

    # Load subject metadata for gender
    meta_files = sorted(glob.glob(str(subj_dir / "*_metadata.json")))
    gender = "neutral"
    if meta_files:
        with open(meta_files[0], "r") as f:
            subj_meta = json.load(f)
        gender = pick_gender(subj_meta)

    # Load SMPL model
    smpl_model, bm_fname = build_smpl_model(models_dir, gender, dev)

    # Load marker data
    markers_np, marker_names, fps = load_markers_npz(input_path)

    # Map markers to vertex IDs
    vids, canonical_names, valid_indices = map_markers_to_vertex_ids(marker_names)

    # Prepare markers for fitting
    markers_torch = prepare_markers_for_fitting(
        markers_np, valid_indices, vicon_up, vicon_forward
    )

    # Fit SMPL parameters using IK_Engine
    result = fit_sequence_with_ik_engine(markers_torch, vids, bm_fname, dev, batch_size)

    # Prepare output
    poses72 = np.concatenate(
        [
            result["root_orient"],
            result["pose_body"],
            np.zeros((result["pose_body"].shape[0], 6)),
        ],
        axis=1,
    )

    # Compute joints
    poses_torch = torch.from_numpy(poses72).float().to(dev)
    trans_torch = torch.from_numpy(result["trans"]).float().to(dev)
    betas_torch = torch.from_numpy(result["betas"]).float().unsqueeze(0).to(dev)
    betas_expanded = betas_torch.expand(poses_torch.shape[0], -1)

    with torch.no_grad():
        body = smpl_model(
            pose_body=poses_torch[:, 3:66],
            betas=betas_expanded,
            trans=trans_torch,
            root_orient=poses_torch[:, :3],
        )
        joints = body.Jtr.detach().cpu().numpy()

    np.savez(
        output_path,
        poses=poses72.astype(np.float32),
        trans=result["trans"].astype(np.float32),
        betas=result["betas"].astype(np.float32),
        gender=gender,
        subject_id=subj,
        trial_name=trial_name,
        fps=fps,
        n_frames=result["trans"].shape[0],
        joints=joints.astype(np.float32),
    )

    # Save metadata report
    report = {
        "subject_id": subj,
        "trial_name": trial_name,
        "gender": gender,
        "frames_fitted": int(markers_torch.shape[0]),
        "fps": fps,
        "markers_used": canonical_names,
        "vertex_ids": vids,
        "settings": {
            "vicon_up": vicon_up,
            "vicon_forward": vicon_forward,
            "batch_size": batch_size,
            "device": str(dev),
        },
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[{subj}] Finished {trial_name} -> {output_path.name}")
    return result["betas"]


def main():
    parser = argparse.ArgumentParser(
        description="Fit SMPL parameters to a single Van Criekinge marker NPZ file using human_body_prior",
    )

    parser.add_argument("--input", required=True, help="Path to input marker NPZ")
    parser.add_argument("--output", required=True, help="Path to output SMPL NPZ")
    parser.add_argument("--models_dir", required=True, help="Path to SMPL models")
    parser.add_argument(
        "--device", default="cuda", choices=["cpu", "cuda"], help="Device to use"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Maximum frames per batch"
    )
    parser.add_argument(
        "--vicon_up", default="Z", choices=["X", "Y", "Z"], help="Vicon up axis"
    )
    parser.add_argument(
        "--vicon_forward",
        default="Y",
        choices=["X", "Y", "Z"],
        help="Vicon forward axis",
    )

    args = parser.parse_args()

    run_fitting(
        input_path=args.input,
        output_path=args.output,
        models_dir=args.models_dir,
        device=args.device,
        batch_size=args.batch_size,
        vicon_up=args.vicon_up,
        vicon_forward=args.vicon_forward,
    )


if __name__ == "__main__":
    main()
