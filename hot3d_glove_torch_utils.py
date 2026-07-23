from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import smplx
import torch
from smplx.lbs import batch_rigid_transform, batch_rodrigues, blend_shapes, vertices2joints


ROOT = Path(__file__).resolve().parent


def patch_legacy_dependencies() -> None:
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec

    legacy_numpy_aliases = {
        "bool": bool,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for name, value in legacy_numpy_aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def mano_model_path(hand: str) -> Path:
    return ROOT / "mano_v1_2" / "models" / (
        "MANO_RIGHT.pkl" if hand == "right" else "MANO_LEFT.pkl"
    )


def load_mano_model_torch(hand: str) -> smplx.body_models.MANO:
    patch_legacy_dependencies()
    model = smplx.create(
        str(mano_model_path(hand)),
        use_pca=True,
        num_pca_comps=15,
        is_rhand=(hand == "right"),
    )
    # Match the official HOT3D loader fix for the known MANO left-hand
    # shapedirs sign issue in some SMPL-X/MANO releases.
    if hand == "left":
        right_model = smplx.create(
            str(mano_model_path("right")),
            use_pca=True,
            num_pca_comps=15,
            is_rhand=True,
        )
        shapedirs_diff = torch.sum(
            torch.abs(model.shapedirs[:, 0, :] - right_model.shapedirs[:, 0, :])
        )
        if float(shapedirs_diff.item()) < 1.0:
            model.shapedirs[:, 0, :] *= -1
    return model


def mano_template_vertices(
    model: smplx.body_models.MANO, betas: torch.Tensor | np.ndarray | None = None
) -> np.ndarray:
    if betas is None:
        return model.v_template.detach().cpu().numpy()
    if not torch.is_tensor(betas):
        betas = torch.tensor(betas, dtype=torch.float32)
    betas = betas.to(dtype=torch.float32).view(1, -1)
    v_shaped = model.v_template.unsqueeze(0) + blend_shapes(betas, model.shapedirs)
    return v_shaped[0].detach().cpu().numpy()


def mano_template_data(
    model: smplx.body_models.MANO, betas: torch.Tensor | np.ndarray | None = None
) -> dict:
    v_template = mano_template_vertices(model, betas=betas)
    shapedirs = model.shapedirs.detach().cpu().numpy()
    posedirs = (
        model.posedirs.detach().cpu().numpy().reshape(135, -1, 3).transpose(1, 2, 0)
    )
    weights = model.lbs_weights.detach().cpu().numpy()
    faces = model.faces.astype(np.int64)
    return {
        "v_template": v_template,
        "shapedirs": shapedirs,
        "posedirs": posedirs,
        "weights": weights,
        "faces": faces,
    }


def hot3d_full_pose(
    model: smplx.body_models.MANO,
    global_orient: torch.Tensor,
    hand_pose_coeffs: torch.Tensor,
) -> torch.Tensor:
    hand_pose = torch.einsum("bi,ij->bj", hand_pose_coeffs, model.hand_components)
    return torch.cat([global_orient, hand_pose], dim=1) + model.pose_mean.unsqueeze(0)


def glove_forward_torch(
    glove_template: torch.Tensor,
    glove_shapedirs: torch.Tensor,
    glove_posedirs: torch.Tensor,
    glove_weights: torch.Tensor,
    model: smplx.body_models.MANO,
    betas: torch.Tensor,
    global_orient: torch.Tensor,
    hand_pose_coeffs: torch.Tensor,
    translation: torch.Tensor,
    glove_base_betas: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    dtype = torch.float32
    device = glove_template.device

    betas = betas.to(device=device, dtype=dtype)
    global_orient = global_orient.to(device=device, dtype=dtype)
    hand_pose_coeffs = hand_pose_coeffs.to(device=device, dtype=dtype)
    translation = translation.to(device=device, dtype=dtype)
    if glove_base_betas is None:
        glove_base_betas = torch.zeros_like(betas)
    else:
        glove_base_betas = glove_base_betas.to(device=device, dtype=dtype).view(1, -1)
    glove_shape_delta = betas - glove_base_betas

    full_pose = hot3d_full_pose(model, global_orient, hand_pose_coeffs)
    rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(1, -1, 3, 3)

    v_shaped_mano = model.v_template.unsqueeze(0) + blend_shapes(betas, model.shapedirs)
    joints = vertices2joints(model.J_regressor, v_shaped_mano)
    _, transforms = batch_rigid_transform(rot_mats, joints, model.parents, dtype=dtype)

    glove_v_shaped = glove_template.unsqueeze(0) + blend_shapes(
        glove_shape_delta, glove_shapedirs
    )
    ident = torch.eye(3, dtype=dtype, device=device)
    pose_feature = (rot_mats[:, 1:] - ident).reshape(1, -1)
    glove_posedirs_matrix = glove_posedirs.permute(2, 0, 1).reshape(135, -1)
    glove_pose_offsets = torch.matmul(
        pose_feature, glove_posedirs_matrix
    ).view(1, -1, 3)
    glove_v_posed = glove_v_shaped + glove_pose_offsets

    num_joints = glove_weights.shape[1]
    transforms_flat = transforms.view(1, num_joints, 16)
    blended = torch.matmul(glove_weights.unsqueeze(0), transforms_flat).view(1, -1, 4, 4)

    homogen_coord = torch.ones(
        (1, glove_v_posed.shape[1], 1), dtype=dtype, device=device
    )
    glove_v_homo = torch.cat([glove_v_posed, homogen_coord], dim=2).unsqueeze(-1)
    glove_vertices = torch.matmul(blended, glove_v_homo)[:, :, :3, 0] + translation.unsqueeze(1)

    mano_output = model(
        betas=betas,
        global_orient=global_orient,
        hand_pose=hand_pose_coeffs,
        transl=translation,
        return_verts=True,
    )
    return (
        glove_vertices[0].detach().cpu().numpy(),
        mano_output.vertices[0].detach().cpu().numpy(),
    )
