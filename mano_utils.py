import inspect
import pickle
import warnings
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "mano_v1_2" / "models"


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


def model_path_for_hand(hand: str) -> Path:
    model_name = "MANO_RIGHT.pkl" if hand == "right" else "MANO_LEFT.pkl"
    return MODEL_DIR / model_name


def load_mano_pickle(model_path: Path) -> dict:
    patch_legacy_dependencies()
    with model_path.open("rb") as fp:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            return pickle.load(fp, encoding="latin1")


def load_mano_data(hand: str) -> dict:
    return load_mano_pickle(model_path_for_hand(hand))


def build_canonical_vertices(model_data: dict) -> np.ndarray:
    verts = np.asarray(model_data["v_template"], dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"Unexpected vertex array shape: {verts.shape}")
    return verts


def load_faces(model_data: dict) -> np.ndarray:
    faces = np.asarray(model_data["f"], dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Unexpected face array shape: {faces.shape}")
    if faces.min() < 0:
        raise ValueError("OBJ export requires non-negative face indices.")
    return faces


def load_skinning_weights(model_data: dict) -> np.ndarray:
    weights = np.asarray(model_data["weights"], dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError(f"Unexpected skinning weight array shape: {weights.shape}")
    return weights


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fp:
        for x, y, z in vertices:
            fp.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
        for i, j, k in faces + 1:
            fp.write(f"f {i} {j} {k}\n")


def write_obj_from_template(
    path: Path,
    vertices: np.ndarray,
    template_obj: Path,
) -> None:
    vertex_count = 0
    output_lines: list[str] = []
    with template_obj.open("r", encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            if line.startswith("v "):
                if vertex_count >= len(vertices):
                    raise ValueError(
                        f"Template OBJ has more vertices than provided: {template_obj}"
                    )
                x, y, z = vertices[vertex_count]
                output_lines.append(f"v {x:.8f} {y:.8f} {z:.8f}\n")
                vertex_count += 1
            else:
                output_lines.append(line)

    if vertex_count != len(vertices):
        raise ValueError(
            f"Template OBJ vertex count mismatch: expected {vertex_count}, got {len(vertices)}"
        )

    with path.open("w", encoding="utf-8", newline="\n") as fp:
        fp.writelines(output_lines)


def rodrigues(rotvec: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rotvec)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = rotvec / theta
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    outer = np.outer(axis, axis)
    return (
        np.cos(theta) * np.eye(3, dtype=np.float64)
        + (1.0 - np.cos(theta)) * outer
        + np.sin(theta) * k
    )


def pose_feature_from_full_pose(full_pose: np.ndarray) -> np.ndarray:
    pose = full_pose.reshape(-1, 3)
    features = []
    for joint_rotvec in pose[1:]:
        features.append((rodrigues(joint_rotvec) - np.eye(3)).reshape(-1))
    return np.concatenate(features, axis=0)


def with_zeros(mat3x4: np.ndarray) -> np.ndarray:
    return np.vstack([mat3x4, np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64)])


def pack(vec4: np.ndarray) -> np.ndarray:
    return np.hstack([np.zeros((4, 3), dtype=np.float64), vec4.reshape(4, 1)])


def matmul_4x4(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ab,bc->ac", a, b)


def matvec_4(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ab,b->a", a, b)


def global_rigid_transformation(
    full_pose: np.ndarray, joints: np.ndarray, kintree_table: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pose = full_pose.reshape(-1, 3)
    id_to_col = {int(kintree_table[1, i]): i for i in range(kintree_table.shape[1])}
    parent = {
        i: id_to_col[int(kintree_table[0, i])]
        for i in range(1, kintree_table.shape[1])
    }

    results = {}
    results[0] = with_zeros(
        np.hstack([rodrigues(pose[0]), joints[0].reshape(3, 1)])
    )

    for i in range(1, kintree_table.shape[1]):
        joint_offset = (joints[i] - joints[parent[i]]).reshape(3, 1)
        transform = with_zeros(np.hstack([rodrigues(pose[i]), joint_offset]))
        results[i] = matmul_4x4(results[parent[i]], transform)

    results_global = np.stack([results[i] for i in range(len(results))], axis=0)
    results_local = np.stack(
        [
            results[i] - pack(matvec_4(results[i], np.append(joints[i], 0.0)))
            for i in range(len(results))
        ],
        axis=0,
    )
    return results_local, results_global


def lbs(
    full_pose: np.ndarray,
    v_posed: np.ndarray,
    joints: np.ndarray,
    weights: np.ndarray,
    kintree_table: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    transforms, joint_transforms = global_rigid_transformation(
        full_pose, joints, kintree_table
    )
    blended = np.tensordot(weights, transforms, axes=([1], [0]))
    rest_shape_h = np.hstack([v_posed, np.ones((v_posed.shape[0], 1), dtype=np.float64)])
    vertices = np.einsum("nij,nj->ni", blended, rest_shape_h)[:, :3]
    return vertices, joint_transforms[:, :3, 3]


def apply_lbs(
    vertices_rest: np.ndarray,
    weights: np.ndarray,
    transforms: np.ndarray,
    translation: np.ndarray | None = None,
) -> np.ndarray:
    vertices_rest = np.asarray(vertices_rest, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    transforms = np.asarray(transforms, dtype=np.float64)
    if vertices_rest.ndim != 2 or vertices_rest.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) rest vertices, got {vertices_rest.shape}")
    if weights.ndim != 2:
        raise ValueError(f"Expected (N, J) skinning weights, got {weights.shape}")
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError(f"Expected (J, 4, 4) transforms, got {transforms.shape}")
    if weights.shape[0] != vertices_rest.shape[0]:
        raise ValueError("Vertex count mismatch between rest vertices and weights.")
    if weights.shape[1] != transforms.shape[0]:
        raise ValueError("Joint count mismatch between weights and transforms.")

    blended = np.tensordot(weights, transforms, axes=([1], [0]))
    vertices_h = np.hstack(
        [vertices_rest, np.ones((vertices_rest.shape[0], 1), dtype=np.float64)]
    )
    vertices = np.einsum("nij,nj->ni", blended, vertices_h)[:, :3]
    if translation is not None:
        vertices = vertices + np.asarray(translation, dtype=np.float64)
    return vertices


def mano_deform_state(
    model_data: dict,
    betas: np.ndarray | None = None,
    global_orient: np.ndarray | None = None,
    hand_pose: np.ndarray | None = None,
) -> dict:
    betas = np.zeros(10, dtype=np.float64) if betas is None else np.asarray(betas, dtype=np.float64)
    global_orient = (
        np.zeros(3, dtype=np.float64)
        if global_orient is None
        else np.asarray(global_orient, dtype=np.float64)
    )
    hand_pose = (
        np.zeros(45, dtype=np.float64)
        if hand_pose is None
        else np.asarray(hand_pose, dtype=np.float64)
    )

    v_template = np.asarray(model_data["v_template"], dtype=np.float64)
    shapedirs = np.asarray(model_data["shapedirs"], dtype=np.float64)
    posedirs = np.asarray(model_data["posedirs"], dtype=np.float64)
    weights = load_skinning_weights(model_data)
    kintree_table = np.asarray(model_data["kintree_table"], dtype=np.int64)
    joint_regressor = model_data["J_regressor"]

    v_shaped = v_template + np.tensordot(shapedirs, betas, axes=([2], [0]))
    joints = np.stack(
        [joint_regressor @ v_shaped[:, axis] for axis in range(3)],
        axis=1,
    )

    full_pose = np.concatenate([global_orient, hand_pose], axis=0)
    pose_feature = pose_feature_from_full_pose(full_pose)
    v_posed = v_shaped + np.tensordot(posedirs, pose_feature, axes=([2], [0]))
    transforms, joint_transforms = global_rigid_transformation(
        full_pose, joints, kintree_table
    )

    return {
        "v_template": v_template,
        "v_shaped": v_shaped,
        "v_posed": v_posed,
        "joints_rest": joints,
        "full_pose": full_pose,
        "pose_feature": pose_feature,
        "weights": weights,
        "transforms": transforms,
        "joint_transforms": joint_transforms,
        "betas": betas,
    }


def mano_forward(
    model_data: dict,
    betas: np.ndarray | None = None,
    global_orient: np.ndarray | None = None,
    hand_pose: np.ndarray | None = None,
    translation: np.ndarray | None = None,
) -> dict:
    betas = np.zeros(10, dtype=np.float64) if betas is None else np.asarray(betas, dtype=np.float64)
    global_orient = (
        np.zeros(3, dtype=np.float64)
        if global_orient is None
        else np.asarray(global_orient, dtype=np.float64)
    )
    hand_pose = (
        np.zeros(45, dtype=np.float64)
        if hand_pose is None
        else np.asarray(hand_pose, dtype=np.float64)
    )
    translation = (
        np.zeros(3, dtype=np.float64)
        if translation is None
        else np.asarray(translation, dtype=np.float64)
    )
    state = mano_deform_state(
        model_data,
        betas=betas,
        global_orient=global_orient,
        hand_pose=hand_pose,
    )
    faces = load_faces(model_data)
    vertices = apply_lbs(
        state["v_posed"], state["weights"], state["transforms"], translation=translation
    )
    joints_world = state["joint_transforms"][:, :3, 3] + translation

    return {
        "vertices": vertices,
        "faces": faces,
        "joints": joints_world,
        "v_template": state["v_template"],
        "v_shaped": state["v_shaped"],
        "v_posed": state["v_posed"],
        "full_pose": state["full_pose"],
        "betas": state["betas"],
        "transforms": state["transforms"],
    }
