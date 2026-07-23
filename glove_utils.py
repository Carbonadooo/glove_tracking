from __future__ import annotations

from pathlib import Path

import numpy as np


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    with path.open("r", encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            if line.startswith("v "):
                _, x, y, z = line.strip().split()[:4]
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                if len(parts) != 3:
                    raise ValueError(f"Only triangular OBJ faces are supported: {path}")
                face = [int(part.split("/")[0]) - 1 for part in parts]
                faces.append(face)

    verts = np.asarray(vertices, dtype=np.float64)
    tri_faces = np.asarray(faces, dtype=np.int64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"Unexpected vertex shape in {path}: {verts.shape}")
    if tri_faces.ndim != 2 or tri_faces.shape[1] != 3:
        raise ValueError(f"Unexpected face shape in {path}: {tri_faces.shape}")
    return verts, tri_faces


def bbox_center_and_extent(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = 0.5 * (vmin + vmax)
    extent = vmax - vmin
    return center, extent


def compute_uniform_alignment(
    src_vertices: np.ndarray, dst_vertices: np.ndarray
) -> tuple[float, np.ndarray]:
    src_center, src_extent = bbox_center_and_extent(src_vertices)
    dst_center, dst_extent = bbox_center_and_extent(dst_vertices)

    valid = src_extent > 1e-8
    if not np.any(valid):
        raise ValueError("Source mesh has degenerate bounding box.")
    scale = float(np.median(dst_extent[valid] / src_extent[valid]))
    translation = dst_center - scale * src_center
    return scale, translation


def apply_uniform_alignment(
    vertices: np.ndarray, scale: float, translation: np.ndarray
) -> np.ndarray:
    return np.asarray(vertices, dtype=np.float64) * scale + np.asarray(
        translation, dtype=np.float64
    )


def _pairwise_sq_dists_chunked(
    query_vertices: np.ndarray, ref_vertices: np.ndarray, chunk_size: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    query_vertices = np.asarray(query_vertices, dtype=np.float64)
    ref_vertices = np.asarray(ref_vertices, dtype=np.float64)
    all_indices = []
    all_dists = []
    for start in range(0, query_vertices.shape[0], chunk_size):
        stop = min(start + chunk_size, query_vertices.shape[0])
        chunk = query_vertices[start:stop]
        diff = chunk[:, None, :] - ref_vertices[None, :, :]
        sq_dists = np.sum(diff * diff, axis=2)
        all_indices.append(np.argsort(sq_dists, axis=1))
        all_dists.append(np.take_along_axis(sq_dists, all_indices[-1], axis=1))
    return np.vstack(all_indices), np.vstack(all_dists)


def knn_vertex_correspondence(
    query_vertices: np.ndarray,
    ref_vertices: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    if k <= 0:
        raise ValueError("k must be positive.")
    if k > ref_vertices.shape[0]:
        raise ValueError("k cannot exceed the number of reference vertices.")

    sorted_indices, sorted_sq_dists = _pairwise_sq_dists_chunked(
        query_vertices, ref_vertices
    )
    return sorted_indices[:, :k], sorted_sq_dists[:, :k]


def transfer_weights_knn(
    query_vertices: np.ndarray,
    ref_vertices: np.ndarray,
    ref_weights: np.ndarray,
    k: int,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    knn_indices, knn_sq_dists = knn_vertex_correspondence(
        query_vertices, ref_vertices, k=k
    )
    knn_dists = np.sqrt(knn_sq_dists)
    inv = 1.0 / (knn_dists + eps)
    blend = inv / inv.sum(axis=1, keepdims=True)
    transferred = np.einsum("nk,nkj->nj", blend, ref_weights[knn_indices])
    transferred = np.clip(transferred, 0.0, None)
    transferred /= transferred.sum(axis=1, keepdims=True)
    return transferred, knn_indices, knn_dists
