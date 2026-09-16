#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PhantomDataset: generatore di coppie di point cloud (src, tgt, R, t) sintetici
da una mesh STL del phantom. Sostituisce data.py per il training V2.

Modi:
  sweep  -> patch connesso (~512 punti) con trasformazioni continue
  sparse -> punti sparsi (20-50) con trasformazioni continue

Placeholder TARGET_POINTS: riempire con coordinate anatomiche nel frame phantom
quando la Prof le fornisce (almeno punto vicino al midollo spinale).
"""

import os
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from collections import deque
from scipy.sparse import csr_matrix


class MeshLoadError(Exception):
    pass


def load_mesh(stl_path):
    """Carica la mesh mantenendo coordinate e unità originali.Presente nella documentazione Trimesh."""
    import trimesh

    mesh = trimesh.load(
        stl_path,
        force='mesh',
        process=True,
    )

    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        raise MeshLoadError("La mesh è vuota.")

    if not np.isfinite(mesh.vertices).all():
        raise MeshLoadError(
            "La mesh contiene coordinate non finite."
        )

    areas = np.asarray(mesh.area_faces)

    if not np.isfinite(areas).all():
        raise MeshLoadError(
            "La mesh contiene aree non finite."
        )

    valid = areas > 0

    if not valid.any():
        raise MeshLoadError(
            "La mesh non contiene triangoli di area positiva."
        )

    if not valid.all():
        mesh.update_faces(valid)
        mesh.remove_unreferenced_vertices()

    return mesh

def build_face_neighbors(mesh):
    """Grafo sparso delle facce che condividono uno spigolo."""
    adjacency = np.asarray(mesh.face_adjacency)

    if len(adjacency) == 0:
        raise MeshLoadError(
            "Nessuna adiacenza tra facce: "
            "controllare la topologia della mesh."
        )

    graph = csr_matrix(
        (
            np.ones(len(adjacency), dtype=np.uint8),
            (adjacency[:, 0], adjacency[:, 1]),
        ),
        shape=(len(mesh.faces), len(mesh.faces)),
    )

    # Le connessioni devono essere percorribili in entrambe
    # le direzioni.
    return graph.maximum(graph.T).tocsr()


def _sample_triangles(mesh, face_ids, rng):
    """Genera un punto uniforme in ciascuna faccia selezionata."""
    triangles = np.asarray(mesh.vertices)[
        np.asarray(mesh.faces)[face_ids]
    ]

    uv = rng.random((len(face_ids), 2))
    root = np.sqrt(uv[:, 0])

    return (
        (1.0 - root)[:, None] * triangles[:, 0]
        + (root * (1.0 - uv[:, 1]))[:, None] * triangles[:, 1]
        + (root * uv[:, 1])[:, None] * triangles[:, 2]
    )


def sample_mesh_surface(
    mesh,
    n_points,
    strategy='random',
    rng=None,
    face_neighbors=None,
    face_probabilities=None,
    patch_radius_mm=20.0,
):
    """
    Restituisce esattamente n_points in formato [N, 3].

    La mesh deve avere coordinate in millimetri.
    """
    if n_points < 1:
        raise ValueError("n_points deve essere almeno 1.")

    if rng is None:
        rng = np.random.default_rng()

    # Se il dataset le ha già calcolate, riutilizziamo
    # direttamente le probabilità.
    if face_probabilities is None:
        areas = np.asarray(mesh.area_faces)
        total = areas.sum()

        if (
            not np.isfinite(areas).all()
            or not np.isfinite(total)
            or total <= 0
        ):
            raise MeshLoadError("Area della mesh non valida.")

        face_probabilities = areas / total

    if strategy == 'patch':
        if face_neighbors is None:
            raise ValueError(
                "Per lo sweep servono le adiacenze."
            )

        return _sample_patch(
            mesh,
            n_points,
            rng,
            face_neighbors,
            face_probabilities,
            patch_radius_mm,
        )

    if strategy != 'random':
        raise ValueError(
            f"Strategia sconosciuta: {strategy}"
        )

    selected = rng.choice(
        len(mesh.faces),
        size=n_points,
        replace=True,
        p=face_probabilities,
    )

    return _sample_triangles(mesh, selected, rng)


def _sample_patch(
    mesh,
    n_points,
    rng,
    face_neighbors,
    face_probabilities,
    patch_radius_mm,
):
    """
    Costruisce una patch connessa a partire da una faccia.

    Include le facce raggiungibili i cui centroidi distano
    al massimo patch_radius_mm dal centro della faccia seed.

    Poi campiona esattamente n_points nella patch.
    """
    if (
        not np.isfinite(patch_radius_mm)
        or patch_radius_mm <= 0
    ):
        raise ValueError(
            "patch_radius_mm deve essere positivo e finito."
        )

    centers = mesh.triangles_center
    radius_squared = patch_radius_mm ** 2
    max_seed_attempts = 100
    min_patch_faces = min(
        64,
        max(8, n_points // 4),)

    
    # Alcune facce della mesh possono essere isolate oppure
    # non avere vicini sufficientemente vicini al seme.
    # In quel caso scegliamo automaticamente un altro seme.
    for seed_attempt in range(max_seed_attempts):
        seed = int(
            rng.choice(
                len(mesh.faces),
                p=face_probabilities,
            )
        )

        origin = centers[seed]
        queue = deque([seed])
        seen = {seed}
        patch_faces = []

        while queue:
            face_id = queue.popleft()
            patch_faces.append(face_id)

            start = face_neighbors.indptr[face_id]
            end = face_neighbors.indptr[face_id + 1]

            for neighbor in face_neighbors.indices[start:end]:
                neighbor = int(neighbor)

                if neighbor in seen:
                    continue

                seen.add(neighbor)
                delta = centers[neighbor] - origin

                if np.dot(delta, delta) <= radius_squared:
                    queue.append(neighbor)

        if len(patch_faces) >= min_patch_faces:
            break

    else:
        raise MeshLoadError(
            "Impossibile costruire una patch con almeno "
            f"{min_patch_faces} facce dopo "
            f"{max_seed_attempts} tentativi."
        )

    patch_faces = np.asarray(patch_faces)
    areas = np.asarray(mesh.area_faces)[patch_faces]
    probabilities = areas / areas.sum()

    accepted = []
    remaining = n_points

    # Campiona sulla superficie e accetta soltanto i punti
    # entro il raggio. Non modifica le loro coordinate.
    for _ in range(100):
        selected = rng.choice(
            patch_faces,
            size=max(remaining, 256),
            replace=True,
            p=probabilities,
        )

        candidates = _sample_triangles(
            mesh, selected, rng
        )

        squared_distances = np.sum(
            (candidates - origin) ** 2,
            axis=1,
        )

        inside = candidates[
            squared_distances <= radius_squared
        ]

        take = min(remaining, len(inside))

        if take > 0:
            accepted.append(inside[:take])
            remaining -= take

        if remaining == 0:
            return np.vstack(accepted)

    raise MeshLoadError(
        "Impossibile campionare abbastanza punti nel raggio: "
        "controllare geometria della mesh e dimensione della patch."
    )


def generate_transform(
    rng,
    rot_max=np.pi / 4,       # max rotation per axis
    trans_max=0.05,           # max translation per axis (meter)
    euler_seq='zyx',
):
    """
    Genera una trasformazione rigida casuale (R, t).

    Returns:
        R_ab: rotation matrix 3x3
        t_ab: translation vector 3
        R_ba: inverse rotation 3x3
        t_ba: inverse translation 3
    """
    if rng is None:
        rng = np.random.default_rng()

    angles = rng.uniform(-rot_max, rot_max, size=3)
    R_ab = Rotation.from_euler(euler_seq, angles).as_matrix()
    R_ba = R_ab.T
    t_ab = rng.uniform(-trans_max, trans_max, size=3)
    t_ba = -R_ba @ t_ab

    return R_ab, t_ab, R_ba, t_ba


class PhantomDataset(Dataset):
    """
    Dataset sintetico da phantom STL.

    Args:
        stl_path:         path al file STL
        mode:             'sweep' (patch connesso) o 'sparse' (punti sparsi)
        num_samples:      numero di coppie da generare
        n_points:         numero punti per cloud (default 512 sweep, 25 sparse)
        rot_max:          max rotazione per asse (radiani)
        trans_max:        max traslazione per asse (metri)
        seed:             seed per riproducibilità
    """

    # TARGET_POINTS: placeholder — coordinate anatomiche nel frame phantom (mm)
    # Esempio: riempire con quando le fornisce la Prof
    # TARGET_POINTS = np.array([
    #     [x1, y1, z1],  # midollo spinale C3
    #     [x2, y2, z2],  # processo spinoso C3
    #     ...
    # ], dtype=np.float64)
    TARGET_POINTS = np.empty((0, 3), dtype=np.float64)

    def __init__(
        self,
        stl_path=None,
        mode='sweep',
        num_samples=1000,
        n_points=None,
        rot_max=np.pi / 4,
        trans_max=50.0,
        noise_sigma=0.3,
        factor=None,
        seed=42,
        network_scale_mm=100.0,
        patch_radius_mm=20.0,
    ):
        self.mesh = load_mesh(stl_path)
        self.mode = mode
        self.num_samples = num_samples
        self.rot_max = rot_max
        self.trans_max = trans_max
        self.noise_sigma = noise_sigma
        self.factor = factor
        self.seed = int(seed)
        self.network_scale_mm = float(network_scale_mm)
        self.patch_radius_mm = float(patch_radius_mm)

        if (
            not np.isfinite(self.network_scale_mm)
            or self.patch_radius_mm <= 0
        ):
            raise ValueError(
                "patch_radius_mm deve essere positivo e finito."
            )

        if n_points is None:
            n_points = 512 if mode == 'sweep' else 25

        self.n_points = n_points

        # 4 landmark casuali per baseline SVD (fissati con seed)
        landmark_rng = np.random.default_rng(seed)
        vertices = np.asarray(self.mesh.vertices, dtype=np.float64)
        num_mesh_verts = len(vertices)
        self.landmark_indices = landmark_rng.choice(
            num_mesh_verts, 4, replace=False
        )
        self.landmark_pts = vertices[self.landmark_indices].copy()
        # Ground truth landmark transforms (identity per default)
        self.landmark_R_gt = np.eye(3)
        self.landmark_t_gt = np.zeros(3)

        self.rng = np.random.default_rng(seed)

        # Probabilità calcolate una volta per dataset.
        areas = np.asarray(self.mesh.area_faces)
        total_area = areas.sum()

        if not np.isfinite(total_area) or total_area <= 0:
            raise MeshLoadError("Area totale non valida.")

        self.face_probabilities = areas / total_area

        # Le adiacenze servono soltanto per lo sweep.
        self.face_neighbors = None

        if self.mode == 'sweep':
            self.face_neighbors = build_face_neighbors(self.mesh)

            # Precalcola i centroidi nella cache della mesh.
            _ = self.mesh.triangles_center

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        rand = np.random.default_rng(
            np.random.SeedSequence([self.seed, int(item)])
        )

        # Sample src point cloud
        if self.mode == 'sparse':
            n = self.n_points  # 20-50
            strategy = 'random'
        else:
            n = self.n_points  # ~512
            strategy = 'patch'

        observed_phantom = sample_mesh_surface(
            self.mesh,
            n,
            strategy=strategy,
            rng=rand,
            face_neighbors=self.face_neighbors,
            face_probabilities=self.face_probabilities,
            patch_radius_mm=self.patch_radius_mm,
        )

        # 2. GT: Polaris → phantom
        R_gt, t_gt, _, _ = generate_transform(rand, self.rot_max, self.trans_max)

        # 3. Porta punti nel frame Polaris + rumore → src
        src = (R_gt.T @ (observed_phantom.T - t_gt[:, None])).T
        if self.noise_sigma > 0:
            src = src + rand.normal(0, self.noise_sigma, src.shape)

        # 4. tgt = punti phantom puliti
        tgt = observed_phantom

        # Coordinate fisiche in mm -> coordinate della rete.
        scale = self.network_scale_mm

        src_tensor = (src.T / scale).astype(np.float32)
        tgt_tensor = (tgt.T / scale).astype(np.float32)

        # La rotazione non cambia con la scala.
        R_gt_tensor = R_gt.astype(np.float32)
        t_gt_tensor = (t_gt / scale).astype(np.float32)

        return src_tensor, tgt_tensor, R_gt_tensor, t_gt_tensor


if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt

    if len(sys.argv) < 2:
        print("Usage: python phantom_data.py <stl_path>")
        sys.exit(1)

    # Rumore disabilitato SOLO per questo controllo geometrico.
    ds = PhantomDataset(
        stl_path=sys.argv[1],
        mode='sweep',
        num_samples=10,
        n_points=512,
        noise_sigma=0.0,
        seed=42,
        network_scale_mm=100.0,
        patch_radius_mm=20.0,
    )

    print(f"Mesh bounds:\n{ds.mesh.bounds}")
    print(f"Mesh extents: {ds.mesh.extents}")
    print(f"Vertici: {len(ds.mesh.vertices)}")
    print(f"Facce: {len(ds.mesh.faces)}")
    print(f"Coppie di facce adiacenti: {ds.face_neighbors.nnz // 2}")

    src, tgt, R, t = ds[0]

    # Riporta i punti nelle coordinate fisiche, in millimetri.
    scale = ds.network_scale_mm

    target_mm = tgt.T.astype(np.float64) * scale

    aligned_mm = (
        R.astype(np.float64) @ src.astype(np.float64)
        + t.astype(np.float64)[:, None]
    ).T * scale

    errors_mm = np.linalg.norm(
        aligned_mm - target_mm,
        axis=1,
    )

    patch_extents = np.ptp(target_mm, axis=0)

    print(f"Source shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    print(f"Estensione patch XYZ [mm]: {patch_extents}")
    print(f"Errore medio con GT [mm]: {errors_mm.mean():.8f}")
    print(f"Errore massimo con GT [mm]: {errors_mm.max():.8f}")

    if errors_mm.max() > 0.001:
        raise RuntimeError(
            "La GT non riallinea i punti entro 0.001 mm: "
            "controllare trasformazioni e scala."
        )

    # Campione della mesh per mostrare dove si trova la patch.
    context_mm = sample_mesh_surface(
        ds.mesh,
        n_points=10000,
        strategy='random',
        rng=np.random.default_rng(123),
        face_probabilities=ds.face_probabilities,
    )

    fig = plt.figure(figsize=(13, 6))

    ax1 = fig.add_subplot(121, projection='3d')

    ax1.scatter(
        *context_mm.T,
        s=2,
        c='gray',
        alpha=0.4,
        label='Superficie di riferimento',
    )

    ax1.scatter(
        *target_mm.T,
        s=8,
        c='red',
        alpha=1.0,
        label='Patch sweep',
    )

    ax1.set_title("Posizione della patch nel phantom")

    ax1.set_box_aspect(
        np.array(ds.mesh.extents, dtype=float, copy=True)
    )

    ax1.legend()

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(
        *target_mm.T,
        s=6,
        c='red',
    )
    ax2.set_title("Dettaglio della patch")
    ax2.set_box_aspect(
        np.maximum(patch_extents, 1e-3)
    )

    for ax in (ax1, ax2):
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm [mm]")
        ax.set_zlabel("Z [mm]")

    plt.tight_layout()
    plt.show()
