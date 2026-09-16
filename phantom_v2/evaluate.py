#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate.py: TRE + baseline SVD (4 landmark points) per PhantomDataset.

Usage:
  python evaluate.py --model_path checkpoints/exp/models/model.best.t7 --stl phantom.stl --mode sweep
  python evaluate.py --mode sparse --num_eval 2000
"""

import argparse
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running from phantom_v2/ with dcp-master nearby
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from phantom_data import PhantomDataset, load_mesh
from model import DCP, SVDHead
from util import transform_point_cloud


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a') if path else None

    def cprint(self, text):
        print(text)
        if self.f:
            self.f.write(text + '\n')
            self.f.flush()

    def close(self):
        if self.f:
            self.f.close()


# ============================================================
# TRE targets — REPLACE when Prof provides anatomical landmarks
# ============================================================
# Coordinate nel frame phantom (mm o m, coerente con la mesh STL).
# Placeholder: nessun target definito → TRE su tutti i punti allineati.
TRE_TARGETS = np.empty((0, 3), dtype=np.float64)

# 4 landmark points nel frame phantom (mm o m, coerente con la mesh STL).
LANDMARK_POINTS = np.empty((0, 3), dtype=np.float64)

def apply_rigid_transform(points, R, t):
    """Applica una trasformazione a punti [N, 3]."""
    points = np.asarray(points, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            "I punti devono avere forma [N, 3]."
        )

    if R.shape != (3, 3) or t.shape != (3,):
        raise ValueError(
            "R deve essere [3, 3] e t deve essere [3]."
        )

    return points @ R.T + t


def fit_rigid_svd(src_pts, tgt_pts):
    """
    Stima source -> target da corrispondenze note.

    src_pts[i] e tgt_pts[i] devono rappresentare
    lo stesso punto fisico nei due sistemi di riferimento.
    """
    src = np.asarray(src_pts, dtype=np.float64)
    tgt = np.asarray(tgt_pts, dtype=np.float64)

    if (
        src.ndim != 2
        or src.shape[1] != 3
        or src.shape != tgt.shape
        or len(src) < 3
    ):
        raise ValueError(
            "Servono almeno 3 coppie corrispondenti "
            "in forma [N, 3]."
        )

    if not np.isfinite(src).all() or not np.isfinite(tgt).all():
        raise ValueError(
            "I punti devono avere coordinate finite."
        )

    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)

    src_c = src - src_mean
    tgt_c = tgt - tgt_mean

    if (
        np.linalg.matrix_rank(src_c) < 2
        or np.linalg.matrix_rank(tgt_c) < 2
    ):
        raise ValueError(
            "Landmark coincidenti o allineati: "
            "rotazione non determinabile."
        )

    H = src_c.T @ tgt_c
    U, _, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = tgt_mean - R @ src_mean

    return R, t

def rotation_error_deg(R_pred, R_gt):
    """Angolo della rotazione relativa, in gradi."""
    R_pred = np.asarray(R_pred, dtype=np.float64)
    R_gt = np.asarray(R_gt, dtype=np.float64)

    relative = R_pred @ R_gt.T
    cosine = (np.trace(relative) - 1.0) / 2.0

    return float(
        np.degrees(
            np.arccos(np.clip(cosine, -1.0, 1.0))
        )
    )


def translation_error_mm(t_pred_mm, t_gt_mm):
    """Distanza euclidea tra le traslazioni, in mm."""
    t_pred_mm = np.asarray(t_pred_mm, dtype=np.float64)
    t_gt_mm = np.asarray(t_gt_mm, dtype=np.float64)

    return float(
        np.linalg.norm(t_pred_mm - t_gt_mm)
    )

def compute_tre(
    R_pred,
    t_pred_mm,
    R_gt,
    t_gt_mm,
    target_pts_phantom_mm,
):
    """
    Restituisce un errore in mm per ciascun target anatomico.

    I target sono definiti nel frame phantom.
    Le trasformazioni sono Polaris -> phantom.
    """
    targets = np.asarray(
        target_pts_phantom_mm,
        dtype=np.float64,
    )

    if (
        targets.ndim != 2
        or targets.shape[1] != 3
        or len(targets) == 0
    ):
        raise ValueError(
            "Servono target anatomici in forma [N, 3]."
        )

    R_gt = np.asarray(R_gt, dtype=np.float64)
    t_gt_mm = np.asarray(t_gt_mm, dtype=np.float64)

    # Porta i target nel frame Polaris usando l'inversa GT.
    targets_polaris = apply_rigid_transform(
        targets,
        R_gt.T,
        -R_gt.T @ t_gt_mm,
    )

    # Riporta i target nel phantom con la trasformazione stimata.
    predicted = apply_rigid_transform(
        targets_polaris,
        R_pred,
        t_pred_mm,
    )

    return np.linalg.norm(
        predicted - targets,
        axis=1,
    )

def svd_baseline_4pts(
    landmark_pts_mm,
    R_gt,
    t_gt_mm,
    rng,
    noise_sigma_mm=0.3,
):
    """
    Simula l'acquisizione di quattro landmark noti
    e stima la trasformazione Polaris -> phantom.

    landmark_pts_mm: [4, 3], nel frame phantom.
    noise_sigma_mm: deviazione standard per coordinata.
    """
    landmarks = np.asarray(
        landmark_pts_mm,
        dtype=np.float64,
    )

    if landmarks.shape != (4, 3):
        raise ValueError(
            "La baseline richiede esattamente "
            "4 landmark [4, 3]."
        )

    if (
        not np.isfinite(noise_sigma_mm)
        or noise_sigma_mm < 0
    ):
        raise ValueError(
            "noise_sigma_mm deve essere finito e non negativo."
        )

    R_gt = np.asarray(R_gt, dtype=np.float64)
    t_gt_mm = np.asarray(t_gt_mm, dtype=np.float64)

    # Posizione ideale dei landmark nel frame Polaris.
    source = apply_rigid_transform(
        landmarks,
        R_gt.T,
        -R_gt.T @ t_gt_mm,
    )

    # Simula l'errore di acquisizione.
    source += rng.normal(
        0.0,
        noise_sigma_mm,
        source.shape,
    )

    return fit_rigid_svd(source, landmarks)

def test_evaluation_math():
    # Trasformazione nota: rotazione di 90° attorno a Z.
    R_gt = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ])

    t_gt = np.array([10.0, -20.0, 30.0])

    # Coordinate sintetiche utilizzate soltanto dal test.
    landmarks = np.array([
        [0.0,  0.0,  0.0],
        [35.0, 0.0,  0.0],
        [0.0, 40.0,  0.0],
        [0.0,  0.0, 25.0],
    ])

    targets = np.array([
        [5.0, 10.0, 15.0],
        [-8.0, 6.0, 12.0],
    ])

    # Senza rumore, la baseline deve recuperare la GT.
    R_fit, t_fit = svd_baseline_4pts(
        landmarks,
        R_gt,
        t_gt,
        rng=np.random.default_rng(42),
        noise_sigma_mm=0.0,
    )

    np.testing.assert_allclose(
        R_fit, R_gt, atol=1e-10, rtol=0
    )
    np.testing.assert_allclose(
        t_fit, t_gt, atol=1e-10, rtol=0
    )

    np.testing.assert_allclose(
        compute_tre(R_fit, t_fit, R_gt, t_gt, targets),
        0.0,
        atol=1e-10,
        rtol=0,
    )

    # Un errore puro di traslazione di 1 mm
    # deve produrre TRE di 1 mm su tutti i target.
    shift = np.array([1.0, 0.0, 0.0])

    np.testing.assert_allclose(
        compute_tre(
            R_gt, t_gt + shift, R_gt, t_gt, targets
        ),
        1.0,
        atol=1e-10,
        rtol=0,
    )

    np.testing.assert_allclose(
        translation_error_mm(t_gt + shift, t_gt),
        1.0,
    )

    np.testing.assert_allclose(
        rotation_error_deg(R_gt, np.eye(3)),
        90.0,
    )

    print(
        "OK: SVD, TRE nullo, errore di 1 mm "
        "e rotazione di 90 gradi."
    )

def evaluate(args):
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(
            f"Checkpoint non trovato: {args.model_path}"
        )

    if args.num_eval < 1 or args.batch_size < 1:
        raise ValueError(
            "num_eval e batch_size devono essere positivi."
        )

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    checkpoint = torch.load(
        args.model_path,
        map_location='cpu',
        weights_only=False,
    )

    required_keys = (
        'state_dict',
        'model_args',
        'network_scale_mm',
    )

    if (
        not isinstance(checkpoint, dict)
        or any(key not in checkpoint for key in required_keys)
    ):
        raise ValueError(
            "Il checkpoint deve contenere state_dict, "
            "model_args e network_scale_mm."
        )

    saved_args = checkpoint['model_args']
    config = (
        saved_args
        if isinstance(saved_args, dict)
        else vars(saved_args)
    )

    scale = float(checkpoint['network_scale_mm'])

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Scala del checkpoint non valida.")

    # Ricostruisce esattamente la configurazione salvata.
    net = DCP(argparse.Namespace(**config)).to(device)

    net.load_state_dict(
        checkpoint['state_dict'],
        strict=True,
    )
    net.eval()

    if args.bn_batch_stats:
        for module in net.modules():
            if isinstance(
                module,
                (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d),
            ):
                module.train()
                module.track_running_stats = False

        print(
            "DIAGNOSTICA: BatchNorm usa le statistiche del batch. "
            "Pesi e checkpoint non vengono modificati."
        )

    mode = args.mode or config.get('dset_mode', 'sweep')

    seed = (
        args.seed
        if args.seed is not None
        else int(config.get('manualSeed', 42)) + 2
    )

    n_points = (
        args.n_points
        if args.n_points is not None
        else config.get('dset_n_points')
    )

    noise_sigma = (
        args.noise_sigma
        if args.noise_sigma is not None
        else float(config.get('noise_sigma', 0.3))
    )

    if not np.isfinite(noise_sigma) or noise_sigma < 0:
        raise ValueError(
            "noise_sigma deve essere finito e non negativo."
        )

    stl_path = (
        args.stl
        or config.get('stl')
        or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'phantom.stl',
        )
    )

    print(f"Caricamento mesh: {stl_path}")

    dataset = PhantomDataset(
        stl_path=stl_path,
        mode=mode,
        num_samples=args.num_eval,
        n_points=n_points,
        rot_max=float(
            config.get('dset_rot_max', np.pi / 4)
        ),
        trans_max=float(
            config.get('dset_trans_max', 50.0)
        ),
        noise_sigma=noise_sigma,
        seed=seed,
        network_scale_mm=scale,
        patch_radius_mm=float(
            config.get('patch_radius_mm', 20.0)
        ),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    targets = np.asarray(TRE_TARGETS, dtype=np.float64)
    landmarks = np.asarray(
        LANDMARK_POINTS, dtype=np.float64
    )

    has_targets = targets.size > 0
    has_landmarks = landmarks.size > 0

    if has_targets and (
        targets.ndim != 2
        or targets.shape[1] != 3
        or not np.isfinite(targets).all()
    ):
        raise ValueError(
            "TRE_TARGETS deve contenere coordinate finite [N, 3]."
        )

    if has_landmarks and (
        landmarks.shape != (4, 3)
        or not np.isfinite(landmarks).all()
    ):
        raise ValueError(
            "LANDMARK_POINTS deve contenere "
            "quattro coordinate finite [4, 3]."
        )

    # Un generatore per l'intera valutazione:
    # il rumore cambia tra acquisizioni.
    baseline_rng = np.random.default_rng(
        np.random.SeedSequence(seed).spawn(1)[0]
    )

    metrics = {}

    def record(name, value):
        value = float(value)

        if not np.isfinite(value):
            raise RuntimeError(
                f"Metrica non finita: {name}"
            )

        metrics.setdefault(name, []).append(value)

    with torch.no_grad():
        for src, tgt, R_gt, t_gt in tqdm(
            loader,
            desc='Evaluation',
        ):
            R_pred, t_pred, _, _ = net(
                src.to(device),
                tgt.to(device),
            )

            R_pred = R_pred.cpu().numpy().astype(np.float64)
            t_pred = t_pred.cpu().numpy().astype(np.float64)

            R_gt_np = R_gt.numpy().astype(np.float64)
            t_gt_np = t_gt.numpy().astype(np.float64)

            for i in range(src.size(0)):
                # Conversione unica dalle unità della rete ai mm.
                src_mm = (
                    src[i].numpy().T.astype(np.float64) * scale
                )

                R_p = R_pred[i]
                t_p = t_pred[i] * scale
                R_g = R_gt_np[i]
                t_g = t_gt_np[i] * scale

                record(
                    'dcp_rotation_deg',
                    rotation_error_deg(R_p, R_g),
                )
                record(
                    'dcp_translation_mm',
                    translation_error_mm(t_p, t_g),
                )

                # Disaccordo tra trasformazione predetta e GT
                # sui punti osservati. Non è TRE anatomico.
                expected = apply_rigid_transform(
                    src_mm, R_g, t_g
                )
                predicted = apply_rigid_transform(
                    src_mm, R_p, t_p
                )

                record(
                    'dcp_source_proxy_mm',
                    np.linalg.norm(
                        predicted - expected,
                        axis=1,
                    ).mean(),
                )

                if has_targets:
                    errors = compute_tre(
                        R_p, t_p, R_g, t_g, targets
                    )

                    record('dcp_tre_mean_mm', errors.mean())

                    for j, error in enumerate(errors):
                        record(
                            f'dcp_tre_target_{j}_mm',
                            error,
                        )

                if has_landmarks:
                    R_b, t_b = svd_baseline_4pts(
                        landmarks,
                        R_g,
                        t_g,
                        rng=baseline_rng,
                        noise_sigma_mm=noise_sigma,
                    )

                    record(
                        'baseline_rotation_deg',
                        rotation_error_deg(R_b, R_g),
                    )
                    record(
                        'baseline_translation_mm',
                        translation_error_mm(t_b, t_g),
                    )

                    baseline_aligned = apply_rigid_transform(
                        src_mm, R_b, t_b
                    )

                    record(
                        'baseline_source_proxy_mm',
                        np.linalg.norm(
                            baseline_aligned - expected,
                            axis=1,
                        ).mean(),
                    )

                    if has_targets:
                        errors = compute_tre(
                            R_b, t_b, R_g, t_g, targets
                        )

                        record(
                            'baseline_tre_mean_mm',
                            errors.mean(),
                        )

                        for j, error in enumerate(errors):
                            record(
                                f'baseline_tre_target_{j}_mm',
                                error,
                            )

    log_dir = os.path.join(
        'checkpoints', args.exp_name
    )
    os.makedirs(log_dir, exist_ok=True)

    textio = IOStream(
        os.path.join(log_dir, 'eval.log')
    )

    summary = {}

    try:
        textio.cprint(f"\nCheckpoint: {args.model_path}")
        textio.cprint(
            "Modalità rete: diagnostica BatchNorm con statistiche del batch"
            if args.bn_batch_stats
            else "Modalità rete: evaluation standard"
    )
        textio.cprint(f"Mesh: {stl_path}")
        textio.cprint(
            f"Mode: {mode}, samples: {len(dataset)}, "
            f"points: {dataset.n_points}, seed: {seed}"
        )
        textio.cprint(
            f"Scale: {scale} mm, "
            f"noise sigma per coordinata: {noise_sigma} mm"
        )
        textio.cprint(
            f"Rotazione massima per asse: {dataset.rot_max} rad, "
            f"traslazione massima per asse: {dataset.trans_max} mm"
        )
        textio.cprint(
            f"Raggio patch: {dataset.patch_radius_mm} mm"
        )

        if not has_targets:
            textio.cprint(
                "TRE anatomico non calcolato: "
                "TRE_TARGETS non ancora definiti."
            )

        if not has_landmarks:
            textio.cprint(
                "Baseline non calcolata: "
                "LANDMARK_POINTS non ancora definiti."
            )

        for name, values in metrics.items():
            values = np.asarray(values, dtype=np.float64)

            stats = {
                'mean': float(values.mean()),
                'std': float(values.std()),
                'median': float(np.median(values)),
                'p95': float(np.percentile(values, 95)),
                'max': float(values.max()),
            }
            summary[name] = stats

            textio.cprint(
                f"{name}: "
                f"mean={stats['mean']:.6f}, "
                f"std={stats['std']:.6f}, "
                f"median={stats['median']:.6f}, "
                f"p95={stats['p95']:.6f}, "
                f"max={stats['max']:.6f}"
            )

    finally:
        textio.close()

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Valutazione registrazione phantom'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Checkpoint da valutare',
    )
    parser.add_argument(
        '--stl',
        type=str,
        default=None,
        help='Percorso mesh; default dalla configurazione salvata',
    )
    parser.add_argument(
        '--mode',
        choices=['sweep', 'sparse'],
        default=None,
        help='Default: modalità usata nel training',
    )
    parser.add_argument(
        '--num_eval',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
    )
    parser.add_argument(
        '--n_points',
        type=int,
        default=None,
        help='Default: numero di punti usato nel training',
    )
    parser.add_argument(
        '--noise_sigma',
        type=float,
        default=None,
        help='Rumore per coordinata in mm; default dal training',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Default: seed del training + 2',
    )
    parser.add_argument(
        '--exp_name',
        type=str,
        default='eval',
    )
    parser.add_argument(
        "--bn_batch_stats",
        action="store_true",
        help="Solo diagnostica: usa le statistiche del batch nelle BatchNorm",
    )

    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
