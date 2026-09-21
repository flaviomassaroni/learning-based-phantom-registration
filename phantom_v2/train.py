#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training loop per registro phantom con DCP.
Adatto da main.py: sostituisce ModelNet40 con PhantomDataset.
Batch: (src, tgt, R, t) -> 4 tensori invece di 8.
"""

import hashlib
from correspondence_loss import geometric_correspondence_loss
import os
import gc
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from phantom_data import PhantomDataset
from model import DCP
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from pathlib import Path
import shutil

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def _init_(args):
    # Cartella dei sorgenti effettivamente utilizzati.
    source_dir = Path(__file__).resolve().parent

    # Stesso percorso usato dal resto del training.
    experiment_dir = Path("checkpoints") / args.exp_name
    if not args.resume and any(
        (experiment_dir / "models").glob("*.t7")
    ):
        raise FileExistsError(
            f"{experiment_dir} contiene già checkpoint. "
            "Usa un nuovo --exp_name oppure --resume "
            "per riprendere."
    )
    (experiment_dir / "models").mkdir(
        parents=True,
        exist_ok=True,
    )

    for filename in (
        "train.py",
        "model.py",
        "phantom_data.py",
        "util.py",
        "correspondence_loss.py",
    ):
        source_path = source_dir / filename
        backup_path = experiment_dir / f"{filename}.backup"

        shutil.copy2(source_path, backup_path)


def predict_with_aux_loss(
    args,
    net,
    src,
    target,
    rotation_gt,
    translation_gt,
):
    # La GT non entra mai nel forward della rete.
    if getattr(args, "corr_weight", 0.0) == 0:
        return net(src, target), src.new_zeros(())

    outputs = net(
        src,
        target,
        return_correspondence=True,
    )

    corr_loss = geometric_correspondence_loss(
        outputs[4],
        src,
        target,
        rotation_gt,
        translation_gt,
        args.network_scale_mm,
        args.corr_sigma_mm,
    )

    return outputs[:4], corr_loss

def train_one_epoch(args, net, train_loader, optimizer, textio, epoch):
    net.train()

    total_loss = 0.0
    total_cycle_loss = 0.0
    total_mse_ab = 0.0
    total_mae_ab = 0.0
    total_mse_ba = 0.0
    total_mae_ba = 0.0
    total_rotation_loss = 0.0
    total_translation_loss = 0.0
    total_grad_norm = 0.0
    total_examples = 0
    total_corr_loss = 0.0
    total_objective = 0.0

    from util import transform_point_cloud

    for src, target, rotation_ab, translation_ab in tqdm(train_loader):
        src = src.to(args.device)
        target = target.to(args.device)
        rotation_ab = rotation_ab.to(args.device)
        translation_ab = translation_ab.to(args.device)

        batch_size = src.size(0)
        total_examples += batch_size

        optimizer.zero_grad(set_to_none=True)

        predictions, corr_loss = predict_with_aux_loss(
            args,
            net,
            src,
            target,
            rotation_ab,
            translation_ab,
        )

        (
            rotation_ab_pred,
            translation_ab_pred,
            rotation_ba_pred,
            translation_ba_pred,
        ) = predictions

        identity = torch.eye(
            3,
            device=args.device,
        ).unsqueeze(0).expand(batch_size, -1, -1)

        rotation_loss = F.mse_loss(
            torch.matmul(
                rotation_ab_pred.transpose(2, 1),
                rotation_ab,
            ),
            identity,
        )

        translation_loss = F.mse_loss(
            translation_ab_pred,
            translation_ab,
        )

        loss = rotation_loss + translation_loss
        weighted_cycle_loss = 0.0

        if args.cycle:
            rotation_loss_cycle = F.mse_loss(
                torch.matmul(
                    rotation_ba_pred,
                    rotation_ab_pred,
                ),
                identity,
            )

            cycle_translation = (
                torch.matmul(
                    rotation_ba_pred,
                    translation_ab_pred.unsqueeze(2),
                ).squeeze(2)
                + translation_ba_pred
            )

            translation_loss_cycle = torch.mean(
                cycle_translation ** 2
            )

            cycle_loss = (
                rotation_loss_cycle
                + translation_loss_cycle
            )

            loss = loss + 0.1 * cycle_loss
            weighted_cycle_loss = 0.1 * cycle_loss.item()

        if not torch.isfinite(loss).item():
            raise RuntimeError(
                "Loss non finita durante il training."
            )
        
        pose_cycle_loss = loss

        loss = (
            loss
            + getattr(args, "corr_weight", 0.0) * corr_loss
        )

        loss.backward()

        max_norm = (
            args.grad_clip
            if args.grad_clip > 0
            else float('inf')
        )

        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
            net.parameters(),
            max_norm=max_norm,
        )

        if not torch.isfinite(grad_norm_tensor).item():
            raise RuntimeError(
                "Gradienti non finiti: "
                "controllare dati e stabilità della SVD."
            )

        optimizer.step()

        # GT inversa: phantom -> Polaris
        rotation_ba_gt = rotation_ab.transpose(
            2, 1
        ).contiguous()

        translation_ba_gt = -torch.matmul(
            rotation_ba_gt,
            translation_ab.unsqueeze(2),
        ).squeeze(2)

        # Metriche indipendenti dall'ordine dei punti.
        with torch.no_grad():
            transformed_src_pred = transform_point_cloud(
                src,
                rotation_ab_pred,
                translation_ab_pred,
            )
            transformed_src_gt = transform_point_cloud(
                src,
                rotation_ab,
                translation_ab,
            )

            transformed_target_pred = transform_point_cloud(
                target,
                rotation_ba_pred,
                translation_ba_pred,
            )
            transformed_target_gt = transform_point_cloud(
                target,
                rotation_ba_gt,
                translation_ba_gt,
            )

            mse_ab = torch.mean(
                (transformed_src_pred - transformed_src_gt) ** 2
            )
            mae_ab = torch.mean(
                torch.abs(
                    transformed_src_pred - transformed_src_gt
                )
            )
            mse_ba = torch.mean(
                (transformed_target_pred - transformed_target_gt) ** 2
            )
            mae_ba = torch.mean(
                torch.abs(
                    transformed_target_pred
                    - transformed_target_gt
                )
            )

        total_loss += pose_cycle_loss.item() * batch_size
        total_corr_loss += corr_loss.item() * batch_size
        total_objective += loss.item() * batch_size
        total_cycle_loss += weighted_cycle_loss * batch_size
        total_rotation_loss += rotation_loss.item() * batch_size
        total_translation_loss += (
            translation_loss.item() * batch_size
        )
        total_grad_norm += (
            float(grad_norm_tensor.item()) * batch_size
        )

        total_mse_ab += mse_ab.item() * batch_size
        total_mae_ab += mae_ab.item() * batch_size
        total_mse_ba += mse_ba.item() * batch_size
        total_mae_ba += mae_ba.item() * batch_size

    return {
        'train_loss': total_loss / total_examples,
        'train_corr_loss': total_corr_loss / total_examples,
        'train_total_loss': total_objective / total_examples,
        'train_cycle_loss': total_cycle_loss / total_examples,
        'train_mse_ab': total_mse_ab / total_examples,
        'train_mae_ab': total_mae_ab / total_examples,
        'train_mse_ba': total_mse_ba / total_examples,
        'train_mae_ba': total_mae_ba / total_examples,
        'train_rotation_loss': (
            total_rotation_loss / total_examples
        ),
        'train_translation_loss': (
            total_translation_loss / total_examples
        ),
        'grad_norm': total_grad_norm / total_examples,
    }

@torch.no_grad()
def validate_one_epoch(args, net, test_loader, textio, epoch):
    net.eval()

    total_loss = 0.0
    total_corr_loss = 0.0
    total_objective = 0.0
    total_cycle_loss = 0.0
    total_mse_ab = 0.0
    total_mae_ab = 0.0
    total_mse_ba = 0.0
    total_mae_ba = 0.0
    num_examples = 0

    from util import transform_point_cloud

    for src, target, rotation_ab, translation_ab in tqdm(test_loader):
        src = src.to(args.device)
        target = target.to(args.device)
        rotation_ab = rotation_ab.to(args.device)
        translation_ab = translation_ab.to(args.device)

        batch_size = src.size(0)
        num_examples += batch_size

        predictions, corr_loss = predict_with_aux_loss(
            args,
            net,
            src,
            target,
            rotation_ab,
            translation_ab,
        )

        (
            rotation_ab_pred,
            translation_ab_pred,
            rotation_ba_pred,
            translation_ba_pred,
        ) = predictions

        identity = torch.eye(
            3,
            device=args.device,
        ).unsqueeze(0).expand(batch_size, -1, -1)

        rotation_loss = F.mse_loss(
            torch.matmul(
                rotation_ab_pred.transpose(2, 1),
                rotation_ab,
            ),
            identity,
        )

        translation_loss = F.mse_loss(
            translation_ab_pred,
            translation_ab,
        )

        loss = rotation_loss + translation_loss
        weighted_cycle_loss = 0.0

        if args.cycle:
            rotation_loss_cycle = F.mse_loss(
                torch.matmul(
                    rotation_ba_pred,
                    rotation_ab_pred,
                ),
                identity,
            )

            cycle_translation = (
                torch.matmul(
                    rotation_ba_pred,
                    translation_ab_pred.unsqueeze(2),
                ).squeeze(2)
                + translation_ba_pred
            )

            translation_loss_cycle = torch.mean(
                cycle_translation ** 2
            )

            cycle_loss = (
                rotation_loss_cycle
                + translation_loss_cycle
            )

            loss = loss + 0.1 * cycle_loss
            weighted_cycle_loss = 0.1 * cycle_loss.item()

        objective = (
            loss
            + getattr(args, "corr_weight", 0.0) * corr_loss
        )

        if not torch.isfinite(objective).item():
            raise RuntimeError(
                "Loss non finita durante la validation."
            )

        rotation_ba_gt = rotation_ab.transpose(
            2, 1
        ).contiguous()

        translation_ba_gt = -torch.matmul(
            rotation_ba_gt,
            translation_ab.unsqueeze(2),
        ).squeeze(2)

        transformed_src_pred = transform_point_cloud(
            src,
            rotation_ab_pred,
            translation_ab_pred,
        )
        transformed_src_gt = transform_point_cloud(
            src,
            rotation_ab,
            translation_ab,
        )

        transformed_target_pred = transform_point_cloud(
            target,
            rotation_ba_pred,
            translation_ba_pred,
        )
        transformed_target_gt = transform_point_cloud(
            target,
            rotation_ba_gt,
            translation_ba_gt,
        )

        mse_ab = torch.mean(
            (transformed_src_pred - transformed_src_gt) ** 2
        )
        mae_ab = torch.mean(
            torch.abs(
                transformed_src_pred - transformed_src_gt
            )
        )
        mse_ba = torch.mean(
            (transformed_target_pred - transformed_target_gt) ** 2
        )
        mae_ba = torch.mean(
            torch.abs(
                transformed_target_pred - transformed_target_gt
            )
        )

        total_loss += loss.item() * batch_size
        total_corr_loss += corr_loss.item() * batch_size
        total_objective += objective.item() * batch_size
        total_cycle_loss += weighted_cycle_loss * batch_size
        total_mse_ab += mse_ab.item() * batch_size
        total_mae_ab += mae_ab.item() * batch_size
        total_mse_ba += mse_ba.item() * batch_size
        total_mae_ba += mae_ba.item() * batch_size

    return {
        'val_loss': total_loss / num_examples,
        'val_corr_loss': total_corr_loss / num_examples,
        'val_total_loss': total_objective / num_examples,
        'val_cycle_loss': total_cycle_loss / num_examples,
        'val_mse_ab': total_mse_ab / num_examples,
        'val_mae_ab': total_mae_ab / num_examples,
        'val_mse_ba': total_mse_ba / num_examples,
        'val_mae_ba': total_mae_ba / num_examples,
    }


def main():
    parser = argparse.ArgumentParser(description='Training loop per registro phantom con DCP')

    # Data parameters
    parser.add_argument('--stl', type=str, default='phantom.stl', help='Path to phantom STL file')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument("--patch_radius_mm",type=float,default=20.0,help="Raggio della patch sweep, misurato sui centroidi delle facce",)

    # Model parameters
    parser.add_argument('--emb_nn', type=str, default='pointnet', choices=['pointnet', 'dgcnn'], help='Embedding network')
    parser.add_argument('--emb_dims', type=int, default=512, help='Dimension of the point embeddings')
    parser.add_argument('--pointer', type=str, default='transformer', choices=['identity', 'transformer'])
    parser.add_argument('--dgcnn_k', type=int, default=20, help='Numero di vicini usati da DGCNN')
    parser.add_argument('--n_blocks', type=int, default=1, help='Number of blocks in the encoder/decoder')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of heads in the multi-head attention')
    parser.add_argument('--ff_dims', type=int, default=1024, help='Dimension of the feed-forward network')
    parser.add_argument('--dropout', type=float, default=0.0,choices=[0.0], help='Dropout probability')
    parser.add_argument('--head', type=str, default='svd', choices=['svd', 'mlp'], help='Head type')
    parser.add_argument('--cycle', action='store_true', help='Enable cycle consistency loss')

    # Training parameters
    parser.add_argument('--manualSeed', type=int, default=42, help='Random seed')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=250, help='Number of epochs')
    parser.add_argument('--scheduler', type=int, default=50, help='Every how many epochs to reduce the lr')
    parser.add_argument('--grad_clip', type=float, default=0.0, help='Clip gradients norm (0=none)')
    parser.add_argument('--val_num_samples',type=int, default=100, help='Numero di campioni di validation')
    # Experiment parameters
    parser.add_argument('--exp_name', type=str, default='default', help='Experiment name')

    # Phantom dataset parameters
    parser.add_argument('--dset_mode', type=str, default='sweep', choices=['sweep', 'sparse'], help='Dataset mode')
    parser.add_argument('--dset_num_samples', type=int, default=5000, help='Number of training samples')
    parser.add_argument('--dset_n_points', type=int, default=None, help='Number of points')
    parser.add_argument('--target_n_points', type=int, default=1024, help='Numero di punti del digital twin globale')
    parser.add_argument('--dset_rot_max', type=float, default=np.pi / 4, help='Max rotation per axis (radians)')
    parser.add_argument('--dset_trans_max', type=float, default=50.0, help='Max translation per axis (mm)')
    parser.add_argument('--noise_sigma', type=float, default=0.3, help='Polaris tracker noise sigma in mm')
    parser.add_argument('--network_scale_mm', type=float, default=100.0, help='Common Network scale in mm')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--pretrained', type=str, default=None, help=('Carica solamente i pesi iniziali. '
                                                                      'Optimizer e scheduler ripartono da zero.'),
    )
    parser.add_argument("--corr_weight", type=float, default=0.0, help=("Peso della KL geometrica source->target; "
                                                                        "0 conserva la baseline"),)
    parser.add_argument("--corr_sigma_mm", type=float, default=5.0, help=("Sigma in mm delle etichette geometriche soft"),)


    args = parser.parse_args()

    if (
        not np.isfinite(args.corr_weight)
        or args.corr_weight < 0
    ):
        parser.error(
            "--corr_weight deve essere finito e non negativo."
        )

    if (
        not np.isfinite(args.corr_sigma_mm)
        or args.corr_sigma_mm <= 0
    ):
        parser.error(
            "--corr_sigma_mm deve essere positivo e finito."
        )

    if args.corr_weight > 0 and args.head != "svd":
        parser.error(
            "--corr_weight > 0 richiede --head svd."
        )

    if (
        args.batch_size < 1
        or args.epochs < 1
        or args.scheduler < 1
    ):
        parser.error(
            "batch_size, epochs e scheduler "
            "devono essere positivi."
        )

    if args.dset_num_samples < 1 or args.val_num_samples < 1:
        parser.error(
            "Training e validation devono contenere "
            "almeno un campione."
        )

    if args.resume and not os.path.isfile(args.resume):
        parser.error(
            f"Checkpoint resume non trovato: {args.resume}"
        )
    if args.resume is not None and args.pretrained is not None:
        parser.error(
            '--resume e --pretrained non possono essere usati insieme.')
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    torch.cuda.manual_seed_all(args.manualSeed)

    _init_(args)

    # ========= Data =========
    textio = IOStream(os.path.join('checkpoints', args.exp_name, 'train.log'))
    textio.cprint(f"Args: {args}")
    mesh_sha256 = hashlib.sha256(
        Path(args.stl).read_bytes()
    ).hexdigest()

    textio.cprint(
        "Obiettivo: pose + cycle + corr_weight * KL geometrica. "
        "model.best selezionato sulla val_loss "
        "(pose + cycle, senza KL)."
    )

    train_dataset = PhantomDataset(
        stl_path=args.stl,
        mode=args.dset_mode,
        num_samples=args.dset_num_samples,
        n_points=args.dset_n_points,
        target_n_points=args.target_n_points,
        rot_max=args.dset_rot_max,
        trans_max=args.dset_trans_max,
        noise_sigma=args.noise_sigma,
        seed=args.manualSeed,
        network_scale_mm=args.network_scale_mm,
        patch_radius_mm=args.patch_radius_mm,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device == 'cuda'),
    )
    textio.cprint(f"Training set size: {len(train_dataset)}")

    # Evaluate on fixed set of samples
    eval_dataset = PhantomDataset(
        stl_path=args.stl,
        mode=args.dset_mode,
        num_samples=args.val_num_samples,
        n_points=args.dset_n_points,
        target_n_points=args.target_n_points,
        rot_max=args.dset_rot_max,
        trans_max=args.dset_trans_max,
        noise_sigma=args.noise_sigma,
        seed=args.manualSeed + 1,
        network_scale_mm=args.network_scale_mm,
        patch_radius_mm=args.patch_radius_mm,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device == 'cuda'),
    )
    textio.cprint(f"Evaluation set size: {len(eval_dataset)}")

    # ========= Model =========
    model_args = argparse.Namespace(
        emb_nn=args.emb_nn,
        pointer=args.pointer,
        head=args.head,
        emb_dims=args.emb_dims,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        ff_dims=args.ff_dims,
        dropout=args.dropout,
        cycle=args.cycle,
        dgcnn_k=args.dgcnn_k,
    )
    net = DCP(model_args).to(args.device)

    if args.pretrained is not None:
        if not os.path.isfile(args.pretrained):
            raise FileNotFoundError(
                f"Checkpoint pretrained non trovato: {args.pretrained}"
            )

        pretrained_checkpoint = torch.load(
            args.pretrained,
            map_location=args.device,
            weights_only=False,
        )

        net.load_state_dict(
            pretrained_checkpoint['state_dict'],
            strict=True,
        )

        textio.cprint(
            f"Pesi iniziali caricati da: {args.pretrained}"
        )


    textio.cprint(str(net))
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    textio.cprint(f"Total parameters: {total_params}")

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=args.scheduler, gamma=0.5)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        saved = ckpt["model_args"]
        saved = (
            saved
            if isinstance(saved, dict)
            else vars(saved)
        )

        if (
            args.corr_weight != saved.get("corr_weight", 0.0)
            or args.corr_sigma_mm
            != saved.get("corr_sigma_mm", 5.0)
        ):
            raise ValueError(
                "Per cambiare corr_weight/corr_sigma_mm "
                "usa --pretrained con un nuovo esperimento, "
                "non --resume."
            )

        if ckpt.get("mesh_sha256", mesh_sha256) != mesh_sha256:
            raise ValueError(
                "La mesh di --resume non coincide "
                "con quella del checkpoint."
            )

        if "reference_phantom_mm" in ckpt:
            reference = (
                ckpt["reference_phantom_mm"]
                .detach()
                .cpu()
                .numpy()
            )

            if (
                reference.shape != (args.target_n_points, 3)
                or not np.isfinite(reference).all()
            ):
                raise ValueError(
                    "Digital twin del checkpoint incompatibile "
                    "con --target_n_points."
                )

            train_dataset.reference_phantom = reference.copy()
            eval_dataset.reference_phantom = reference.copy()
        net.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_eval_loss = ckpt.get('best_val_loss', ckpt.get('val_loss', float('inf')),)
        textio.cprint(f"Resumed from epoch {start_epoch} (val_loss={ckpt.get('val_loss', 'N/A')})")
    else:
        best_eval_loss = float('inf')

    writer = SummaryWriter(os.path.join('checkpoints', args.exp_name, 'logs'))

    # ========= Training Loop =========
    for epoch in range(start_epoch, args.epochs):
        textio.cprint(f"\nEpoch [{epoch+1}/{args.epochs}] (LR: {optimizer.param_groups[0]['lr']:.6f})")

        # Train
        train_metrics = train_one_epoch(args, net, train_loader, optimizer, textio, epoch)
        for k, v in train_metrics.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(f'train/{k}', v, epoch)
                textio.cprint(f"  train_{k}: {v:.6f}" if isinstance(v, float) else f"  train_{k}: {v}")

        # Validate
        val_metrics = validate_one_epoch(args, net, eval_loader, textio, epoch)
        for k, v in val_metrics.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(f'val/{k}', v, epoch)
                textio.cprint(f"  val_{k}: {v:.6f}" if isinstance(v, float) else f"  val_{k}: {v}")

        scheduler.step()

        # Salva sia l'ultima epoca sia la migliore sulla validation.
        is_best = val_metrics['val_loss'] < best_eval_loss

        if is_best:
            best_eval_loss = val_metrics['val_loss']

        checkpoint_state = {
            'state_dict': net.state_dict(),
            'model_args': args,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'network_scale_mm': args.network_scale_mm,
            'val_loss': val_metrics['val_loss'],
            'best_val_loss': best_eval_loss,
            'epoch': epoch,
            'data_protocol': 'partial_to_global_fixed_fps_v1',
            'selection_metric': 'val_loss_pose_cycle',
            'corr_weight': args.corr_weight,
            'corr_sigma_mm': args.corr_sigma_mm,
            'reference_phantom_mm': torch.from_numpy(
                train_dataset.reference_phantom.copy()
            ),
            'mesh_sha256': mesh_sha256,
        }

        model_dir = os.path.join(
            'checkpoints',
            args.exp_name,
            'models',
        )

        torch.save(
            checkpoint_state,
            os.path.join(model_dir, 'model.last.t7'),
        )

        if is_best:
            torch.save(
                checkpoint_state,
                os.path.join(model_dir, 'model.best.t7'),
            )

            textio.cprint(
                f"  >> Best model saved! "
                f"(val_loss={best_eval_loss:.6f})"
            )

        gc.collect()

    writer.close()
    textio.cprint(f"\nTraining complete. Best val_loss: {best_eval_loss:.6f}")
    textio.close()


if __name__ == '__main__':
    main()

