import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from correspondence_loss import geometric_correspondence_loss
from model import DCP
from phantom_data import PhantomDataset
from train import train_one_epoch, validate_one_epoch


def small_args(**overrides):
    values = dict(
        emb_nn="dgcnn",
        emb_dims=16,
        pointer="transformer",
        head="svd",
        n_blocks=1,
        n_heads=4,
        ff_dims=32,
        dropout=0.0,
        cycle=False,
        dgcnn_k=4,
        device="cpu",
        grad_clip=1.0,
        network_scale_mm=100.0,
        corr_weight=0.1,
        corr_sigma_mm=5.0,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


class CorrespondenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)
        self.src = torch.randn(2, 3, 16) * 0.1
        self.tgt = torch.randn(2, 3, 24) * 0.15
        self.r = torch.eye(3).expand(2, -1, -1).clone()
        self.t = torch.zeros(2, 3)

    def loss(self, logits, src=None, tgt=None, scale=100.0):
        return geometric_correspondence_loss(
            logits,
            self.src if src is None else src,
            self.tgt if tgt is None else tgt,
            self.r,
            self.t,
            scale,
            5.0,
        )

    def test_forward_and_checkpoint(self):
        net = DCP(small_args()).eval()

        with torch.no_grad():
            plain = net(self.src, self.tgt)
            aux = net(
                self.src, self.tgt,
                return_correspondence=True,
            )

        self.assertEqual(len(plain), 4)
        self.assertEqual(tuple(aux[4].shape), (2, 16, 24))

        for a, b in zip(plain, aux[:4]):
            torch.testing.assert_close(a, b)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.t7"
            torch.save(net.state_dict(), path)

            restored = DCP(small_args()).eval()
            restored.load_state_dict(
                torch.load(path, weights_only=True),
                strict=True,
            )

            with torch.no_grad():
                result = restored(self.src, self.tgt)

            for a, b in zip(plain, result):
                torch.testing.assert_close(a, b)

    def test_geometric_optimum_and_scale(self):
        distances = torch.cdist(
            self.src.transpose(1, 2),
            self.tgt.transpose(1, 2),
        )
        logits = -0.5 * (distances / 0.05) ** 2

        self.assertLess(abs(self.loss(logits).item()), 1e-6)
        self.assertGreater(
            self.loss(torch.zeros_like(logits)).item(),
            0.1,
        )

        random_logits = torch.randn_like(logits)

        torch.testing.assert_close(
            self.loss(random_logits),
            self.loss(
                random_logits,
                self.src * 2,
                self.tgt * 2,
                scale=50.0,
            ),
        )

    def test_gradients_only_through_prediction(self):
        logits = torch.randn(2, 16, 24, requires_grad=True)

        for value in (self.src, self.tgt, self.r, self.t):
            value.requires_grad_(True)

        self.loss(logits).backward()

        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad.abs().sum().item(), 0)

        for value in (self.src, self.tgt, self.r, self.t):
            self.assertIsNone(value.grad)

    def test_permutation_invariance(self):
        net = DCP(small_args()).eval()
        p = torch.randperm(16)
        q = torch.randperm(24)

        with torch.no_grad():
            normal = net(
                self.src, self.tgt,
                return_correspondence=True,
            )
            permuted = net(
                self.src[:, :, p],
                self.tgt[:, :, q],
                return_correspondence=True,
            )

        for a, b in zip(normal[:4], permuted[:4]):
            torch.testing.assert_close(
                a, b, atol=2e-4, rtol=2e-4,
            )

        torch.testing.assert_close(
            self.loss(normal[4]),
            self.loss(
                permuted[4],
                self.src[:, :, p],
                self.tgt[:, :, q],
            ),
            atol=2e-5,
            rtol=2e-5,
        )

    def test_backward_with_and_without_cycle(self):
        for cycle in (False, True):
            net = DCP(small_args(cycle=cycle))
            result = net(
                self.src, self.tgt,
                return_correspondence=True,
            )

            objective = ((result[0] - self.r) ** 2).mean()
            objective = (
                objective
                + (result[1] ** 2).mean()
                + 0.1 * self.loss(result[4])
            )

            if cycle:
                objective = objective + 0.1 * (
                    (result[2] @ result[0] - self.r) ** 2
                ).mean()

            objective.backward()

            gradients = [
                p.grad for p in net.parameters()
                if p.grad is not None
            ]

            self.assertTrue(gradients)
            self.assertTrue(
                all(torch.isfinite(g).all() for g in gradients)
            )
            self.assertGreater(
                sum(g.abs().sum().item() for g in gradients),
                0,
            )

    def test_epoch_metrics_and_disabled_loss(self):
        src = torch.randn(5, 3, 16) * 0.1
        tgt = torch.randn(5, 3, 24) * 0.15
        r = torch.eye(3).expand(5, -1, -1).clone()
        t = torch.zeros(5, 3)

        loader = DataLoader(
            TensorDataset(src, tgt, r, t),
            batch_size=2,
        )

        for weight in (0.0, 0.1):
            args = small_args(corr_weight=weight)
            net = DCP(args)
            optimizer = torch.optim.Adam(
                net.parameters(), lr=0.0,
            )

            before = {
                k: v.clone()
                for k, v in net.state_dict().items()
            }

            training = train_one_epoch(
                args, net, loader, optimizer, None, 0,
            )
            validation = validate_one_epoch(
                args, net, loader, None, 0,
            )

            self.assertTrue(
                all(np.isfinite(v) for v in training.values())
            )
            self.assertAlmostEqual(
                training["train_loss"],
                validation["val_loss"],
                places=5,
            )
            self.assertAlmostEqual(
                training["train_total_loss"],
                training["train_loss"]
                + weight * training["train_corr_loss"],
                places=5,
            )

            single = validate_one_epoch(
                args,
                net,
                DataLoader(loader.dataset, batch_size=1),
                None,
                0,
            )

            self.assertAlmostEqual(
                validation["val_loss"],
                single["val_loss"],
                places=4,
            )

            for k, value in net.state_dict().items():
                torch.testing.assert_close(value, before[k])

            if weight == 0:
                self.assertEqual(
                    training["train_corr_loss"], 0.0,
                )

    def test_dataset_reference_and_validation(self):
        import trimesh

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "sphere.stl")
            trimesh.creation.icosphere(
                subdivisions=2, radius=30,
            ).export(path)

            settings = dict(
                stl_path=path,
                mode="sparse",
                num_samples=2,
                n_points=16,
                target_n_points=32,
                noise_sigma=0,
            )

            first = PhantomDataset(**settings, seed=42)
            second = PhantomDataset(**settings, seed=43)

            np.testing.assert_array_equal(
                first.reference_phantom,
                second.reference_phantom,
            )

            a = first[0]
            b = first[0]

            for x, y in zip(a, b):
                np.testing.assert_array_equal(x, y)

            self.assertEqual(a[0].shape, (3, 16))
            self.assertEqual(a[1].shape, (3, 32))

            for value in (0, -1, float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    PhantomDataset(
                        **settings,
                        network_scale_mm=value,
                    )
                with self.assertRaises(ValueError):
                    PhantomDataset(
                        **settings,
                        patch_radius_mm=value,
                    )

    def test_evaluation_math(self):
        from evaluate import test_evaluation_math
        test_evaluation_math()


if __name__ == "__main__":
    unittest.main()