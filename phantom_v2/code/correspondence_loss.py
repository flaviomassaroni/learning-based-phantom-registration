"""Supervisione geometrica source -> target, usata solo con GT disponibile."""

import math
import torch
import torch.nn.functional as F


def geometric_correspondence_loss(
    logits,
    src,
    target,
    rotation_gt,
    translation_gt,
    network_scale_mm,
    sigma_mm,
):
    """
    KL(q_GT || p_rete), mediata su batch e punti source.

    Le coordinate sono in unità rete; sigma_mm è in millimetri.
    La GT costruisce le etichette della loss, non gli input della rete.
    """
    if not math.isfinite(network_scale_mm) or network_scale_mm <= 0:
        raise ValueError(
            "network_scale_mm deve essere positivo e finito."
        )

    if not math.isfinite(sigma_mm) or sigma_mm <= 0:
        raise ValueError(
            "corr_sigma_mm deve essere positivo e finito."
        )

    expected = (
        src.shape[0],
        src.shape[2],
        target.shape[2],
    )

    if tuple(logits.shape) != expected:
        raise ValueError(
            f"Logits {tuple(logits.shape)}, attesi {expected}."
        )

    with torch.no_grad():
        aligned = (
            rotation_gt @ src
            + translation_gt.unsqueeze(-1)
        )

        distances = torch.cdist(
            aligned.transpose(1, 2),
            target.transpose(1, 2),
        )

        sigma_network = sigma_mm / network_scale_mm
        geometry_logits = -0.5 * (
            distances / sigma_network
        ) ** 2

        log_q = F.log_softmax(geometry_logits, dim=-1)
        q = log_q.exp()

    log_p = F.log_softmax(logits, dim=-1)

    return (q * (log_q - log_p)).sum(dim=-1).mean()