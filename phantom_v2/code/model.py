#!/usr/bin/env python
# -*- coding: utf-8 -*-


import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import quat2mat


# Part of the code is referred from: http://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding

# Cloning modules 
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# Scaled dot product attention
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn

# NN to find the closest point in dst for every point in src. Expading squared eucldean distance, building the esplicit distances matrix.
# topk(k=1) chooses the minimum for each point
def nearest_neighbor(src, dst):
    inner = -2 * torch.matmul(src.transpose(1, 0).contiguous(), dst)  # src, dst (num_dims, num_points)
    distances = -torch.sum(src ** 2, dim=0, keepdim=True).transpose(1, 0).contiguous() - inner - torch.sum(dst ** 2,
                                                                                                           dim=0,
                                                                                                           keepdim=True)
    distances, indices = distances.topk(k=1, dim=-1)
    return distances, indices

# KNN to find k nearest points closer to a single point, building a graph of the point cloud.
def knn(x, k=20):
    """
    x: [B, C, N]
    Restituisce gli indici dei vicini: [B, N, k_eff].

    Il punto stesso è incluso, come nell'implementazione esistente.
    """
    num_points = x.size(2)

    if num_points == 0:
        raise ValueError("Il point cloud non può essere vuoto.")
    if k < 1:
        raise ValueError("k deve essere almeno 1.")

    k_eff = min(k, num_points)

    inner = -2 * torch.matmul(
        x.transpose(2, 1), x
    )
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    negative_distances = -xx - inner - xx.transpose(2, 1)

    return negative_distances.topk(
        k=k_eff, dim=-1
    ).indices


def get_graph_feature(x, k=20):
    """
    x: [B, C, N]
    Output: [B, 2*C, N, k_eff]

    Ogni feature contiene [vicino, centro], mantenendo
    la rappresentazione usata nel DCP.
    """
    idx = knn(x, k=k)
    batch_size, num_points, k_eff = idx.shape
    num_dims = x.size(1)

    # Offset per distinguere i punti dei diversi elementi del batch.
    idx_base = (
        torch.arange(batch_size, device=x.device)
        .view(-1, 1, 1) * num_points
    )
    flat_idx = (idx + idx_base).reshape(-1)

    points = x.transpose(2, 1).contiguous()  # [B, N, C]

    neighbors = points.reshape(
        batch_size * num_points, num_dims
    )[flat_idx]

    neighbors = neighbors.reshape(
        batch_size, num_points, k_eff, num_dims
    )

    centers = points.unsqueeze(2).expand(
        -1, -1, k_eff, -1
    )

    return torch.cat(
        (neighbors, centers), dim=-1
    ).permute(0, 3, 1, 2).contiguous()


# EncoderDecoder follows the original "Attention is All You Need" design pattern,
# separating Encoder, Decoder, and the orchestrating wrapper into distinct classes.
# This allows reusing the same model instance with swapped roles:
# calling self.model(src, tgt) makes A encode and B decode (cross-attend to A),
# while calling self.model(tgt, src) makes B encode and A decode (cross-attend to B).
# src_embed, tgt_embed, and generator are empty Sequential() because DGCNN already
# produces the embeddings upstream, and SVDHead handles the final transformation —
# EncoderDecoder is only responsible for the attention-based feature refinement.


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture (Transformer). Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    # Memory is the encoder's output, using cross attention to test point cloud A comprehension
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))


class Generator(nn.Module):
    def __init__(self, emb_dims):
        super(Generator, self).__init__()
        self.nn = nn.Sequential(nn.Linear(emb_dims, emb_dims // 2),
                                nn.BatchNorm1d(emb_dims // 2),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 2, emb_dims // 4),
                                nn.BatchNorm1d(emb_dims // 4),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 4, emb_dims // 8),
                                nn.BatchNorm1d(emb_dims // 8),
                                nn.ReLU())
        self.proj_rot = nn.Linear(emb_dims // 8, 4)
        self.proj_trans = nn.Linear(emb_dims // 8, 3)

    def forward(self, x):
        x = self.nn(x.max(dim=1)[0]) # Max pooling to collapse all points in a single vector
        rotation = self.proj_rot(x) # Predicting rotation
        translation = self.proj_trans(x) # Predicting translation
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        return rotation, translation


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N) # Stacking clones of its layer
        self.norm = LayerNorm(layer.size) # Final layer norm

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N) # Stiacking clones of its layer
        self.norm = LayerNorm(layer.size) # Final layer norm

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    # Normalizing on the last dimension, a_2 and b_2 are learnable 
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=None):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)

    # Residual Connections + Layer norm of original Transformer
    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x)) # Residual Connection


class EncoderLayer(nn.Module):
    # Self attention on Point Cloud A and feedforward
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    # Self attention on Point Cloud B, cross-attention and feedforward
    
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

# Each head learns attention of a different geometry feature. Space dimention = 128. Unsupervised Learning
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4) # 4 identical linear projections (for Q, K, V and output)
        self.attn = None
        self.dropout = None

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

# 2 layer MLP applied independently on each point. Standand Transformer feed forward block
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.norm = nn.Sequential()  # nn.BatchNorm1d(d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = None

    def forward(self, x):
        return self.w_2(self.norm(F.relu(self.w_1(x)).transpose(2, 1).contiguous()).transpose(2, 1).contiguous())


class PointNet(nn.Module):
    def __init__(self, emb_dims=512):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False) # 3 = xyz channels
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(emb_dims)
        # 5 1D convolutions with kernel = 1, matematically identical to a Linear applied point by point

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        return x

def make_group_norm(num_channels, max_groups=8):
    """
    Crea una GroupNorm usando il maggior numero di gruppi,
    fino a max_groups, che divida esattamente i canali.
    """
    for num_groups in (max_groups, 4, 2, 1):
        if num_channels % num_groups == 0:
            return nn.GroupNorm(num_groups, num_channels)

    return nn.GroupNorm(1, num_channels)

class DGCNN(nn.Module):
    # The network processes a 2D information (point, neighbor)
    def __init__(self, emb_dims=512, k=20):
        super(DGCNN, self).__init__()
        self.k = k
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
        self.bn1 = make_group_norm(64)
        self.bn2 = make_group_norm(64)
        self.bn3 = make_group_norm(128)
        self.bn4 = make_group_norm(256)
        self.bn5 = make_group_norm(emb_dims)

    def forward(self, x):
        batch_size, num_dims, num_points = x.size() 
        x = get_graph_feature(x, k=self.k) # Obtaining edge features
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn2(self.conv2(x))) # Conv2 on edge features, to learn abstract features
        x2 = x.max(dim=-1, keepdim=True)[0] # Max Pooling to aggregate info of k nearest in a single vector per point

        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=True)[0]

        # Concat is needed so the final embedding contains simultaneously information at 4 abstraction levels
        # x1 → low level features, raw geometry
        # x2, x3 → intermediate features
        # x4 → high level features, semantic 
        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
        return x

# Alternative to SVDHead, predicts (R, t) via quaternion direct regression. MLPHead has no the corrispondance of A points and B points. 
# The network learns the implicit transformation.
class MLPHead(nn.Module):
    def __init__(self, args):
        super(MLPHead, self).__init__()
        emb_dims = args.emb_dims
        self.emb_dims = emb_dims
        self.nn = nn.Sequential(nn.Linear(emb_dims * 2, emb_dims // 2),
                                nn.BatchNorm1d(emb_dims // 2),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 2, emb_dims // 4),
                                nn.BatchNorm1d(emb_dims // 4),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 4, emb_dims // 8),
                                nn.BatchNorm1d(emb_dims // 8),
                                nn.ReLU())
        self.proj_rot = nn.Linear(emb_dims // 8, 4)
        self.proj_trans = nn.Linear(emb_dims // 8, 3)

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        embedding = torch.cat((src_embedding, tgt_embedding), dim=1)
        embedding = self.nn(embedding.max(dim=-1)[0]) # Summarizing all the point cloud in a global vector
        # Regression step
        rotation = self.proj_rot(embedding)
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        translation = self.proj_trans(embedding)
        return quat2mat(rotation), translation


class Identity(nn.Module):

    """
    No-op pointer used in DCP-v1: returns the embeddings unchanged.
    When summed back in DCP.forward, the effect is a uniform scaling
    of the embeddings — which does not affect the soft correspondences
    in SVDHead since those depend only on relative similarity scores.
    This allows DCP-v1 (no inter-cloud communication) and DCP-v2
    (Transformer cross-attention) to share the same forward pass
    by swapping only the pointer module.
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, *input):
        return input

# DCP-v2 pointer (each A point sees B points before matching)
class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.emb_dims = args.emb_dims
        self.N = args.n_blocks
        self.dropout = args.dropout
        self.ff_dims = args.ff_dims
        self.n_heads = args.n_heads
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.emb_dims)
        ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)
        # The Encoder and the Decoder must have different weights
        self.model = EncoderDecoder(Encoder(EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N),
                                    Decoder(DecoderLayer(self.emb_dims, c(attn), c(attn), c(ff), self.dropout), self.N),
                                    nn.Sequential(),
                                    nn.Sequential(),
                                    nn.Sequential())

    # The model is called twice with swapped src/tgt to produce context-aware embeddings
    # for both point clouds. Each call lets one cloud encode its structure while the other
    # cross-attends to it, so every point embedding is informed by the other cloud.
    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src = src.transpose(2, 1).contiguous()
        tgt = tgt.transpose(2, 1).contiguous()
        # Every A point obtains an updated embedding which contains B information, and viceversa. The embedding are aware of the 
        # other point cloud
        tgt_embedding = self.model(src, tgt, None, None).transpose(2, 1).contiguous()
        src_embedding = self.model(tgt, src, None, None).transpose(2, 1).contiguous()
        return src_embedding, tgt_embedding

# Transforming embeddings in a rigid transformation (R, t) via SVD
class SVDHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.emb_dims = args.emb_dims

        # Costante geometrica, non parametro da addestrare.
        # Manteniamo il nome "reflect" per i checkpoint esistenti.
        reflect = torch.eye(3)
        reflect[2, 2] = -1
        self.register_buffer("reflect", reflect)

    def forward(
        self,
        src_embedding,
        tgt_embedding,
        src,
        tgt,
        return_correspondence=False,
    ):
        """
        src_embedding: [B, D, Ns]
        tgt_embedding: [B, D, Nt]
        src: [B, 3, Ns]
        tgt: [B, 3, Nt]

        Output:
            R: [B, 3, 3]
            t: [B, 3]

        La trasformazione stimata soddisfa:
            tgt ≈ R @ src + t
        """
        d_k = src_embedding.size(1)

        # Corrispondenze probabilistiche source -> target.
        scores = torch.matmul(
            src_embedding.transpose(2, 1),
            tgt_embedding,
        ) / math.sqrt(d_k)

        weights = torch.softmax(scores, dim=-1)

        # Un punto target virtuale per ogni punto source.
        src_corr = torch.matmul(
            tgt, weights.transpose(2, 1)
        )

        src_mean = src.mean(dim=2, keepdim=True)
        corr_mean = src_corr.mean(dim=2, keepdim=True)

        src_centered = src - src_mean
        corr_centered = src_corr - corr_mean

        H = torch.matmul(
            src_centered,
            corr_centered.transpose(2, 1),
        )

        # torch.linalg.svd restituisce Vh = V^T.
        U, _, Vh = torch.linalg.svd(
            H, full_matrices=False
        )
        V = Vh.transpose(2, 1)
        Ut = U.transpose(2, 1)

        # Impone det(R) = +1, eliminando eventuali riflessioni.
        has_reflection = torch.det(V @ Ut) < 0

        identity = torch.eye(
            3, device=H.device, dtype=H.dtype
        )

        correction = torch.where(
            has_reflection[:, None, None],
            self.reflect.to(dtype=H.dtype),
            identity,
        )

        R = V @ correction @ Ut
        t = (corr_mean - R @ src_mean).squeeze(2)

        if return_correspondence:
            return R, t, scores

        return R, t

class DCP(nn.Module):
    def __init__(self, args):
        super(DCP, self).__init__()
        self.emb_dims = args.emb_dims
        self.cycle = args.cycle
        if args.emb_nn == 'pointnet':
            self.emb_nn = PointNet(emb_dims=self.emb_dims)
        elif args.emb_nn == 'dgcnn':
            dgcnn_k = getattr(args, 'dgcnn_k', 20)
            self.emb_nn = DGCNN(emb_dims=self.emb_dims, k=dgcnn_k)
        else:
            raise Exception('Not implemented')

        if args.pointer == 'identity':
            self.pointer = Identity()
        elif args.pointer == 'transformer':
            self.pointer = Transformer(args=args)
        else:
            raise Exception("Not implemented")

        if args.head == 'mlp':
            self.head = MLPHead(args=args)
        elif args.head == 'svd':
            self.head = SVDHead(args=args)
        else:
            raise Exception('Not implemented')

    def forward(self, *input, return_correspondence=False):
        src = input[0]
        tgt = input[1]

        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt)

        src_embedding_p, tgt_embedding_p = self.pointer(
            src_embedding,
            tgt_embedding,
        )

        src_embedding = src_embedding + src_embedding_p
        tgt_embedding = tgt_embedding + tgt_embedding_p

        if return_correspondence:
            if not isinstance(self.head, SVDHead):
                raise ValueError(
                    "Le corrispondenze richiedono head='svd'."
                )

            rotation_ab, translation_ab, logits = self.head(
                src_embedding,
                tgt_embedding,
                src,
                tgt,
                return_correspondence=True,
            )
        else:
            rotation_ab, translation_ab = self.head(
                src_embedding,
                tgt_embedding,
                src,
                tgt,
            )

        if self.cycle:
            rotation_ba, translation_ba = self.head(
                tgt_embedding,
                src_embedding,
                tgt,
                src,
            )
        else:
            rotation_ba = rotation_ab.transpose(
                2, 1
            ).contiguous()

            translation_ba = -torch.matmul(
                rotation_ba,
                translation_ab.unsqueeze(2),
            ).squeeze(2)

        outputs = (
            rotation_ab,
            translation_ab,
            rotation_ba,
            translation_ba,
        )

        if return_correspondence:
            return (*outputs, logits)

        return outputs
