# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import itertools
import torch
import torch.nn as nn
from pytorch_msssim import ms_ssim
from timm.models.vision_transformer import PatchEmbed, Block
# from util.block import block 自己设计的block
from util.pos_embed import get_2d_sincos_pos_embed

class Original_model(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))



        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def forward(self, x):
        pass

class MaskedAutoencoderViT(Original_model):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__(img_size, patch_size, in_chans,
                 embed_dim, depth, num_heads,
                 mlp_ratio, norm_layer)
        num_patches = self.patch_embed.num_patches
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

        #
        # self.dim = 300
        # self.head = nn.Linear(embed_dim, self.dim)
        # --------------------------------------------------------------------------

    def initialize_weights(self):

        super().initialize_weights()
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.mask_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)


    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        K = int(1 / (1 - mask_ratio))
        x_masked = torch.zeros([K, N, int(L / K), D], device=x.device)
        mask = torch.zeros([K, N, L], device=x.device)
        for i in range(K):
            step = int(L / K)
            ids = ids_shuffle[:, i * step:(i + 1) * step]
            x_masked[i] = torch.gather(x, dim=1, index=ids.unsqueeze(-1).repeat(1, 1, D))  # select patch
            # generate the binary mask: 0 is keep, 1 is remove
            m_i = torch.ones(N, L, device=x.device)
            m_i[:, i * step:(i + 1) * step] = 0
            mask[i] = torch.gather(m_i, dim=1, index=ids_restore)
        mask = mask.reshape(-1, L)
        x_masked = x_masked.reshape(-1, int(L / K), D)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # 记录cls
        cls_e = x[:, 0]
        # cls_e = self.head(cls_e)
        return x, mask, ids_restore, cls_e

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        # x N*K,L/K=l,D
        x = self.decoder_embed(x)
        NK, l, D = x.shape
        N, L = ids_restore.shape
        K = int(NK / N)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(N, L + 1 - l, 1)  # x里面还包括一个位置编码
        x_ = x.reshape(K, N, l, D)
        x_1 = torch.zeros(K, N, L, D, device=x.device)
        step = int(L / K)
        for i in range(K):
            x_tem = torch.cat((mask_tokens[:, 0:i * step, :], x_[i][:, 1:, :], mask_tokens[:, 0:(K - i) * step, :]),
                              dim=1)  # no cls token
            x_1[i] = torch.gather(x_tem, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        x = x_1.reshape(-1, L, D)  # 转化位原来的维度
        x = torch.cat([x[:, :1, :], x], dim=1)  # append cls token
        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        cls_d = x[:, 0]
        # cls_d = self.head(cls_d)
        # remove cls token
        x = x[:, 1:, :]

        return x, cls_d

    def forward_loss(self, imgs, pred, mask, cls_d):
        """
        imgs: [N, 3, H, W]
        pred: [KN, L, p*p*3]
        mask: [KN, L], 0 is keep, 1 is remove,
        target: [N, L, D]
        """
        target = self.patchify(imgs)  # [N, L, D]
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        data_range = 4.378
        weight = 0.15
        K = int(pred.shape[0] / target.shape[0])
        l2_loss_rec = (pred - target.repeat(K, 1, 1)) ** 2
        l2_loss_rec = l2_loss_rec.mean(-1)
        l2_loss_rec = (l2_loss_rec * mask).sum() / mask.sum()

        pred_cons = pred * (mask.unsqueeze(-1).repeat(1, 1, pred.shape[2]))
        pred_cons = pred_cons.reshape(-1, target.shape[0], pred.shape[1], pred.shape[2]).sum(0) / (
                K - 1)

        ms_ssim_batch_wise = 1 - ms_ssim(self.unpatchify(pred_cons),
                                         self.unpatchify(target), data_range=data_range, size_average=True, win_size=11)


        loss = weight * ms_ssim_batch_wise + (1 - weight) * l2_loss_rec

        return loss, pred, l2_loss_rec

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore, cls_e = self.forward_encoder(imgs, mask_ratio)
        pred_tem, cls_d = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss, pred1, loss_svd = self.forward_loss(imgs, pred_tem, mask, cls_d)
        return loss, loss_svd, pred1, mask,

class ViT(Original_model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Linear(768, 768)
        self.initialize_weights()
        self.R = 0.0
        self.c = None
        self.sigma = None


    def initialize_weights(self):
        super().initialize_weights()
        self.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        #x = self.head(x)
        return x

def mae_vit_shap_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=2, num_heads=16,
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_shap_patch16_dec512d8b(**kwargs):
    model = ViT(
        patch_size=16, embed_dim=768, depth=2, num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch16_dec512d8b(**kwargs):
    model = ViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# set recommended archs
mae_vit_shap_patch16 = mae_vit_shap_patch16_dec512d8b
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
vit_shap_patch16 = vit_shap_patch16_dec512d8b
vit_base_patch16 = vit_base_patch16_dec512d8b