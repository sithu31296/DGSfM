# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# MASt3R model class
# --------------------------------------------------------
import torch
import torch.nn.functional as F
import os

from mast3r.catmlp_dpt_head import mast3r_head_factory

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.model import AsymmetricCroCo3DStereo  # noqa
from dust3r.utils.misc import transpose_to_landscape, freeze_all_params  # noqa


inf = float('inf')


def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class AsymmetricMASt3R(AsymmetricCroCo3DStereo):
    def __init__(self, desc_mode=('norm'), two_confs=False, desc_conf_mode=None, add_dg_pred_head=False, output_mode_dg='dg_score', head_type_dg='transformer', **kwargs):
        self.desc_mode = desc_mode
        self.two_confs = two_confs
        self.desc_conf_mode = desc_conf_mode
        self.add_dg_pred_head = add_dg_pred_head
        self.output_mode_dg = output_mode_dg
        self.head_type_dg = head_type_dg
        super().__init__(**kwargs)

        if add_dg_pred_head:
            self.set_dg_downstream_head(head_type_dg, output_mode_dg)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            return super(AsymmetricMASt3R, cls).from_pretrained(pretrained_model_name_or_path, **kw)

    # @classmethod
    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none': [],
            'mask': [self.mask_token],
            'encoder': [self.patch_embed, self.enc_blocks],
            'decoder': [self.dec_blocks, self.dec_blocks2, self.dec_norm, self.decoder_embed],
            'head': [self.downstream_head1, self.downstream_head2] if self.add_dg_pred_head else [self.head1, self.head2],
        }
        if isinstance(freeze, str):
            freeze_all_params(to_be_frozen[freeze])
        elif isinstance(freeze, list):
            for f in freeze:
                freeze_all_params(to_be_frozen[f])

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size, **kw):
        assert img_size[0] % patch_size == 0 and img_size[
            1] % patch_size == 0, f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        if self.desc_conf_mode is None:
            self.desc_conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = mast3r_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = mast3r_head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def set_dg_downstream_head(self, head_type_dg, output_mode_dg, **kw):
        self.output_mode_dg = output_mode_dg
        self.head_type_dg = head_type_dg
        self.head3 = mast3r_head_factory(head_type=head_type_dg, output_mode=output_mode_dg, net=self)
        self.head4 = mast3r_head_factory(head_type=head_type_dg, output_mode=output_mode_dg, net=self)


    def forward(self, view1, view2):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)
            if self.add_dg_pred_head:
                dopp_pred1 = self._downstream_head(3, [tok.float() for tok in dec1], shape1)
                dopp_pred2 = self._downstream_head(4, [tok.float() for tok in dec2], shape2)

        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        if self.add_dg_pred_head:
            return res1, res2, dopp_pred1, dopp_pred2
        else:
            return res1, res2