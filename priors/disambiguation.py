import numpy as np
import torch
import argparse
from tqdm import tqdm
from scipy.special import softmax
from dgsfm.database.config import DGSfMConfig, load_dgsfm_config
from dgsfm.database.dataset import COLMAPDataset
from dgsfm.utils.fileio import *

import sys
sys.path.insert(0, "third_party")
from mast3r.model import AsymmetricMASt3R
from mast3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs



class Disambiguation:
    def __init__(self, cfg: DGSfMConfig) -> None:
        self.cfg = cfg
        pretrained = "third_party/checkpoints/checkpoint-dg.pth"
        self.method = AsymmetricMASt3R(pos_embed='RoPE100', patch_embed_cls='ManyAR_PatchEmbed',
                        img_size=(512, 512), head_type='catmlp+dpt', head_type_dg='transformer',
                        output_mode='pts3d+desc24', output_mode_dg='dg_score',
                        depth_mode=('exp', -np.inf, np.inf), conf_mode=('exp', 1, np.inf),
                        enc_embed_dim=1024, enc_depth=24, enc_num_heads=16,
                        dec_embed_dim=768, dec_depth=12, dec_num_heads=12,
                        two_confs=True, desc_conf_mode=('exp', 0, np.inf),
                        add_dg_pred_head=True, freeze=['mask','encoder','decoder','head']).from_pretrained(pretrained).to(cfg.device)

    def run(self, dataset: COLMAPDataset):
        for scene in dataset:
            print(scene.scene_name)
            image_pairs = scene.image_pairs
            pair_scores = []
            for pair_idx in tqdm(range(len(image_pairs))):
                image_file1, image_file2 = image_pairs[pair_idx]
                img_paths = [str(scene.image_dir / img) for img in [image_file1, image_file2]]
                images = load_images(img_paths, size=512, verbose=False)
                output = inference(make_pairs(images), self.method, self.cfg.device, verbose=False)
                pred1, pred2 = output['pred1'], output['pred2']
                pred1 = torch.stack(pred1, dim=0) if isinstance(pred1, list) else pred1
                pred2 = torch.stack(pred2, dim=0) if isinstance(pred2, list) else pred2

                score_s1 = softmax(pred1.detach().cpu().numpy(), axis=1)
                score_s2 = softmax(pred2.detach().cpu().numpy(), axis=1)

                vote_0 = sum(score_s1[:,0] > score_s1[:,1]) + sum(score_s2[:,0] > score_s2[:,1])
                vote_1 = sum(score_s1[:,1] > score_s1[:,0]) + sum(score_s2[:,1] > score_s2[:,0])
                if vote_1 > vote_0:
                    score = np.max((score_s1[:,1], score_s2[:,1]))
                elif vote_1 < vote_0:
                    score = np.min((score_s1[:,1], score_s2[:,1]))
                else:
                    score = np.mean((score_s1[:,1], score_s2[:,1]))

                pair_scores.append(score)
            save_image_pairs_with_scores(image_pairs, pair_scores, scene.doppelgangers_pairs_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="Path to the configuration file", default="configs/eth3d.yaml")
    args = parser.parse_args()
    cfg = load_dgsfm_config(args.config)
    dataset = COLMAPDataset(cfg)
    disamb = Disambiguation(cfg)
    disamb.run(dataset)
