import numpy as np
import torch
from loma import LoMa as LoMaModel
from loma.loma import LoMaB, LoMaL, LoMaG, filter_matches, to_pixel_coords



class LoMa:
    def __init__(self, variant="B", num_feats=4096):
        if variant == 'L':
            self.model = LoMaModel(LoMaL())
        elif variant == 'G':
            self.model = LoMaModel(LoMaG())
        else:
            self.model = LoMaModel(LoMaB())
        self.num_feats = num_feats

    def extract(self, img_path):
        kpts, descs, h, w = self.model.detect_and_describe(str(img_path), num_keypoints=self.num_feats)
        return kpts, descs, h, w

    def to_pixel_coords(self, kpts, h, w):
        kpts = to_pixel_coords(kpts[0], h, w).cpu().numpy()
        return kpts

    def match(self, img1_path, img2_path):
        kpts1, kpts2 = self.model.match(img1_path, img2_path)
        return kpts1, kpts2

    def match_from_feats(self, kpts1, kpts2, desc1, desc2):
        with torch.inference_mode():
            scores = self.model(kpts1, kpts2, desc1, desc2)['scores']
        m0, m1, mscores0, mscores1 = filter_matches(scores, self.model.cfg.filter_threshold)
        valid = m0[0] > -1
        # matched1 = to_pixel_coords(kpts1[0][torch.where(valid)[0]], h1, w1).cpu().numpy()
        # matched2 = to_pixel_coords(kpts2[0][m0[0][valid]], h2, w2).cpu().numpy()
        # return matched1, matched2
        if valid.sum() > 0:
            match_indices1 = np.arange(kpts1.shape[1])[valid.cpu().numpy()]
            match_indices2 = np.arange(kpts2.shape[1])[m0[0, valid].cpu().numpy()]
            matches = np.concatenate((match_indices1.reshape(-1,1), match_indices2.reshape(-1,1)), axis=1)
        else:
            matches = np.zeros((1, 2))
        return matches

    def __call__(self, image: np.ndarray):
        return

