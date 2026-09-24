import numpy as np
import cv2

EPS = 1e-7

class SIFT:
    def __init__(self, num_feats=4096):
        self.descriptor = cv2.SIFT_create(num_feats)

    def root_sift(self, descs):
        """Apply the Hellinger kernel by first L1-normalizing, taking the square-root, and then l2-normalizing"""
        descs /= (descs.sum(axis=1, keepdims=True) + EPS)
        descs = np.sqrt(descs)
        return descs

    def to_gray(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def __call__(self, image: np.ndarray):
        gray = self.to_gray(image)
        feats, des = self.descriptor.detectAndCompute(gray, None) # (#num_kpts), (#num_kpts, num_feats)
        des = self.root_sift(des)   # apply normalization (rootSIFT)
        scores = np.ones(len(feats), dtype=np.float16)
        return np.array([feat.pt for feat in feats]).astype(np.uint32), des, scores



if __name__ == '__main__':
    from globalsfm.utils.io import read_image
    model = SIFT()
    image = read_image("./assets/")