import numpy as np
import poselib
from copy import deepcopy


class RelPose:
    def __init__(self, s=1.0, R=np.eye(3), t=np.zeros(3), K1=np.eye(3), K2=np.eye(3), inliers_mask=None, inliers_ratio=0.0):
        """Relative pose from first image to second image
        scale: scale factor to correct second image's depths
        R, t: rotation and translation to transform first image to second image
        K1, K2: intrinsic of first and second image
        inliers_mask: boolean mask of inliers
        inliers_ratio: inliers ratio
        """
        self.scale = s
        self.rotation = R
        self.translation = t
        self.intrinsic1 = K1
        self.intrinsic2 = K2
        self.inliers_mask = inliers_mask
        self.inliers_ratio = inliers_ratio
        self.inliers = np.where(self.inliers_mask)[0] if self.inliers_mask is not None else []


class RePoseDEstimator:
    def __init__(self, max_epipolar_error=1.0, max_reproj_error=12.0):
        self.poselib_cam_dict = {
            "model": "SIMPLE_PINHOLE", "width": -1, "height": -1,
            "params": [0, 0, 0],    # [focal, cu, cv]
        }
        ransac_dict = {
            "max_iterations": 10000,
            "min_iterations": 1000,
            "progressive_sampling": False
        }
        bundle_dict = {
            'max_iterations': 100,
            'verbose': False,
            'loss_type': 'TRUNCATED_CAUCHY',
            # 'loss_type': 'CAUCHY'
        }
        self.monodepth_dict = {
            "max_errors": [max_reproj_error, max_epipolar_error],
            "estimate_shift": False,
            "ransac": ransac_dict,
            "bundle": bundle_dict,
            # "weight_sampson": -1.0
        }

    def solve(self, points1, points2, depths1, depths2, im1_size, im2_size, intrinsic1=None, intrinsic2=None, shared=True):
        """
        Arguments:
            points1, points2: matched pixel coordinates
            depths1, depths2: corresponding depths for matched pixels
            intrinsic1, intrinsic2: camera intrinsic for matched images
            shared: shared camera intrinsic for matched images or not
        Returns:
            scale: depth scale factor to correct the depths in second image
            R: rotation to transform the points in first to second image
            t: translation to transform the points in first to second image
        """
        points1, points2 = points1.astype(np.float64), points2.astype(np.float64)
        depths1, depths2 = depths1.squeeze().astype(np.float64), depths2.squeeze().astype(np.float64)

        # known focals
        if intrinsic1 is not None and intrinsic2 is not None:
            geometry, info = self.solve_pose_with_known_focals(points1, points2, depths1, depths2, intrinsic1, intrinsic2)
        # unknown focals
        else:
            # assume principal point at image center
            h1, w1 = im1_size
            h2, w2 = im2_size
            pp1 = np.array([w1/2, h1/2], dtype=np.float64)
            pp2 = np.array([w2/2, h2/2], dtype=np.float64)
            points1 -= pp1
            points2 -= pp2
            # shared unknown focals
            if shared:
                image_pair, info = self.solve_pose_with_shared_unknown_focal(points1, points2, depths1, depths2)
            # two unknown focals
            else:
                image_pair, info = self.solve_pose_with_unknown_focals(points1, points2, depths1, depths2)

            geometry = image_pair.geometry
            intrinsic1 = np.eye(3)
            intrinsic2 = np.eye(3)
            intrinsic1[:2, :2] *= image_pair.camera1.focal()
            intrinsic1[:2, -1] = pp1
            intrinsic2[:2, :2] *= image_pair.camera2.focal()
            intrinsic2[:2, -1] = pp2

        return RelPose(geometry.scale, geometry.pose.R, geometry.pose.t.flatten(), intrinsic1, intrinsic2, info['inliers'], info['inlier_ratio'])

    def solve_pose_with_known_focals(self, points1, points2, depths1, depths2, intrinsic1, intrinsic2):
        """Estimate relative pose of two images taken by calibrated cameras
        """
        # intrinsic1, intrinsic2 = intrinsic1.astype(np.float64), intrinsic2.astype(np.float64)
        camera1 = deepcopy(self.poselib_cam_dict)
        camera1["params"] = [intrinsic1[0, 0], intrinsic1[0, 2], intrinsic1[1, 2]]
        camera2 = deepcopy(self.poselib_cam_dict)
        camera2['params'] = [intrinsic2[0, 0], intrinsic2[0, 2], intrinsic2[1, 2]]

        # geometry, info = poselib.estimate_monodepth_relative_pose(points1, points2, depths1, depths2, camera1, camera2, self.ransac_dict, self.ba_dict)
        geometry, info = poselib.estimate_monodepth_relative_pose(points1, points2, depths1, depths2, camera1, camera2, self.monodepth_dict)
        # image_pair = poselib.MonoDepthImagePair(geometry, camera1, camera2)
        return geometry, info

    def solve_pose_with_shared_unknown_focal(self, points1, points2, depths1, depths2):
        """Estimate relative pose of two images taken by a single uncalibrated camera
        """
        # image_pair, info = poselib.estimate_monodepth_shared_focal_relative_pose(points1, points2, depths1, depths2, self.ransac_dict, self.ba_dict)
        image_pair, info = poselib.estimate_monodepth_shared_focal_relative_pose(points1, points2, depths1, depths2, self.monodepth_dict)
        return image_pair, info

    def solve_pose_with_unknown_focals(self, points1, points2, depths1, depths2):
        """Estimate relative pose of two images taken by different uncalibrated cameras
        """
        # image_pair, info = poselib.estimate_monodepth_varying_focal_relative_pose(points1, points2, depths1, depths2, self.ransac_dict, self.ba_dict)
        image_pair, info = poselib.estimate_monodepth_varying_focal_relative_pose(points1, points2, depths1, depths2, self.monodepth_dict)
        return image_pair, info