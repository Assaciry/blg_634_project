from mf3D.db import SfMDatabase
import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import List, Tuple
import cv2
from tqdm import tqdm
from scipy.optimize import least_squares

from SuperGluePretrainedNetwork.models.matching import Matching
from SuperGluePretrainedNetwork.models.utils import frame2tensor

class FeatureMatcher:
    def __init__(self, device='cuda'):
        self.device = device

        config = {
            'superpoint': {
                'nms_radius': 3,
                'keypoint_threshold': 0.001,
                'max_keypoints': -1
            },
            'superglue': {
                'weights': "outdoor",
                'sinkhorn_iterations':70,
                'match_threshold': 0.1,
            }
        }
        self.model = Matching(config).eval().to(device)

    def match_features(self, img1, kpts1, scores1, desc1, img2, kpts2, scores2, desc2):
        kpts1 = torch.from_numpy(kpts1).float().unsqueeze(0).to(self.device)
        kpts2 = torch.from_numpy(kpts2).float().unsqueeze(0).to(self.device)
        scores1 = torch.from_numpy(scores1).float().unsqueeze(0).to(self.device)
        scores2 = torch.from_numpy(scores2).float().unsqueeze(0).to(self.device)
        desc1 = torch.from_numpy(desc1).float().unsqueeze(0).to(self.device)
        desc2 = torch.from_numpy(desc2).float().unsqueeze(0).to(self.device)

        data = {'image0' : img1, 'image1' : img2, 'keypoints0': kpts1, 'scores0' : scores1, 'scores1' : scores2,
                'keypoints1': kpts2, 'descriptors0': desc1, 'descriptors1': desc2}
        
        with torch.no_grad():
            pred = self.model(data)
        matches = pred['matches0'][0].cpu().numpy()
        match_scores = pred['matching_scores0'][0].cpu().numpy() 
        valid = matches > -1
        return torch.arange(len(kpts1))[valid], torch.arange(len(kpts2))[matches[valid]] ,match_scores[valid]

class SfMPipeline:
    def __init__(self, db_path: str, image_scaling : int = 1, device:str = "cuda"):
        self.db = SfMDatabase(db_path)
        self.device = device
        self.matcher   = FeatureMatcher(device)
        self.image_scaling = image_scaling

        self.frames = {}

        self.reconstructed_points = {}  # point3D_id -> np.array([X, Y, Z])
        self.camera_poses = {}          # photo_id -> (R, t)
        self.point_observations = {}    # point3D_id -> List[(photo_id, feature_id)]

    def insert_image(self, image_path: str, K: np.ndarray, mask_path: str = None) -> int:
        """Process a single image and store it in the database"""
        
        img = self.get_image_tensor(image_path, mask_path)
        height, width = img.shape[:2]

        cursor = self.db.conn.cursor()
        # Insert photo with calibration matrix K
        cursor.execute('''
        INSERT INTO photos (path, mask_path, width, height, fx, fy, cx, cy)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            str(image_path), 
            str(mask_path),
            width, 
            height,
            float(K[0, 0]*self.image_scaling),  # fx
            float(K[1, 1]*self.image_scaling),  # fy
            float(K[0, 2]*self.image_scaling),  # cx
            float(K[1, 2]*self.image_scaling),  # cy
        ))
        photo_id = cursor.lastrowid
        self.db.conn.commit()

        with torch.no_grad():
            res = self.matcher.model.superpoint({'image': img})
            self.db.insert_features(photo_id, res["keypoints"][0].cpu().detach().numpy(), res["scores"][0].cpu().detach().numpy(), res["descriptors"][0].cpu().detach().numpy())

        return photo_id
    
    def get_image_tensor(self, image_path : str, mask_path : str):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if mask_path:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.bitwise_and(img, img, mask=mask)
        
        height, width = img.shape[:2]
        height, width = int(height*self.image_scaling), int(width*self.image_scaling)
        img = cv2.resize(img, (int(width), int(height))) 
        frame_tensor = frame2tensor(img, device=self.device)
        return frame_tensor

    def match_image_pairs(self):
        """Match features between all image pairs"""
        cursor = self.db.conn.cursor()
        photos = cursor.execute('SELECT photo_id, path, mask_path FROM photos').fetchall()
        keys = ['keypoints', 'scores', 'descriptors']

        for i in tqdm(range(len(photos))):
            photo_1_id, path_1, mask_path_1 = photos[i]

            img = self.get_image_tensor(path_1, mask_path_1)
            # ksd1 = self.db.get_features_by_photo_id(photo_1_id)
            ksd1 = self.matcher.model.superpoint({'image': img})
            last_data = {k+'0': ksd1[k] for k in keys}
            last_data["image0"] = img
            
            for j in range(i+1, len(photos)):
                photo_2_id, path_2, mask_path_2 = photos[j]

                frame_tensor = self.get_image_tensor(path_2, mask_path_2)
                
                with torch.no_grad():
                    pred = self.matcher.model({**last_data, 'image1': frame_tensor})
                
                kpts0 = last_data['keypoints0'][0].detach().cpu().numpy()
                kpts1 = pred['keypoints1'][0].detach().cpu().numpy()
                matches = pred['matches0'][0].detach().cpu().numpy()
                confidence = pred['matching_scores0'][0].detach().cpu().numpy()
                
                valid = matches > -1
                confident = confidence>0.1
                # possible = valid & confident
                possible = valid

                # mkpts0 = kpts0[possible]
                # mkpts1 = kpts1[matches[possible]]

                # Feature IDs
                fid1 = np.arange(len(kpts0))[possible]
                fid2 = np.arange(len(kpts1))[matches[possible]]

                # Store matches
                self.db.insert_matches(photo_1_id, photo_2_id, fid1, fid2)
                

    def _get_matches_between_images(self, photo_id_1: int, photo_id_2: int) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        """Get matching points between two images with their feature IDs"""
        cursor = self.db.conn.cursor()

        photo_id_1, photo_id_2 = min(photo_id_1, photo_id_2), max(photo_id_1, photo_id_2)

        matches = cursor.execute('''
            SELECT f1.keypoint, f2.keypoint, m.feature_id_1, m.feature_id_2
            FROM matches m
            JOIN features f1 ON m.feature_id_1 = f1.feature_id
            JOIN features f2 ON m.feature_id_2 = f2.feature_id
            WHERE m.photo_id_1 = ? AND m.photo_id_2 = ?
        ''', (photo_id_1, photo_id_2)).fetchall()

        pts1 = np.array([np.frombuffer(desc1, dtype=np.float32) for desc1, _, _, _, in matches])
        pts2 = np.array([np.frombuffer(desc2, dtype=np.float32) for _, desc2, _, _ in matches])
        feature_pairs = [(f1, f2) for _, _, f1, f2 in matches]

        return pts1, pts2, feature_pairs

    def _initialize_reconstruction(self):
        """Initialize reconstruction from best pair of images"""
        photo_id_1, photo_id_2 = self._find_best_image_pair()
        print("Best match:", photo_id_1, photo_id_2 )
        
        # Get matching points and feature IDs
        pts1, pts2, feature_pairs = self._get_matches_between_images(photo_id_1, photo_id_2)
        
        # Get calibration matrices
        K1 = self.get_calibration_matrix(photo_id_1)
        K2 = self.get_calibration_matrix(photo_id_2)

        # Normalize points
        pts1_norm = cv2.undistortPoints(pts1, K1, None)
        pts2_norm = cv2.undistortPoints(pts2, K2, None)

        # Estimate essential matrix and recover pose
        E, mask = cv2.findEssentialMat(
            pts1_norm, pts2_norm, np.eye(3),
            method=cv2.RANSAC,
            prob=0.99,
            threshold=0.005
        )

        _, R, t, mask = cv2.recoverPose(E, pts1_norm, pts2_norm, np.eye(3), mask=mask)

        # Set first camera as identity (world origin)
        self.camera_poses[photo_id_1] = (np.eye(3), np.zeros((3, 1)))
        self.camera_poses[photo_id_2] = (R, t)

        # Triangulate points
        P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K2 @ np.hstack([R, t])
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = (points_4d[:3] / points_4d[3]).T

        # Store only points with acceptable reprojection error
        for i, (point_3d, (f1, f2)) in enumerate(zip(points_3d, feature_pairs)):
            if not mask[i]:
                continue

            # Check reprojection error
            proj1 = P1 @ np.hstack([point_3d, 1])
            proj2 = P2 @ np.hstack([point_3d, 1])
            
            proj1 = proj1[:2] / proj1[2]
            proj2 = proj2[:2] / proj2[2]

            error1 = np.linalg.norm(proj1 - pts1[i])
            error2 = np.linalg.norm(proj2 - pts2[i])

            if error1 < 7.0 and error2 < 7.0:  # 5 pixel threshold
                point3D_id = len(self.reconstructed_points)
                self.reconstructed_points[point3D_id] = point_3d
                self.point_observations[point3D_id] = [(photo_id_1, f1), (photo_id_2, f2)]

    def _find_best_image_pair(self) -> Tuple[int, int]:
        """Find the best pair of images for initialization"""
        cursor = self.db.conn.cursor()
        
        # Get all image pairs sorted by number of matches
        pairs = cursor.execute('''
            SELECT photo_id_1, photo_id_2, COUNT(*) as match_count
            FROM matches
            GROUP BY photo_id_1, photo_id_2
            HAVING match_count > 50
            ORDER BY match_count DESC
        ''').fetchall()

        best_pair = None
        max_inliers = 0

        for photo_id_1, photo_id_2,_ in pairs:
            pts1, pts2, _ = self._get_matches_between_images(photo_id_1, photo_id_2)

            # Get calibration matrices
            K1 = self.get_calibration_matrix(photo_id_1)
            K2 = self.get_calibration_matrix(photo_id_2)
            
            # Normalize points
            pts1 = cv2.undistortPoints(pts1, K1, None)
            pts2 = cv2.undistortPoints(pts2, K2, None)

            # Compute essential matrix
            E, mask = cv2.findEssentialMat(
                pts1, pts2, np.eye(3), 
                method=cv2.RANSAC, 
                prob=0.99,
                threshold=0.005
            )

            num_inliers = np.sum(mask)
            inlier_ratio = num_inliers / len(pts1)
            if inlier_ratio > 0.2 and num_inliers > max_inliers:
                best_pair = (photo_id_1, photo_id_2)
                max_inliers = num_inliers

        if best_pair is None:
            raise Exception("No suitable image pair found")

        return best_pair


    def _find_2d_3d_correspondences(self, photo_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Find 2D-3D correspondences for a new image"""
        points_3d = []
        points_2d = []
        cursor = self.db.conn.cursor()
              
        # For each 3D point, check if it has matches with the new image
        for point3D_id, observations in self.point_observations.items():
            for obs_photo_id, obs_feature_id in observations:
                # Look for matches between the observed photo and the new photo
                # matches = self._get_matches_between_images(photo_id, obs_photo_id)
                
                if photo_id > obs_photo_id:
                    query = f'''
                        SELECT m.feature_id_2
                        FROM matches m
                        WHERE m.photo_id_1 = {obs_photo_id} AND m.photo_id_2 = {photo_id} AND m.feature_id_1 = {obs_feature_id}
                    '''
                else:
                    query = f'''
                        SELECT m.feature_id_1
                        FROM matches m
                        WHERE m.photo_id_1 = {photo_id} AND m.photo_id_2 = {obs_photo_id} AND m.feature_id_2 = {obs_feature_id}
                    '''
                match_id = cursor.execute(query).fetchone()

                if match_id:
                    fid = match_id[0]
                    query = f'''
                        SELECT keypoint
                        FROM features 
                        WHERE feature_id = {fid}
                    '''
                    feature = cursor.execute(query).fetchone()
                    if feature:
                        feature = np.frombuffer(feature[0], dtype=np.float32)
                        points_3d.append(self.reconstructed_points[point3D_id])
                        points_2d.append([feature[0], feature[1]])
                        break  # Use only one observation per 3D point

        return np.array(points_3d), np.array(points_2d)

    def _register_next_image(self, photo_id: int) -> bool:
        """Register a new image to the reconstruction"""
        if photo_id in self.camera_poses:
            return False

        # Find 2D-3D correspondences
        points_3d, points_2d = self._find_2d_3d_correspondences(photo_id)

        if len(points_3d) < 10:
            return False

        # Get calibration matrix
        K = self.get_calibration_matrix(photo_id)

        # Solve PnP
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d, points_2d, K, None,
            iterationsCount=500,
            reprojectionError=12.0,
            confidence=0.7,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success or len(inliers) < 5:
            return False

        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)
        self.camera_poses[photo_id] = (R, tvec)

        # Triangulate new points
        self._triangulate_new_points(photo_id)
        return True

    def _triangulate_new_points(self, new_photo_id: int):
        """Triangulate new points visible in the newly added image"""
        K_new = self.get_calibration_matrix(new_photo_id)
        R_new, t_new = self.camera_poses[new_photo_id]
        P_new = K_new @ np.hstack([R_new, t_new])

        # For each existing camera, triangulate new points with the new camera
        for other_photo_id in self.camera_poses:
            if other_photo_id == new_photo_id:
                continue

            K_other = self.get_calibration_matrix(other_photo_id)
            R_other, t_other = self.camera_poses[other_photo_id]
            P_other = K_other @ np.hstack([R_other, t_other])

            pts1, pts2, feature_pairs = self._get_matches_between_images(other_photo_id, new_photo_id)

            for i, ((x1, y1), (x2, y2), (f1, f2)) in enumerate(zip(pts1, pts2, feature_pairs)):
                # Skip if the point is already reconstructed
                if any(f1 in [f for _, f in obs] for obs in self.point_observations.values()):
                    continue

                # Triangulate
                point_4d = cv2.triangulatePoints(P_other, P_new,
                                               np.array([[x1], [y1]]),
                                               np.array([[x2], [y2]]))
                point_3d = (point_4d[:3] / point_4d[3]).T[0]

                # Check reprojection error
                proj1 = P_other @ np.hstack([point_3d, 1])
                proj2 = P_new @ np.hstack([point_3d, 1])
                
                proj1 = proj1[:2] / proj1[2]
                proj2 = proj2[:2] / proj2[2]

                error1 = np.linalg.norm(proj1 - np.array([x1, y1]))
                error2 = np.linalg.norm(proj2 - np.array([x2, y2]))

                if error1 < 9.0 and error2 < 9.0:  # 5 pixel threshold
                    point3D_id = len(self.reconstructed_points)
                    self.reconstructed_points[point3D_id] = point_3d
                    self.point_observations[point3D_id] = [(other_photo_id, f1), (new_photo_id, f2)]


    def get_calibration_matrix(self, photo_id: int) -> np.ndarray:
        """Retrieve the calibration matrix K for a given photo"""
        cursor = self.db.conn.cursor()
        result = cursor.execute('''
        SELECT fx, fy, cx, cy
        FROM photos
        WHERE photo_id = ?
        ''', (photo_id,)).fetchone()
        
        if result:
            fx, fy, cx, cy = result
            K = np.array([
                [fx, 0, cx],
                [0,  fy,   cy],
                [0,   0,    1]
            ])
            return K
        else:
            raise ValueError(f"Photo with id: {photo_id} is not in the database.")
        
    def bundle_adjustment(self):
        """Perform bundle adjustment to refine the reconstruction"""
        # Prepare data for optimization
        camera_ids = list(self.camera_poses.keys())
        point3D_ids = list(self.reconstructed_points.keys())
        
        # Parameters: [camera_params (R, t), point_coords]
        x0 = []
        
        # Add camera parameters
        for photo_id in camera_ids:
            R, t = self.camera_poses[photo_id]
            rvec, _ = cv2.Rodrigues(R)
            x0.extend(rvec.flatten())
            x0.extend(t.flatten())
        
        # Add 3D points
        for point3D_id in point3D_ids:
            x0.extend(self.reconstructed_points[point3D_id])
        
        # Create parameter indices maps
        camera_param_indices = {photo_id: i * 6 for i, photo_id in enumerate(camera_ids)}
        point_indices = {point3D_id: len(camera_ids) * 6 + i * 3 
                        for i, point3D_id in enumerate(point3D_ids)}
        
        def objective(x):
            """Bundle adjustment objective function"""
            residuals = []
            
            for point3D_id, observations in self.point_observations.items():
                point_idx = point_indices[point3D_id]
                point_3d = x[point_idx:point_idx + 3]
                
                for photo_id, feature_id in observations:
                    camera_idx = camera_param_indices[photo_id]
                    rvec = x[camera_idx:camera_idx + 3]
                    tvec = x[camera_idx + 3:camera_idx + 6]
                    
                    # Get observed 2D point
                    cursor = self.db.conn.cursor()
                    result = cursor.execute('''
                        SELECT keypoint_x, keypoint_y
                        FROM features
                        WHERE feature_id = ?
                    ''', (feature_id,)).fetchone()
                    
                    if result:
                        x_obs, y_obs = result
                        
                        # Project 3D point
                        K = self.get_calibration_matrix(photo_id)
                        R, _ = cv2.Rodrigues(rvec)
                        point_proj = K @ (R @ point_3d.reshape(3, 1) + tvec.reshape(3, 1))
                        x_proj = point_proj[0] / point_proj[2]
                        y_proj = point_proj[1] / point_proj[2]
                        
                        # Add residuals
                        residuals.extend([x_proj - x_obs, y_proj - y_obs])
            
            return np.array(residuals)
        
        # Optimize
        result = least_squares(objective, x0, method='lm', max_nfev=100)
        
        # Update reconstruction with optimized parameters
        x = result.x
        
        for photo_id in camera_ids:
            idx = camera_param_indices[photo_id]
            rvec = x[idx:idx + 3]
            tvec = x[idx + 3:idx + 6]
            R, _ = cv2.Rodrigues(rvec)
            self.camera_poses[photo_id] = (R, tvec.reshape(3, 1))
        
        for point3D_id in point3D_ids:
            idx = point_indices[point3D_id]
            self.reconstructed_points[point3D_id] = x[idx:idx + 3]
    
    def run_incremental_sfm(self):
        """Run the complete incremental SfM pipeline"""
        # Initialize from best pair
        self._initialize_reconstruction()
        print("Registered:",len(self.reconstructed_points))

        # Get all photo IDs
        cursor = self.db.conn.cursor()
        all_photos = cursor.execute('SELECT photo_id FROM photos').fetchall()
        remaining_photos = set(p[0] for p in all_photos) - set(self.camera_poses.keys())
        
        # Incrementally add new images
        while remaining_photos:
            # Find next best image to add
            best_photo_id = None
            max_common_points = 0
            
            for search_photo_id in remaining_photos:
                for registered_photo_id in self.camera_poses.keys():
                    cursor = self.db.conn.execute(
                        '''
                        SELECT COUNT(*) FROM matches
                        WHERE photo_id_1 = ? AND photo_id_2 = ?
                        ''', 
                        (search_photo_id, registered_photo_id))
                    
                    common_points = cursor.fetchone()[0]
                    if common_points > max_common_points:
                        max_common_points = common_points
                        best_photo_id = search_photo_id
            
            print("Registering: ", best_photo_id)

            if best_photo_id is None:
                break
            
            flag = self._register_next_image(best_photo_id)
            print("Registered:",len(self.reconstructed_points))

            remaining_photos.remove(best_photo_id)

