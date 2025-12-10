#!/usr/bin/env python3

import cv2
import numpy as np
import os
import fnmatch
from scipy.spatial import ConvexHull, distance
from scipy.cluster.vq import kmeans2
from image_annotation import MarkerDetector


class Calibration:
    """Class for camera calibration using ArUco markers."""
    
    markers_required = 0
    markers_total = 0
    image_folders = {}
    
    left_calibration = None
    right_calibration = None
    left_fisheye_calibration = None
    right_fisheye_calibration = None
    stereo_calibration = None
    
    cameras = {"left":1, "right":0}
    
    rectification_alpha = 0.95
    
    
    def __init__(self, marker_dimensions=(8, 5), square_length=0.05, marker_length=0.037):
        self.marker_detector = MarkerDetector(marker_dimensions=marker_dimensions)
        self.marker_detector.board = self.marker_detector.create_board(square_length, marker_length)
        self.markers_total = self.marker_detector.markers_total
        self.markers_required = self.marker_detector.markers_required

        self.image_folders = {"left":"images/left", "right":"images/right", "stereo":"images/stereo"}
        self.images = {"left":[], "right":[], "stereo":[]}
        # self.create_image_folders()
    
    @classmethod
    def from_config(cls, config):
        """
        Create a Calibration instance from a config object.
        
        Args:
            config: ConfigParser object with configuration
            
        Returns:
            Configured Calibration instance
        """
        marker_dims = (
            config.getint('aruco', 'marker_dimensions_width'),
            config.getint('aruco', 'marker_dimensions_height')
        )
        
        instance = cls(
            marker_dimensions=marker_dims,
            square_length=config.getfloat('aruco', 'square_length'),
            marker_length=config.getfloat('aruco', 'marker_length')
        )
        
        # Set camera indices
        instance.cameras['left'] = config.getint('cameras', 'left')
        instance.cameras['right'] = config.getint('cameras', 'right')
        
        # Set image folders
        instance.image_folders['left'] = config.get('paths', 'left_images')
        instance.image_folders['right'] = config.get('paths', 'right_images')
        instance.image_folders['stereo'] = config.get('paths', 'stereo_images')
        instance.image_folders['calibration'] = config.get('paths', 'calibration_output')
        
        # Set rectification alpha
        instance.rectification_alpha = config.getfloat('calibration', 'rectification_alpha')
        instance.rectification_alpha_left = config.getfloat('calibration', 'rectification_alpha_left')
        instance.rectification_alpha_right = config.getfloat('calibration', 'rectification_alpha_right')
        
        # Set individual calibration flags
        instance.calib_use_intrinsic_guess = config.getboolean('calibration', 'calib_use_intrinsic_guess')
        instance.calib_fix_principal_point = config.getboolean('calibration', 'calib_fix_principal_point')
        instance.calib_fix_aspect_ratio = config.getboolean('calibration', 'calib_fix_aspect_ratio')
        instance.calib_zero_tangent_dist = config.getboolean('calibration', 'calib_zero_tangent_dist')
        instance.calib_rational_model = config.getboolean('calibration', 'calib_rational_model')
        instance.calib_thin_prism_model = config.getboolean('calibration', 'calib_thin_prism_model')
        instance.calib_tilted_model = config.getboolean('calibration', 'calib_tilted_model')
        instance.calib_fix_focal_length = config.getboolean('calibration', 'calib_fix_focal_length')
        instance.calib_fix_k1 = config.getboolean('calibration', 'calib_fix_k1')
        instance.calib_fix_k2 = config.getboolean('calibration', 'calib_fix_k2')
        instance.calib_fix_k3 = config.getboolean('calibration', 'calib_fix_k3')
        instance.calib_fix_k4 = config.getboolean('calibration', 'calib_fix_k4')
        instance.calib_fix_k5 = config.getboolean('calibration', 'calib_fix_k5')
        instance.calib_fix_k6 = config.getboolean('calibration', 'calib_fix_k6')
        
        # Set stereo calibration flags
        instance.stereo_fix_intrinsic = config.getboolean('calibration', 'stereo_fix_intrinsic')
        instance.stereo_fix_focal_length = config.getboolean('calibration', 'stereo_fix_focal_length')
        instance.stereo_fix_principal_point = config.getboolean('calibration', 'stereo_fix_principal_point')
        instance.stereo_fix_aspect_ratio = config.getboolean('calibration', 'stereo_fix_aspect_ratio')
        instance.stereo_use_intrinsic_guess = config.getboolean('calibration', 'stereo_use_intrinsic_guess')
        instance.stereo_same_focal_length = config.getboolean('calibration', 'stereo_same_focal_length')
        instance.stereo_zero_tangent_dist = config.getboolean('calibration', 'stereo_zero_tangent_dist')
        instance.stereo_rational_model = config.getboolean('calibration', 'stereo_rational_model')
        instance.stereo_thin_prism_model = config.getboolean('calibration', 'stereo_thin_prism_model')
        instance.stereo_tilted_model = config.getboolean('calibration', 'stereo_tilted_model')
        instance.stereo_fix_k1 = config.getboolean('calibration', 'stereo_fix_k1')
        instance.stereo_fix_k2 = config.getboolean('calibration', 'stereo_fix_k2')
        instance.stereo_fix_k3 = config.getboolean('calibration', 'stereo_fix_k3')
        instance.stereo_fix_k4 = config.getboolean('calibration', 'stereo_fix_k4')
        instance.stereo_fix_k5 = config.getboolean('calibration', 'stereo_fix_k5')
        instance.stereo_fix_k6 = config.getboolean('calibration', 'stereo_fix_k6')
        
        # Set debug output flag
        instance.generate_debug_images = config.getboolean('calibration', 'generate_debug_images')
        
        # Set marker color
        instance.marker_detector.marker_colour = (
            config.getint('aruco', 'marker_colour_b'),
            config.getint('aruco', 'marker_colour_g'),
            config.getint('aruco', 'marker_colour_r')
        )
        
        return instance
        
        
    def save_calibration(self, name, data, save_format='npz'):
        """
        Save calibration data in specified format(s).
        
        Args:
            name: Base filename without extension
            data: Dictionary containing calibration data
            save_format: Format to save - 'npz', 'yaml', or 'both'
        """
        # Ensure calibration output folder exists
        calib_folder = self.image_folders.get('calibration', 'calibration')
        os.makedirs(calib_folder, exist_ok=True)
        
        # Create full path
        filepath = os.path.join(calib_folder, name)
        
        if save_format in ('npz', 'both'):
            np.savez(f"{filepath}.npz", **data)
            print(f"Saved {filepath}.npz")
        
        if save_format in ('yaml', 'both'):
            # Use OpenCV's FileStorage for YAML output
            fs = cv2.FileStorage(f"{filepath}.yaml", cv2.FILE_STORAGE_WRITE)
            
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    fs.write(key, value)
                elif isinstance(value, list):
                    # Handle list of arrays (rvecs, tvecs)
                    if len(value) > 0 and isinstance(value[0], np.ndarray):
                        # Write as sequence of matrices
                        fs.write(key, value)
                    else:
                        # Regular list
                        fs.write(key, value)
                elif isinstance(value, (int, float, str)):
                    fs.write(key, value)
            
            fs.release()
            print(f"Saved {filepath}.yaml")        
    def create_image_folders(self):
        """Create image folders for storing calibration images."""
        for folder in self.image_folders.values():
            if not os.path.exists(folder):
                os.makedirs(folder)
                print(f"Created folder: {folder}")
            else:
                for file in os.listdir(folder):
                    file_path = os.path.join(folder, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    print(f"Deleted existing files from {file_path}")
       
       
    def estimate_board_pose(self, charuco_corners, charuco_ids, image_shape):
        """
        Estimate 3D pose (rotation, distance) of the calibration board.
        
        Args:
            charuco_corners: Detected charuco corners
            charuco_ids: Corresponding IDs
            image_shape: Tuple of (height, width)
            
        Returns:
            Dictionary with 'distance', 'tilt_x', 'tilt_y', 'rotation' or None if estimation fails
        """
        if charuco_corners is None or len(charuco_corners) < 4:
            return None
        
        # Get 3D object points for the detected corners
        obj_points = self.marker_detector.board.getChessboardCorners()[charuco_ids.flatten()]
        
        # Use rough camera matrix estimate for initial pose
        h, w = image_shape
        focal_length = w  # Rough estimate
        camera_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float32)
        dist_coeffs = np.zeros(5)  # Assume no distortion for pose estimation
        
        try:
            # Solve PnP to get rotation and translation vectors
            success, rvec, tvec = cv2.solvePnP(
                obj_points,
                charuco_corners,
                camera_matrix,
                dist_coeffs
            )
            
            if not success:
                return None
            
            # Convert rotation vector to angles
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            
            # Extract Euler angles (in degrees)
            # tilt_x: rotation around x-axis (pitch)
            # tilt_y: rotation around y-axis (yaw)
            # rotation: rotation around z-axis (roll)
            sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
            
            if sy > 1e-6:
                tilt_x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                tilt_y = np.arctan2(-rotation_matrix[2, 0], sy)
                rotation = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            else:
                tilt_x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                tilt_y = np.arctan2(-rotation_matrix[2, 0], sy)
                rotation = 0
            
            # Convert to degrees
            tilt_x = np.degrees(tilt_x)
            tilt_y = np.degrees(tilt_y)
            rotation = np.degrees(rotation)
            
            # Calculate distance (Euclidean distance in 3D)
            distance = np.linalg.norm(tvec)
            
            return {
                'distance': distance,
                'tilt_x': tilt_x,
                'tilt_y': tilt_y,
                'rotation': rotation,
                'tvec': tvec,
                'rvec': rvec
            }
        except:
            return None
    
    
    def analyze_image_coverage(self, images_data, image_shape):
        """
        Analyze spatial coverage and pose diversity of calibration images using scipy.
        
        Args:
            images_data: List of dicts with 'file', 'corners', 'ids', 'charuco_corners', 'charuco_ids' keys
            image_shape: Tuple of (height, width)
            
        Returns:
            List of image score dictionaries
        """
        if len(images_data) == 0:
            return []
        
        h, w = image_shape
        cx, cy = w / 2, h / 2
        max_radius = np.sqrt(cx**2 + cy**2)
        
        # Define radial zones (4 zones from center to edge)
        num_radial_zones = 4
        # Define angular sectors (8 sectors around center)
        num_angular_sectors = 8
        
        # Collect all corner points across all images for density analysis
        all_corners_global = []
        
        # Track coverage score for each image
        image_scores = []
        
        for idx, img_data in enumerate(images_data):
            corners = img_data['corners']
            charuco_corners = img_data.get('charuco_corners')
            charuco_ids = img_data.get('charuco_ids')
            
            if corners is None or len(corners) == 0:
                image_scores.append({
                    'idx': idx,
                    'radial_zones': set(),
                    'angular_sectors': set(),
                    'num_corners': 0,
                    'pose': None,
                    'hull_area': 0,
                    'corner_spread': 0
                })
                continue
            
            # Estimate board pose
            pose = self.estimate_board_pose(charuco_corners, charuco_ids, image_shape)
            
            # Analyze spatial coverage using radial and angular sectors
            radial_zones = set()
            angular_sectors = set()
            corner_points = []
            
            for corner in corners:
                if len(corner) > 0:
                    # Corner is already a 2D array, flatten to get [x, y]
                    pt = corner.reshape(-1)
                    for i in range(0, len(pt), 2):
                        if i + 1 < len(pt):
                            x, y = float(pt[i]), float(pt[i + 1])
                            corner_points.append((x, y))
                            all_corners_global.append((x, y))
                            
                            # Calculate radial zone (distance from center)
                            dx, dy = x - cx, y - cy
                            distance_from_center = np.sqrt(dx**2 + dy**2)
                            radial_zone = min(int((distance_from_center / max_radius) * num_radial_zones), num_radial_zones - 1)
                            radial_zones.add(radial_zone)
                            
                            # Calculate angular sector
                            angle = np.arctan2(dy, dx)  # -pi to pi
                            angle_deg = np.degrees(angle) % 360  # 0 to 360
                            angular_sector = int(angle_deg / (360 / num_angular_sectors))
                            angular_sectors.add(angular_sector)
            
            # Calculate convex hull area (indicates spread of corners)
            hull_area = 0
            if len(corner_points) >= 3:
                try:
                    hull = ConvexHull(corner_points)
                    hull_area = hull.volume  # In 2D, volume is area
                except:
                    hull_area = 0
            
            # Calculate corner spread using pairwise distances
            corner_spread = 0
            if len(corner_points) >= 2:
                pts_array = np.array(corner_points)
                # Use maximum pairwise distance as spread metric
                dists = distance.pdist(pts_array)
                corner_spread = np.max(dists) if len(dists) > 0 else 0
            
            image_scores.append({
                'idx': idx,
                'radial_zones': radial_zones,
                'angular_sectors': angular_sectors,
                'num_corners': len(corner_points),
                'pose': pose,
                'hull_area': hull_area,
                'corner_spread': corner_spread
            })
        
        # Analyze global corner density using k-means clustering
        # Identify under-sampled regions
        corner_density_score = {}
        if len(all_corners_global) > 0:
            # Create spatial grid clusters
            num_clusters = min(16, len(all_corners_global))  # 4x4 grid of clusters
            if num_clusters >= 2:
                try:
                    corners_array = np.array(all_corners_global)
                    centroids, labels = kmeans2(corners_array, num_clusters, minit='points')
                    
                    # Count corners per cluster
                    unique, counts = np.unique(labels, return_counts=True)
                    cluster_counts = dict(zip(unique, counts))
                    
                    # Score images based on how many corners they contribute to sparse clusters
                    # (clusters with fewer corners are more valuable)
                    avg_count = np.mean(counts)
                    for idx, img_data in enumerate(images_data):
                        if image_scores[idx]['num_corners'] == 0:
                            corner_density_score[idx] = 0
                            continue
                        
                        # This is a simplified approach - in practice would need to track
                        # which corners belong to which image and which clusters
                        # For now, just boost images with good spatial spread
                        corner_density_score[idx] = image_scores[idx]['hull_area'] / (w * h) * 100
                except:
                    for idx in range(len(images_data)):
                        corner_density_score[idx] = 0
            else:
                for idx in range(len(images_data)):
                    corner_density_score[idx] = 0
        
        # Add density score to image scores
        for idx in range(len(image_scores)):
            image_scores[idx]['density_score'] = corner_density_score.get(idx, 0)
        
        return image_scores
    
    
    def select_diverse_images(self, image_scores, max_images):
        """
        Select images that maximize spatial coverage and pose diversity.
        
        Args:
            image_scores: List of score dicts from analyze_image_coverage
            max_images: Maximum number of images to select
            
        Returns:
            List of selected indices
        """
        if len(image_scores) <= max_images:
            return [score['idx'] for score in image_scores]
        
        selected_indices = []
        covered_radial = set()
        covered_angular = set()
        
        # Track pose diversity bins
        # Distance bins: near, medium, far
        # Tilt bins: 0-15°, 15-30°, 30-45°, 45-60°, 60+°
        # Rotation bins: 0-45°, 45-90°, 90-135°, 135-180°
        covered_poses = {'distance': set(), 'tilt': set(), 'rotation': set()}
        
        remaining = list(image_scores)
        
        # Greedy selection: maximize coverage and pose diversity
        while len(selected_indices) < max_images and remaining:
            best_idx = None
            best_score = -1
            best_item = None
            
            for item in remaining:
                # Spatial coverage score
                new_radial = item['radial_zones'] - covered_radial
                new_angular = item['angular_sectors'] - covered_angular
                spatial_score = len(new_radial) * 20 + len(new_angular) * 10
                
                # Convex hull area bonus (larger spread = better)
                hull_bonus = item['hull_area'] * 0.001
                
                # Corner spread bonus
                spread_bonus = item['corner_spread'] * 0.01
                
                # Density score (fills gaps in coverage)
                density_bonus = item.get('density_score', 0) * 0.5
                
                # Pose diversity score
                pose_score = 0
                distance_bin = None
                tilt_bin = None
                rotation_bin = None
                
                if item['pose'] is not None:
                    pose = item['pose']
                    
                    # Bin distance (3 bins)
                    distance_bin = min(int(pose['distance'] / 0.5), 2)  # 0-0.5m, 0.5-1m, 1m+
                    if distance_bin not in covered_poses['distance']:
                        pose_score += 15
                    
                    # Bin tilt (absolute value, combined x and y)
                    tilt = np.sqrt(pose['tilt_x']**2 + pose['tilt_y']**2)
                    tilt_bin = min(int(abs(tilt) / 15), 4)  # 0-15°, 15-30°, 30-45°, 45-60°, 60+°
                    if tilt_bin not in covered_poses['tilt']:
                        pose_score += 15
                    
                    # Bin rotation (in-plane rotation)
                    rotation_bin = int(abs(pose['rotation']) / 45) % 4  # 0-45°, 45-90°, 90-135°, 135-180°
                    if rotation_bin not in covered_poses['rotation']:
                        pose_score += 10
                
                # Quality bonus
                quality_score = item['num_corners'] * 0.1
                
                # Combined score
                total_score = spatial_score + pose_score + quality_score + hull_bonus + spread_bonus + density_bonus
                
                if total_score > best_score:
                    best_score = total_score
                    best_idx = item['idx']
                    best_item = item
            
            if best_item is None:
                break
            
            # Update coverage tracking
            selected_indices.append(best_idx)
            covered_radial.update(best_item['radial_zones'])
            covered_angular.update(best_item['angular_sectors'])
            
            if best_item['pose'] is not None:
                pose = best_item['pose']
                distance_bin = min(int(pose['distance'] / 0.5), 2)
                tilt = np.sqrt(pose['tilt_x']**2 + pose['tilt_y']**2)
                tilt_bin = min(int(abs(tilt) / 15), 4)
                rotation_bin = int(abs(pose['rotation']) / 45) % 4
                
                covered_poses['distance'].add(distance_bin)
                covered_poses['tilt'].add(tilt_bin)
                covered_poses['rotation'].add(rotation_bin)
            
            remaining.remove(best_item)
        
        # If we need more images, add ones with highest corner counts
        if len(selected_indices) < max_images:
            remaining_sorted = sorted(remaining, key=lambda x: x['num_corners'], reverse=True)
            for item in remaining_sorted[:max_images - len(selected_indices)]:
                selected_indices.append(item['idx'])
        
        return sorted(selected_indices)
    
    
    def calibrate(self, camera_name, fisheye=True, save_format='npz', max_images=30):
        """
        Calibrate a single camera using captured images.
        
        Args:
            camera_name: Name of the camera ("left" or "right")
            fisheye: Whether to use fisheye calibration model
            save_format: Format to save calibration - 'npz', 'json', or 'both'
            max_images: Maximum number of images to use (default: 30, good balance of accuracy/speed)
        """
        folder = self.image_folders[camera_name]
        print(f"Opening folder: {folder}")
        
        # Support multiple image formats
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        images = [os.path.join(folder, f) for f in os.listdir(folder) 
                 if f.lower().endswith(image_extensions)]
        images.sort()  # Ensure files are in order
        
        print(f"Found {len(images)} images in {folder}")
        
        if len(images) == 0:
            print(f"Error: No images found in {folder}")
            print(f"Please capture images first using: python main.py -a capture-{camera_name}")
            return
        
        # First pass: analyze all images for coverage
        print("Analyzing image coverage and diversity...")
        images_data = []
        image_shape = None
        
        for image_file in images:
            image = cv2.imread(image_file)
            if image is None:
                continue
                
            if image_shape is None:
                image_shape = image.shape[:2]
                
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            marker_corners, marker_ids, _ = self.marker_detector.detector.detectMarkers(image_gray)
            
            # Interpolate charuco corners for pose estimation
            charuco_corners = None
            charuco_ids = None
            if marker_corners is not None and len(marker_corners) >= self.markers_required:
                retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    marker_corners, marker_ids, image_gray, self.marker_detector.board)
            
            # Store image data for analysis
            images_data.append({
                'file': image_file,
                'corners': marker_corners,
                'ids': marker_ids,
                'charuco_corners': charuco_corners,
                'charuco_ids': charuco_ids
            })
        
        if image_shape is None:
            print(f"Error: Could not read any images")
            return
        
        # Analyze coverage and select diverse subset
        image_scores = self.analyze_image_coverage(images_data, image_shape)
        
        if len(images) > max_images:
            selected_indices = self.select_diverse_images(image_scores, max_images)
            images_data_selected = [images_data[i] for i in selected_indices]
            
            # Keep track of unused images for backfill
            unused_indices = [i for i in range(len(images_data)) if i not in selected_indices]
            
            print(f"Selected {len(images_data_selected)} diverse images from {len(images)} total")
            
            # Show coverage statistics
            all_radial = set()
            all_angular = set()
            pose_distances = []
            pose_tilts = []
            pose_rotations = []
            
            for idx in selected_indices:
                all_radial.update(image_scores[idx]['radial_zones'])
                all_angular.update(image_scores[idx]['angular_sectors'])
                if image_scores[idx]['pose'] is not None:
                    pose = image_scores[idx]['pose']
                    pose_distances.append(pose['distance'])
                    tilt = np.sqrt(pose['tilt_x']**2 + pose['tilt_y']**2)
                    pose_tilts.append(tilt)
                    pose_rotations.append(abs(pose['rotation']))
            
            print(f"Spatial coverage: {len(all_radial)}/4 radial zones, {len(all_angular)}/8 angular sectors")
            if pose_distances:
                print(f"Pose diversity: distance {min(pose_distances):.2f}-{max(pose_distances):.2f}m, "
                      f"tilt {min(pose_tilts):.1f}-{max(pose_tilts):.1f}°, "
                      f"rotation {min(pose_rotations):.1f}-{max(pose_rotations):.1f}°")
        else:
            images_data_selected = images_data
            unused_indices = []
            print(f"Using all {len(images_data_selected)} images")
        
        # Second pass: extract calibration data from selected images
        all_charuco_corners = []
        all_charuco_ids = []
        failed_indices = []
        total_images = len(images_data_selected)
        
        print("Extracting calibration data...")
        for idx, img_data in enumerate(images_data_selected):
            display_idx = idx + 1
            if display_idx % 10 == 0 or display_idx == total_images:
                print(f"Progress: {display_idx}/{total_images} images processed...")
            
            image = cv2.imread(img_data['file'])
            if image is None:
                failed_indices.append(idx)
                continue
            
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            marker_corners = img_data['corners']
            marker_ids = img_data['ids']
                
            if marker_corners is not None and len(marker_corners) >= self.markers_required:
                if(len(marker_corners) != len(marker_ids)):
                    print(f"Error: Marker corners and IDs are not the same length")
                    failed_indices.append(idx)
                    continue
                retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    marker_corners, marker_ids, image_gray, self.marker_detector.board)
                if retval > 0:
                    all_charuco_corners.append(charuco_corners)
                    all_charuco_ids.append(charuco_ids)
                else:
                    failed_indices.append(idx)
            else:
                failed_indices.append(idx)
        
        # Backfill with unused images if we lost any during validation
        if failed_indices and unused_indices:
            print(f"\n{len(failed_indices)} images failed validation, attempting to backfill from unused images...")
            backfilled = 0
            
            for unused_idx in unused_indices:
                if backfilled >= len(failed_indices):
                    break
                
                img_data = images_data[unused_idx]
                image = cv2.imread(img_data['file'])
                if image is None:
                    continue
                
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                marker_corners = img_data['corners']
                marker_ids = img_data['ids']
                
                if marker_corners is not None and len(marker_corners) >= self.markers_required:
                    if len(marker_corners) != len(marker_ids):
                        continue
                    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                        marker_corners, marker_ids, image_gray, self.marker_detector.board)
                    if retval > 0:
                        all_charuco_corners.append(charuco_corners)
                        all_charuco_ids.append(charuco_ids)
                        backfilled += 1
            
            if backfilled > 0:
                print(f"Successfully backfilled {backfilled} replacement images")
        
        final_ignored = len(failed_indices) - (backfilled if failed_indices and unused_indices else 0)
        if final_ignored > 0:
            print(f"\nIgnored {final_ignored} of {len(images_data_selected)} selected images (could not backfill)")
        print(f"Calibrating using {len(all_charuco_corners)} images with valid corners")
        
        if len(all_charuco_corners) == 0:
            print(f"Error: No valid calibration images found!")
            print(f"Make sure your images contain the ArUco/Charuco board with at least {int(self.markers_required)} markers visible.")
            return
        
        if image_shape is None:
            print(f"Error: Could not determine image dimensions")
            return
                        
        
        if(fisheye == True):
            K = np.zeros((3, 3))
            D = np.zeros((4, 1))
            object_points = []
            image_points = []
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            for corners, ids in zip(all_charuco_corners, all_charuco_ids):
                if(len(corners) > 0):
                    object_points.append(self.marker_detector.board.getChessboardCorners()[ids])
                    image_points.append(corners)

            image_count = len(all_charuco_corners)
            rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(image_count)]
            tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(image_count)]
            
            retval, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                object_points,
                image_points,
                image_shape,
                K,  # Use initialized zeros, not prior calibration
                D,  # Use initialized zeros, not prior calibration
                rvecs,
                tvecs,
                cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW,
                criteria)
            
            print(f"projection error: {retval}")
            print('K: ', K)
            print('Distortion coefficients: ', D)
            
            data = {
                'camera_label': camera_name,
                'image_width': image_shape[1],
                'image_height': image_shape[0],
                'model': 'fisheye',
                'reprojection_error': retval,
                'K': K,
                'D': D,
                'rvecs': rvecs,
                'tvecs': tvecs
            }
            self.save_calibration(f"{camera_name}_fisheye", data, save_format)           
        else:
            # Use robust calibrateCamera with explicit object point mapping
            # This matches the approach from the other script for better accuracy
            all_obj_corners = self.marker_detector.board.getChessboardCorners()
            
            obj_points = []
            img_points = []
            
            for corners, ids in zip(all_charuco_corners, all_charuco_ids):
                # Map detected corner IDs to their 3D positions
                ids_flat = ids.flatten()
                obj_pts = all_obj_corners[ids_flat]
                obj_points.append(obj_pts)
                img_points.append(corners)
            
            # Build calibration flags
            calib_flags = 0
            flag_descriptions = []
            
            if hasattr(self, 'calib_use_intrinsic_guess') and self.calib_use_intrinsic_guess:
                calib_flags |= cv2.CALIB_USE_INTRINSIC_GUESS
                flag_descriptions.append("CALIB_USE_INTRINSIC_GUESS")
            if hasattr(self, 'calib_fix_principal_point') and self.calib_fix_principal_point:
                calib_flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
                flag_descriptions.append("CALIB_FIX_PRINCIPAL_POINT")
            if hasattr(self, 'calib_fix_aspect_ratio') and self.calib_fix_aspect_ratio:
                calib_flags |= cv2.CALIB_FIX_ASPECT_RATIO
                flag_descriptions.append("CALIB_FIX_ASPECT_RATIO")
            if hasattr(self, 'calib_zero_tangent_dist') and self.calib_zero_tangent_dist:
                calib_flags |= cv2.CALIB_ZERO_TANGENT_DIST
                flag_descriptions.append("CALIB_ZERO_TANGENT_DIST")
            if hasattr(self, 'calib_rational_model') and self.calib_rational_model:
                calib_flags |= cv2.CALIB_RATIONAL_MODEL
                flag_descriptions.append("CALIB_RATIONAL_MODEL")
            if hasattr(self, 'calib_thin_prism_model') and self.calib_thin_prism_model:
                calib_flags |= cv2.CALIB_THIN_PRISM_MODEL
                flag_descriptions.append("CALIB_THIN_PRISM_MODEL")
            if hasattr(self, 'calib_tilted_model') and self.calib_tilted_model:
                calib_flags |= cv2.CALIB_TILTED_MODEL
                flag_descriptions.append("CALIB_TILTED_MODEL")
            if hasattr(self, 'calib_fix_focal_length') and self.calib_fix_focal_length:
                calib_flags |= cv2.CALIB_FIX_FOCAL_LENGTH
                flag_descriptions.append("CALIB_FIX_FOCAL_LENGTH")
            if hasattr(self, 'calib_fix_k1') and self.calib_fix_k1:
                calib_flags |= cv2.CALIB_FIX_K1
                flag_descriptions.append("CALIB_FIX_K1")
            if hasattr(self, 'calib_fix_k2') and self.calib_fix_k2:
                calib_flags |= cv2.CALIB_FIX_K2
                flag_descriptions.append("CALIB_FIX_K2")
            if hasattr(self, 'calib_fix_k3') and self.calib_fix_k3:
                calib_flags |= cv2.CALIB_FIX_K3
                flag_descriptions.append("CALIB_FIX_K3")
            if hasattr(self, 'calib_fix_k4') and self.calib_fix_k4:
                calib_flags |= cv2.CALIB_FIX_K4
                flag_descriptions.append("CALIB_FIX_K4")
            if hasattr(self, 'calib_fix_k5') and self.calib_fix_k5:
                calib_flags |= cv2.CALIB_FIX_K5
                flag_descriptions.append("CALIB_FIX_K5")
            if hasattr(self, 'calib_fix_k6') and self.calib_fix_k6:
                calib_flags |= cv2.CALIB_FIX_K6
                flag_descriptions.append("CALIB_FIX_K6")
            
            if calib_flags == 0:
                print("Using default calibration flags")
            else:
                print(f"Using calibration flags: {', '.join(flag_descriptions)}")
            
            # Use cv2.calibrateCamera for more robust optimization
            retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                obj_points,
                img_points,
                image_shape[::-1],  # (width, height) format
                None,
                None,
                flags=calib_flags
            )
            
            print(f"camera matrix: {camera_matrix}")
            print(f"dist coeffs: {dist_coeffs}")
            
            data = {
                'camera_label': camera_name,
                'image_width': image_shape[1],
                'image_height': image_shape[0],
                'model': 'pinhole',
                'reprojection_error': retval,
                'camera_matrix': camera_matrix,
                'dist_coeffs': dist_coeffs,
                'rvecs': rvecs,
                'tvecs': tvecs
            }
            self.save_calibration(camera_name, data, save_format)
            

        print(f"Calibration complete for {camera_name}, error: {retval}")
        
  
    def stereo_calibrate(self, save_format='npz', use_config_flags=True):
        """
        Perform stereo calibration using matched image pairs.
        
        Args:
            save_format: Format to save calibration - 'npz', 'yaml', or 'both'
            use_config_flags: Whether to use stereo calibration flags from config (default: True)
        """
        # Load individual camera calibrations first
        print("Loading individual camera calibrations...")
        if self.left_calibration is None or self.right_calibration is None:
            self.load_calibrations()
        
        if self.left_calibration is None:
            print("Error: Left camera calibration not found. Please run 'calibrate-left' first.")
            return
        
        if self.right_calibration is None:
            print("Error: Right camera calibration not found. Please run 'calibrate-right' first.")
            return
        
        print("Left camera calibration loaded successfully")
        print(f"  Keys: {list(self.left_calibration.keys())}")
        print("Right camera calibration loaded successfully")
        print(f"  Keys: {list(self.right_calibration.keys())}")
        
        stereo_image_folder = self.image_folders["stereo"]
        left_images = []
        right_images = []
        
        # Support multiple image formats
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        images = [os.path.join(stereo_image_folder, f) for f in os.listdir(stereo_image_folder) 
                 if f.lower().endswith(image_extensions)]
        for file in images:
            if(fnmatch.fnmatch(file, '*left*')):
                left_images.append(file)
            else:
                right_images.append(file)
                
        left_images.sort()  # Ensure files are in order
        right_images.sort()  # Ensure files are in order
        
        objpoints = []  # 3d point in real world space
        imgpoints_left = []  # 2d points in image plane for left camera
        imgpoints_right = []  # 2d points in image plane for right camera

        imgids_left = []
        imgids_right = []
        
        allCorners = {'left': [], 'right': []}
        allIds = {'left': [], 'right': []}

        print(f"Left images: {len(left_images)}, Right images: {len(right_images)}")

        for left_image_file, right_image_file in zip(left_images, right_images):
            left_image = cv2.imread(left_image_file)
            right_image = cv2.imread(right_image_file)
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

            # Find the Charuco corners
            leftCorners, leftIds, _ = self.marker_detector.detector.detectMarkers(left_gray)
            rightCorners, rightIds, _ = self.marker_detector.detector.detectMarkers(right_gray)
            print(f"Images: {left_image_file}, {right_image_file}")
            print(f"left corners: {len(allCorners['left'])}, left ids: {len(allIds['left'])}, right corners: {len(allCorners['right'])}, right ids: {len(allIds['right'])}")

            if leftIds is not None:
                retval_left, charuco_corners_left, charuco_ids_left = cv2.aruco.interpolateCornersCharuco(leftCorners, leftIds, left_gray, self.marker_detector.board)
                if(retval_left > 0):
                    allCorners['left'].append(charuco_corners_left)
                    allIds['left'].append(charuco_ids_left)
                
            if rightIds is not None:    
                retval_right, charuco_corners_right, charuco_ids_right = cv2.aruco.interpolateCornersCharuco(rightCorners, rightIds, right_gray, self.marker_detector.board)
                if(retval_right > 0):
                    allCorners['right'].append(charuco_corners_right)
                    allIds['right'].append(charuco_ids_right)

        # Match points and perform calibration
        matched_object_points = []
        matched_corners_left = []
        matched_corners_right = []

        for i in range(min(len(allCorners['left']), len(allCorners['right']))):
            # Ensure matching ids in both left and right images
            common_ids = np.intersect1d(allIds['left'][i], allIds['right'][i])
            if len(common_ids) > 0:
                indices_left = np.isin(allIds['left'][i], common_ids).flatten()
                indices_right = np.isin(allIds['right'][i], common_ids).flatten()

                matched_object_points.append(self.marker_detector.board.getChessboardCorners()[common_ids, :])
                matched_corners_left.append(allCorners['left'][i][indices_left])
                matched_corners_right.append(allCorners['right'][i][indices_right])


        # Now use matched corners to perform stereo calibration
        if len(matched_corners_left) and len(matched_corners_right):
            print(f"\nPerforming stereo calibration with {len(matched_corners_left)} matched pairs...")
            
            # Build stereo calibration flags
            stereo_flags = 0
            flag_descriptions = []
            
            if use_config_flags and hasattr(self, 'stereo_fix_intrinsic') and self.stereo_fix_intrinsic:
                stereo_flags |= cv2.CALIB_FIX_INTRINSIC
                flag_descriptions.append("CALIB_FIX_INTRINSIC: Trust individual calibrations")
            if use_config_flags and hasattr(self, 'stereo_fix_focal_length') and self.stereo_fix_focal_length:
                stereo_flags |= cv2.CALIB_FIX_FOCAL_LENGTH
                flag_descriptions.append("CALIB_FIX_FOCAL_LENGTH: Fix focal lengths")
            if use_config_flags and hasattr(self, 'stereo_fix_principal_point') and self.stereo_fix_principal_point:
                stereo_flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
                flag_descriptions.append("CALIB_FIX_PRINCIPAL_POINT: Fix principal points")
            if use_config_flags and hasattr(self, 'stereo_fix_aspect_ratio') and self.stereo_fix_aspect_ratio:
                stereo_flags |= cv2.CALIB_FIX_ASPECT_RATIO
                flag_descriptions.append("CALIB_FIX_ASPECT_RATIO: Fix fx/fy ratio")
            if use_config_flags and hasattr(self, 'stereo_use_intrinsic_guess') and self.stereo_use_intrinsic_guess:
                stereo_flags |= cv2.CALIB_USE_INTRINSIC_GUESS
                flag_descriptions.append("CALIB_USE_INTRINSIC_GUESS: Use provided intrinsics as initial guess")
            if use_config_flags and hasattr(self, 'stereo_same_focal_length') and self.stereo_same_focal_length:
                stereo_flags |= cv2.CALIB_SAME_FOCAL_LENGTH
                flag_descriptions.append("CALIB_SAME_FOCAL_LENGTH: Enforce identical focal lengths")
            if use_config_flags and hasattr(self, 'stereo_zero_tangent_dist') and self.stereo_zero_tangent_dist:
                stereo_flags |= cv2.CALIB_ZERO_TANGENT_DIST
                flag_descriptions.append("CALIB_ZERO_TANGENT_DIST: Set p1, p2 to zero")
            if use_config_flags and hasattr(self, 'stereo_rational_model') and self.stereo_rational_model:
                stereo_flags |= cv2.CALIB_RATIONAL_MODEL
                flag_descriptions.append("CALIB_RATIONAL_MODEL: Enable k4, k5, k6")
            if use_config_flags and hasattr(self, 'stereo_thin_prism_model') and self.stereo_thin_prism_model:
                stereo_flags |= cv2.CALIB_THIN_PRISM_MODEL
                flag_descriptions.append("CALIB_THIN_PRISM_MODEL: Enable s1, s2, s3, s4")
            if use_config_flags and hasattr(self, 'stereo_tilted_model') and self.stereo_tilted_model:
                stereo_flags |= cv2.CALIB_TILTED_MODEL
                flag_descriptions.append("CALIB_TILTED_MODEL: Enable tauX, tauY")
            if use_config_flags and hasattr(self, 'stereo_fix_k1') and self.stereo_fix_k1:
                stereo_flags |= cv2.CALIB_FIX_K1
                flag_descriptions.append("CALIB_FIX_K1: Fix k1 coefficient")
            if use_config_flags and hasattr(self, 'stereo_fix_k2') and self.stereo_fix_k2:
                stereo_flags |= cv2.CALIB_FIX_K2
                flag_descriptions.append("CALIB_FIX_K2: Fix k2 coefficient")
            if use_config_flags and hasattr(self, 'stereo_fix_k3') and self.stereo_fix_k3:
                stereo_flags |= cv2.CALIB_FIX_K3
                flag_descriptions.append("CALIB_FIX_K3: Fix k3 coefficient")
            if use_config_flags and hasattr(self, 'stereo_fix_k4') and self.stereo_fix_k4:
                stereo_flags |= cv2.CALIB_FIX_K4
                flag_descriptions.append("CALIB_FIX_K4: Fix k4 coefficient")
            if use_config_flags and hasattr(self, 'stereo_fix_k5') and self.stereo_fix_k5:
                stereo_flags |= cv2.CALIB_FIX_K5
                flag_descriptions.append("CALIB_FIX_K5: Fix k5 coefficient")
            if use_config_flags and hasattr(self, 'stereo_fix_k6') and self.stereo_fix_k6:
                stereo_flags |= cv2.CALIB_FIX_K6
                flag_descriptions.append("CALIB_FIX_K6: Fix k6 coefficient")
            
            if stereo_flags == 0:
                print("Using default stereo calibration (no flags)")
            else:
                print("Using stereo calibration flags:")
                for desc in flag_descriptions:
                    print(f"  - {desc}")
            
            ret, camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, R, T, E, F = cv2.stereoCalibrate(
                objectPoints=matched_object_points,
                imagePoints1=matched_corners_left,
                imagePoints2=matched_corners_right,
                cameraMatrix1=self.left_calibration['camera_matrix'],
                distCoeffs1=self.left_calibration['dist_coeffs'],
                cameraMatrix2=self.right_calibration['camera_matrix'],
                distCoeffs2=self.right_calibration['dist_coeffs'],
                imageSize=left_gray.shape[::-1],
                criteria=(cv2.TERM_CRITERIA_EPS +
                        cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
                flags=stereo_flags
            )
            print("Stereo Calibration Reprojection Error:\n", ret)
            print("camera matrix left:\n", camera_matrix_left)
            print("camera matrix right:\n", camera_matrix_right)
            print("Distortion coeffecient left:\n", dist_coeffs_left)
            print("Distortion coeffecient right:\n", dist_coeffs_right)
            print("\nStereo Calibration results:")
            print("Rotation Matrix:\n", R)
            print("Translation Vector:\n", T)
            print("Essential Matrix:\n", E)
            print("Fundamental Matrix:\n", F)

        data = {
            'camera_label_left': 'left',
            'camera_label_right': 'right',
            'image_width': left_gray.shape[1],
            'image_height': left_gray.shape[0],
            'model': 'stereo',
            'reprojection_error': ret,
            'camera_matrix_left': camera_matrix_left,
            'dist_coeffs_left': dist_coeffs_left,
            'camera_matrix_right': camera_matrix_right,
            'dist_coeffs_right': dist_coeffs_right,
            'R': R,
            'T': T,
            'E': E,
            'F': F
        }
        
        # Compute rectification for divergent cameras
        print("\nComputing stereo rectification for divergent cameras...")
        R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
            camera_matrix_left, dist_coeffs_left,
            camera_matrix_right, dist_coeffs_right,
            (left_gray.shape[1], left_gray.shape[0]),
            R, T,
            alpha=self.rectification_alpha,
            flags=cv2.CALIB_ZERO_DISPARITY
        )
        
        # For divergent cameras, also compute individual undistortion maps (no stereo alignment)
        print("Computing individual undistortion maps (recommended for divergent cameras)...")
        # Get optimal new camera matrices for undistortion
        alpha_left = self.rectification_alpha_left if hasattr(self, 'rectification_alpha_left') else self.rectification_alpha
        alpha_right = self.rectification_alpha_right if hasattr(self, 'rectification_alpha_right') else self.rectification_alpha
        
        new_camera_matrix_left, roi_left_undist = cv2.getOptimalNewCameraMatrix(
            camera_matrix_left, dist_coeffs_left, 
            (left_gray.shape[1], left_gray.shape[0]), 
            alpha_left
        )
        new_camera_matrix_right, roi_right_undist = cv2.getOptimalNewCameraMatrix(
            camera_matrix_right, dist_coeffs_right,
            (left_gray.shape[1], left_gray.shape[0]),
            alpha_right
        )
        
        # Create undistortion maps (no rotation, just remove distortion)
        map1_left_undist, map2_left_undist = cv2.initUndistortRectifyMap(
            camera_matrix_left, dist_coeffs_left, 
            None,  # No rotation
            new_camera_matrix_left,
            (left_gray.shape[1], left_gray.shape[0]), 
            cv2.CV_32FC1
        )
        map1_right_undist, map2_right_undist = cv2.initUndistortRectifyMap(
            camera_matrix_right, dist_coeffs_right,
            None,  # No rotation
            new_camera_matrix_right,
            (left_gray.shape[1], left_gray.shape[0]),
            cv2.CV_32FC1
        )
        
        # Compute rectification maps (for reference, but not recommended for divergent cameras)
        map1_left, map2_left = cv2.initUndistortRectifyMap(
            camera_matrix_left, dist_coeffs_left, R1, P1,
            (left_gray.shape[1], left_gray.shape[0]), cv2.CV_32FC1)
        map1_right, map2_right = cv2.initUndistortRectifyMap(
            camera_matrix_right, dist_coeffs_right, R2, P2,
            (left_gray.shape[1], left_gray.shape[0]), cv2.CV_32FC1)
        
        # Compute overlap masks
        print("Computing overlap masks...")
        overlap_mask_left, overlap_mask_right, overlap_info = self.compute_overlap_mask(
            camera_matrix_left, dist_coeffs_left,
            camera_matrix_right, dist_coeffs_right,
            R, T,
            (left_gray.shape[1], left_gray.shape[0])
        )
        
        # Also compute overlap masks for RECTIFIED images (more useful for stereo matching)
        print("Computing overlap masks for rectified images...")
        overlap_mask_left_rect, overlap_mask_right_rect, overlap_info_rect = self.compute_overlap_mask_rectified(
            P1, P2, R1, R2,
            (left_gray.shape[1], left_gray.shape[0])
        )
        
        # Compute overlap from actual marker detections in matched stereo pairs (most accurate)
        print("Computing overlap masks from detected markers in matched stereo pairs...")
        # Build matched IDs for overlap computation
        matched_ids_left = []
        matched_ids_right = []
        for i in range(min(len(allCorners['left']), len(allCorners['right']))):
            common_ids = np.intersect1d(allIds['left'][i], allIds['right'][i])
            if len(common_ids) > 0:
                indices_left = np.isin(allIds['left'][i], common_ids).flatten()
                indices_right = np.isin(allIds['right'][i], common_ids).flatten()
                matched_ids_left.append(allIds['left'][i][indices_left])
                matched_ids_right.append(allIds['right'][i][indices_right])
        
        overlap_mask_left_markers, overlap_mask_right_markers, overlap_info_markers = self.compute_overlap_from_markers(
            {'left': matched_corners_left, 'right': matched_corners_right},
            {'left': matched_ids_left, 'right': matched_ids_right},
            (left_gray.shape[1], left_gray.shape[0])
        )
        
        print(f"\nOverlap Analysis (Original Images):")
        print(f"  Left camera overlap: {overlap_info['left_overlap_percent']:.1f}% of image")
        print(f"  Right camera overlap: {overlap_info['right_overlap_percent']:.1f}% of image")
        print(f"  Overlap region: {overlap_info['overlap_width']}x{overlap_info['overlap_height']} pixels")
        print(f"  Baseline distance: {np.linalg.norm(T):.3f}m")
        print(f"  Vergence angle: {overlap_info['vergence_angle']:.2f}°")
        
        print(f"\nOverlap Analysis (Rectified Images):")
        print(f"  Left camera overlap: {overlap_info_rect['left_overlap_percent']:.1f}% of image")
        print(f"  Right camera overlap: {overlap_info_rect['right_overlap_percent']:.1f}% of image")
        print(f"  Overlap region: {overlap_info_rect['overlap_width']}x{overlap_info_rect['overlap_height']} pixels")
        
        print(f"\nOverlap Analysis (From Detected Markers - Most Accurate):")
        print(f"  Left camera overlap: {overlap_info_markers['left_overlap_percent']:.1f}% of image")
        print(f"  Right camera overlap: {overlap_info_markers['right_overlap_percent']:.1f}% of image")
        print(f"  Overlap region: {overlap_info_markers['overlap_width']}x{overlap_info_markers['overlap_height']} pixels")
        print(f"  Common markers detected: {overlap_info_markers['common_markers']}/{overlap_info_markers['total_markers']}")
        
        # Save only raw calibration data (R, T, E, F)
        # Computed data (rectification, undistortion maps, overlap masks) can be regenerated when needed
        # Note: R1, R2, P1, P2, Q, new_camera_matrix, and overlap masks not saved - compute on-demand
        
        self.save_calibration("stereo_calibration", data, save_format)
        
        # Save overlap masks as separate images for easy inspection
        overlap_folder = self.image_folders.get('calibration', 'calibration')
        os.makedirs(overlap_folder, exist_ok=True)
        cv2.imwrite(os.path.join(overlap_folder, 'overlap_mask_left_original.png'), overlap_mask_left)
        cv2.imwrite(os.path.join(overlap_folder, 'overlap_mask_right_original.png'), overlap_mask_right)
        cv2.imwrite(os.path.join(overlap_folder, 'overlap_mask_left_rectified.png'), overlap_mask_left_rect)
        cv2.imwrite(os.path.join(overlap_folder, 'overlap_mask_right_rectified.png'), overlap_mask_right_rect)
        cv2.imwrite(os.path.join(overlap_folder, 'overlap_mask_left_markers.png'), overlap_mask_left_markers)
        cv2.imwrite(os.path.join(overlap_folder, 'overlap_mask_right_markers.png'), overlap_mask_right_markers)
        print(f"\nSaved overlap masks to {overlap_folder}/overlap_mask_*.png")
        print(f"  Recommended: Use overlap_mask_*_markers.png (based on actual detections)")
        
        # Create example rectified image pair from first stereo pair
        generate_debug = hasattr(self, 'generate_debug_images') and self.generate_debug_images
        if generate_debug and len(left_images) > 0 and len(right_images) > 0:
            print("\nCreating example undistorted images (recommended for divergent cameras)...")
            first_left = cv2.imread(left_images[0])
            first_right = cv2.imread(right_images[0])
            
            if first_left is not None and first_right is not None:
                # Compute undistortion maps (not saved to reduce file size)
                map1_left_undist, map2_left_undist = cv2.initUndistortRectifyMap(
                    camera_matrix_left, dist_coeffs_left, None, new_camera_matrix_left, 
                    (first_left.shape[1], first_left.shape[0]), cv2.CV_32FC1
                )
                map1_right_undist, map2_right_undist = cv2.initUndistortRectifyMap(
                    camera_matrix_right, dist_coeffs_right, None, new_camera_matrix_right, 
                    (first_right.shape[1], first_right.shape[0]), cv2.CV_32FC1
                )
                
                # Apply undistortion
                left_undistorted = cv2.remap(first_left, map1_left_undist, map2_left_undist, cv2.INTER_LINEAR)
                right_undistorted = cv2.remap(first_right, map1_right_undist, map2_right_undist, cv2.INTER_LINEAR)
                
                # Draw valid ROI on undistorted images
                left_undist_roi = left_undistorted.copy()
                right_undist_roi = right_undistorted.copy()
                x, y, w, h = roi_left_undist
                if w > 0 and h > 0:
                    cv2.rectangle(left_undist_roi, (x, y), (x+w, y+h), (0, 255, 0), 2)
                x, y, w, h = roi_right_undist
                if w > 0 and h > 0:
                    cv2.rectangle(right_undist_roi, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Save undistorted images
                cv2.imwrite(os.path.join(overlap_folder, 'example_left_undistorted.png'), left_undistorted)
                cv2.imwrite(os.path.join(overlap_folder, 'example_right_undistorted.png'), right_undistorted)
                
                # Create comparison images with different alpha values
                print("Creating alpha comparison images...")
                alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                alpha_comparison_rows = []
                
                for alpha_val in alpha_values:
                    # Compute new camera matrices with this alpha
                    new_K_left, roi_l = cv2.getOptimalNewCameraMatrix(
                        camera_matrix_left, dist_coeffs_left,
                        (first_left.shape[1], first_left.shape[0]), alpha_val
                    )
                    new_K_right, roi_r = cv2.getOptimalNewCameraMatrix(
                        camera_matrix_right, dist_coeffs_right,
                        (first_right.shape[1], first_right.shape[0]), alpha_val
                    )
                    
                    # Create undistortion maps
                    m1_l, m2_l = cv2.initUndistortRectifyMap(
                        camera_matrix_left, dist_coeffs_left, None, new_K_left,
                        (first_left.shape[1], first_left.shape[0]), cv2.CV_32FC1
                    )
                    m1_r, m2_r = cv2.initUndistortRectifyMap(
                        camera_matrix_right, dist_coeffs_right, None, new_K_right,
                        (first_right.shape[1], first_right.shape[0]), cv2.CV_32FC1
                    )
                    
                    # Apply undistortion
                    left_undist = cv2.remap(first_left, m1_l, m2_l, cv2.INTER_LINEAR)
                    right_undist = cv2.remap(first_right, m1_r, m2_r, cv2.INTER_LINEAR)
                    
                    # Draw ROI boxes
                    left_with_roi = left_undist.copy()
                    right_with_roi = right_undist.copy()
                    x, y, w, h = roi_l
                    if w > 0 and h > 0:
                        cv2.rectangle(left_with_roi, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    x, y, w, h = roi_r
                    if w > 0 and h > 0:
                        cv2.rectangle(right_with_roi, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Create side-by-side pair with label
                    pair = np.hstack([left_with_roi, right_with_roi])
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(pair, f'alpha = {alpha_val:.2f}', (20, 40), font, 1.2, (255, 255, 255), 2)
                    alpha_comparison_rows.append(pair)
                
                # Stack all alpha comparisons vertically
                alpha_comparison = np.vstack(alpha_comparison_rows)
                cv2.imwrite(os.path.join(overlap_folder, 'alpha_comparison.png'), alpha_comparison)
                print(f"Saved alpha comparison to {overlap_folder}/alpha_comparison.png")
                print(f"  Alpha values: {alpha_values}")
                print(f"  Recommendation: Use alpha=0.0 for clean images, or 0.25-0.5 for balance")
                
                # Create individual alpha comparison for left camera only
                print("Creating left camera alpha comparison...")
                left_alpha_rows = []
                for alpha_val in alpha_values:
                    new_K_left, roi_l = cv2.getOptimalNewCameraMatrix(
                        camera_matrix_left, dist_coeffs_left,
                        (first_left.shape[1], first_left.shape[0]), alpha_val
                    )
                    m1_l, m2_l = cv2.initUndistortRectifyMap(
                        camera_matrix_left, dist_coeffs_left, None, new_K_left,
                        (first_left.shape[1], first_left.shape[0]), cv2.CV_32FC1
                    )
                    left_undist = cv2.remap(first_left, m1_l, m2_l, cv2.INTER_LINEAR)
                    left_with_roi = left_undist.copy()
                    x, y, w, h = roi_l
                    if w > 0 and h > 0:
                        cv2.rectangle(left_with_roi, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(left_with_roi, f'Left alpha = {alpha_val:.2f}', (20, 40), font, 1.2, (255, 255, 255), 2)
                    left_alpha_rows.append(left_with_roi)
                
                left_alpha_comparison = np.vstack(left_alpha_rows)
                cv2.imwrite(os.path.join(overlap_folder, 'alpha_comparison_left.png'), left_alpha_comparison)
                print(f"Saved left camera alpha comparison to {overlap_folder}/alpha_comparison_left.png")
                
                # Create individual alpha comparison for right camera only
                print("Creating right camera alpha comparison...")
                right_alpha_rows = []
                for alpha_val in alpha_values:
                    new_K_right, roi_r = cv2.getOptimalNewCameraMatrix(
                        camera_matrix_right, dist_coeffs_right,
                        (first_right.shape[1], first_right.shape[0]), alpha_val
                    )
                    m1_r, m2_r = cv2.initUndistortRectifyMap(
                        camera_matrix_right, dist_coeffs_right, None, new_K_right,
                        (first_right.shape[1], first_right.shape[0]), cv2.CV_32FC1
                    )
                    right_undist = cv2.remap(first_right, m1_r, m2_r, cv2.INTER_LINEAR)
                    right_with_roi = right_undist.copy()
                    x, y, w, h = roi_r
                    if w > 0 and h > 0:
                        cv2.rectangle(right_with_roi, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(right_with_roi, f'Right alpha = {alpha_val:.2f}', (20, 40), font, 1.2, (255, 255, 255), 2)
                    right_alpha_rows.append(right_with_roi)
                
                right_alpha_comparison = np.vstack(right_alpha_rows)
                cv2.imwrite(os.path.join(overlap_folder, 'alpha_comparison_right.png'), right_alpha_comparison)
                print(f"Saved right camera alpha comparison to {overlap_folder}/alpha_comparison_right.png")
                
                # Create overlays with marker-based overlap masks on ORIGINAL images
                first_left_overlay = first_left.copy()
                first_right_overlay = first_right.copy()
                
                # Apply semi-transparent green to overlap regions
                alpha = 0.3
                green = np.zeros_like(first_left_overlay)
                green[:] = (0, 255, 0)
                
                mask_left_3ch = cv2.cvtColor(overlap_mask_left_markers, cv2.COLOR_GRAY2BGR)
                mask_right_3ch = cv2.cvtColor(overlap_mask_right_markers, cv2.COLOR_GRAY2BGR)
                
                first_left_overlay = np.where(mask_left_3ch > 0, 
                                               cv2.addWeighted(first_left_overlay, 1 - alpha, green, alpha, 0),
                                               first_left_overlay)
                first_right_overlay = np.where(mask_right_3ch > 0,
                                                cv2.addWeighted(first_right_overlay, 1 - alpha, green, alpha, 0),
                                                first_right_overlay)
                
                # Create side-by-side comparison
                original_pair = np.hstack([first_left, first_right])
                original_pair_overlay = np.hstack([first_left_overlay, first_right_overlay])
                undistorted_pair = np.hstack([left_undist_roi, right_undist_roi])
                
                # Stack original (with overlay) and undistorted vertically
                comparison = np.vstack([original_pair_overlay, undistorted_pair])
                
                # Add labels
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(comparison, 'Original Images with Overlap (green)', (20, 40), font, 1.2, (255, 255, 255), 2)
                cv2.putText(comparison, 'Undistorted Images (green box = valid ROI)', 
                           (20, first_left.shape[0] + 40), font, 1.2, (255, 255, 255), 2)
                
                cv2.imwrite(os.path.join(overlap_folder, 'undistortion_comparison.png'), comparison)
                print(f"Saved example undistorted images to {overlap_folder}/")
                print(f"  - example_left_undistorted.png")
                print(f"  - example_right_undistorted.png")
                print(f"  - undistortion_comparison.png (shows overlap regions and valid ROI)")
                print(f"\nNote: For divergent cameras, individual undistortion is recommended.")
                print(f"      Stereo rectification (alignment) is NOT suitable for divergent cameras.")
        elif not generate_debug:
            print("\nDebug images disabled (set generate_debug_images=true in config to enable)")
        
        print("Stereo Calibration complete")
    
    
    def compute_overlap_mask(self, K1, D1, K2, D2, R, T, image_size):
        """
        Compute overlap masks for divergent stereo cameras.
        
        Args:
            K1, D1: Camera matrix and distortion coefficients for left camera
            K2, D2: Camera matrix and distortion coefficients for right camera
            R, T: Rotation and translation from left to right camera
            image_size: (width, height) of images
            
        Returns:
            overlap_mask_left: Binary mask showing overlap region in left camera
            overlap_mask_right: Binary mask showing overlap region in right camera
            overlap_info: Dictionary with overlap statistics
        """
        width, height = image_size
        
        # Create meshgrid of pixel coordinates
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        pixels = np.stack([x_coords, y_coords], axis=-1).reshape(-1, 2).astype(np.float32)
        
        # Undistort points from left camera
        points_undist_left = cv2.undistortPoints(
            pixels.reshape(-1, 1, 2), K1, D1, P=K1
        ).reshape(-1, 2)
        
        # Convert to normalized camera coordinates
        points_norm_left = cv2.undistortPoints(
            pixels.reshape(-1, 1, 2), K1, D1
        ).reshape(-1, 2)
        
        # Add z=1 to make homogeneous 3D points
        points_3d = np.column_stack([points_norm_left, np.ones(len(points_norm_left))])
        
        # Transform to right camera coordinate system
        points_3d_right = (R @ points_3d.T).T + T.reshape(1, 3)
        
        # Project to right camera image plane
        points_2d_right = points_3d_right[:, :2] / points_3d_right[:, 2:3]
        
        # Apply camera matrix and distortion
        rvec = np.zeros(3)
        tvec = np.zeros(3)
        points_right_distorted, _ = cv2.projectPoints(
            points_3d_right, rvec, tvec, K2, D2
        )
        points_right_distorted = points_right_distorted.reshape(-1, 2)
        
        # Check which points from left camera fall within right camera's FOV
        valid_in_right = (
            (points_right_distorted[:, 0] >= 0) &
            (points_right_distorted[:, 0] < width) &
            (points_right_distorted[:, 1] >= 0) &
            (points_right_distorted[:, 1] < height) &
            (points_3d_right[:, 2] > 0)  # In front of camera
        )
        
        # Similarly check points from right camera visible in left
        pixels_right = pixels.copy()
        points_norm_right = cv2.undistortPoints(
            pixels_right.reshape(-1, 1, 2), K2, D2
        ).reshape(-1, 2)
        
        points_3d_right_cam = np.column_stack([points_norm_right, np.ones(len(points_norm_right))])
        
        # Transform to left camera coordinate system
        R_inv = R.T
        T_inv = -R.T @ T
        points_3d_left = (R_inv @ points_3d_right_cam.T).T + T_inv.reshape(1, 3)
        
        # Project to left camera
        points_left_distorted, _ = cv2.projectPoints(
            points_3d_left, rvec, tvec, K1, D1
        )
        points_left_distorted = points_left_distorted.reshape(-1, 2)
        
        valid_in_left = (
            (points_left_distorted[:, 0] >= 0) &
            (points_left_distorted[:, 0] < width) &
            (points_left_distorted[:, 1] >= 0) &
            (points_left_distorted[:, 1] < height) &
            (points_3d_left[:, 2] > 0)
        )
        
        # Create masks
        overlap_mask_left = valid_in_right.reshape(height, width).astype(np.uint8) * 255
        overlap_mask_right = valid_in_left.reshape(height, width).astype(np.uint8) * 255
        
        # Compute overlap statistics
        left_overlap_pixels = np.sum(valid_in_right)
        right_overlap_pixels = np.sum(valid_in_left)
        total_pixels = width * height
        
        # Find bounding box of overlap region in left camera
        overlap_coords_left = np.where(overlap_mask_left > 0)
        if len(overlap_coords_left[0]) > 0:
            y_min_left = np.min(overlap_coords_left[0])
            y_max_left = np.max(overlap_coords_left[0])
            x_min_left = np.min(overlap_coords_left[1])
            x_max_left = np.max(overlap_coords_left[1])
            overlap_width_left = x_max_left - x_min_left
            overlap_height_left = y_max_left - y_min_left
        else:
            overlap_width_left = overlap_height_left = 0
        
        # Compute vergence angle (angle between optical axes)
        # Optical axis of left camera in left camera coords
        optical_axis_left = np.array([0, 0, 1])
        # Optical axis of right camera in right camera coords
        optical_axis_right = np.array([0, 0, 1])
        # Transform right camera optical axis to left camera coords
        optical_axis_right_in_left = R_inv @ optical_axis_right
        # Compute angle
        cos_angle = np.dot(optical_axis_left, optical_axis_right_in_left)
        vergence_angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        
        overlap_info = {
            'left_overlap_percent': 100 * left_overlap_pixels / total_pixels,
            'right_overlap_percent': 100 * right_overlap_pixels / total_pixels,
            'overlap_width': overlap_width_left,
            'overlap_height': overlap_height_left,
            'vergence_angle': vergence_angle,
            'baseline': float(np.linalg.norm(T))
        }
        
        return overlap_mask_left, overlap_mask_right, overlap_info
    
    
    def compute_overlap_mask_rectified(self, P1, P2, R1, R2, image_size):
        """
        Compute overlap masks for rectified stereo images.
        After rectification, the images are aligned so corresponding points are on the same row.
        The overlap is much simpler to compute - just find the horizontal overlap region.
        
        Args:
            P1, P2: Projection matrices for left and right cameras after rectification
            R1, R2: Rectification rotation matrices
            image_size: (width, height) of images
            
        Returns:
            overlap_mask_left: Binary mask showing overlap region in left rectified image
            overlap_mask_right: Binary mask showing overlap region in right rectified image
            overlap_info: Dictionary with overlap statistics
        """
        width, height = image_size
        
        # For rectified images, we need to find which columns in each image correspond
        # to the same 3D space. This is done by checking the disparity range.
        
        # Create masks - initially all white (full overlap)
        overlap_mask_left = np.ones((height, width), dtype=np.uint8) * 255
        overlap_mask_right = np.ones((height, width), dtype=np.uint8) * 255
        
        # The principal points tell us where the optical axes intersect the image plane
        cx_left = P1[0, 2]
        cx_right = P2[0, 2]
        
        # The baseline in rectified coordinates
        baseline = P2[0, 3] / P2[0, 0] if P2[0, 0] != 0 else 0
        
        # For divergent cameras, the overlap region is where both cameras can see
        # In rectified space, this is typically a central region
        # Left image: right side overlaps with right camera
        # Right image: left side overlaps with left camera
        
        # Simple approach: mark the overlap based on the offset between principal points
        offset = int(abs(cx_right - cx_left))
        
        if cx_left < cx_right:
            # Left camera sees more on the left, right camera sees more on the right
            # Overlap is the right portion of left image and left portion of right image
            overlap_mask_left[:, :max(0, width - offset)] = 0
            overlap_mask_right[:, min(width, offset):] = 0
        else:
            # Right camera sees more on the left, left camera sees more on the right
            overlap_mask_left[:, min(width, offset):] = 0
            overlap_mask_right[:, :max(0, width - offset)] = 0
        
        # Compute statistics
        left_overlap_pixels = np.sum(overlap_mask_left > 0)
        right_overlap_pixels = np.sum(overlap_mask_right > 0)
        total_pixels = width * height
        
        # Find bounding box of overlap
        overlap_coords_left = np.where(overlap_mask_left > 0)
        if len(overlap_coords_left[0]) > 0:
            y_min = np.min(overlap_coords_left[0])
            y_max = np.max(overlap_coords_left[0])
            x_min = np.min(overlap_coords_left[1])
            x_max = np.max(overlap_coords_left[1])
            overlap_width = x_max - x_min
            overlap_height = y_max - y_min
        else:
            overlap_width = overlap_height = 0
        
        overlap_info = {
            'left_overlap_percent': 100 * left_overlap_pixels / total_pixels,
            'right_overlap_percent': 100 * right_overlap_pixels / total_pixels,
            'overlap_width': overlap_width,
            'overlap_height': overlap_height,
            'baseline': float(abs(baseline))
        }
        
        return overlap_mask_left, overlap_mask_right, overlap_info
    
    
    def compute_overlap_from_markers(self, matchedCorners, matchedIds, image_size):
        """
        Compute overlap masks based on actual detected markers in MATCHED stereo pairs.
        This is the most accurate method as it uses only the calibration pairs with common markers.
        
        Args:
            matchedCorners: Dictionary with 'left' and 'right' lists of matched corners (only pairs with common markers)
            matchedIds: Dictionary with 'left' and 'right' lists of matched marker IDs
            image_size: (width, height) of images
            
        Returns:
            overlap_mask_left: Binary mask showing overlap region in left camera
            overlap_mask_right: Binary mask showing overlap region in right camera
            overlap_info: Dictionary with overlap statistics
        """
        width, height = image_size
        
        # Collect all marker corner positions from matched stereo pairs
        left_points = []
        right_points = []
        common_marker_count = 0
        total_left_markers = set()
        total_right_markers = set()
        pairs_processed = len(matchedCorners['left'])
        
        # Iterate through matched stereo pairs only
        for i in range(len(matchedCorners['left'])):
            left_ids = matchedIds['left'][i].flatten() if matchedIds['left'][i] is not None else []
            right_ids = matchedIds['right'][i].flatten() if matchedIds['right'][i] is not None else []
            
            total_left_markers.update(left_ids)
            total_right_markers.update(right_ids)
            
            common_marker_count = max(common_marker_count, len(left_ids))  # These are already matched
            
            # Extract corner positions (already matched)
            left_corners = matchedCorners['left'][i].reshape(-1, 2)
            right_corners = matchedCorners['right'][i].reshape(-1, 2)
            
            left_points.extend(left_corners)
            right_points.extend(right_corners)
        
        print(f"  Processed {pairs_processed} stereo pairs with common markers")
        print(f"  Collected {len(left_points)} corner points from left camera")
        print(f"  Collected {len(right_points)} corner points from right camera")
        
        if len(left_points) == 0 or len(right_points) == 0:
            # No overlap found - return empty masks
            print("Warning: No common markers detected in stereo pairs!")
            overlap_mask_left = np.zeros((height, width), dtype=np.uint8)
            overlap_mask_right = np.zeros((height, width), dtype=np.uint8)
            overlap_info = {
                'left_overlap_percent': 0,
                'right_overlap_percent': 0,
                'overlap_width': 0,
                'overlap_height': 0,
                'common_markers': 0,
                'total_markers': 0
            }
            return overlap_mask_left, overlap_mask_right, overlap_info
        
        # Convert to numpy arrays
        left_points = np.array(left_points)
        right_points = np.array(right_points)
        
        # Compute convex hull for each camera's overlap region
        from scipy.spatial import ConvexHull
        
        # Left camera overlap region
        hull_left = ConvexHull(left_points)
        hull_points_left = left_points[hull_left.vertices].astype(np.int32)
        
        # Right camera overlap region
        hull_right = ConvexHull(right_points)
        hull_points_right = right_points[hull_right.vertices].astype(np.int32)
        
        # Create masks by filling the convex hulls
        overlap_mask_left = np.zeros((height, width), dtype=np.uint8)
        overlap_mask_right = np.zeros((height, width), dtype=np.uint8)
        
        cv2.fillConvexPoly(overlap_mask_left, hull_points_left, 255)
        cv2.fillConvexPoly(overlap_mask_right, hull_points_right, 255)
        
        # Compute statistics
        left_overlap_pixels = np.sum(overlap_mask_left > 0)
        right_overlap_pixels = np.sum(overlap_mask_right > 0)
        total_pixels = width * height
        
        # Find bounding box
        overlap_coords_left = np.where(overlap_mask_left > 0)
        if len(overlap_coords_left[0]) > 0:
            y_min = np.min(overlap_coords_left[0])
            y_max = np.max(overlap_coords_left[0])
            x_min = np.min(overlap_coords_left[1])
            x_max = np.max(overlap_coords_left[1])
            overlap_width = x_max - x_min
            overlap_height = y_max - y_min
        else:
            overlap_width = overlap_height = 0
        
        total_unique_markers = len(total_left_markers.union(total_right_markers))
        
        overlap_info = {
            'left_overlap_percent': 100 * left_overlap_pixels / total_pixels,
            'right_overlap_percent': 100 * right_overlap_pixels / total_pixels,
            'overlap_width': overlap_width,
            'overlap_height': overlap_height,
            'common_markers': common_marker_count,
            'total_markers': total_unique_markers
        }
        
        return overlap_mask_left, overlap_mask_right, overlap_info
    
    
    def visualize_stereo_overlap(self, left_image_path, right_image_path, stereo_calib_file):
        """
        Visualize the overlap region on actual stereo images.
        
        Args:
            left_image_path: Path to left camera image
            right_image_path: Path to right camera image
            stereo_calib_file: Path to stereo calibration file
        """
        # Load stereo calibration
        if stereo_calib_file.endswith('.npz'):
            data = dict(np.load(stereo_calib_file))
        elif stereo_calib_file.endswith('.yaml'):
            fs = cv2.FileStorage(stereo_calib_file, cv2.FILE_STORAGE_READ)
            data = {}
            root = fs.root()
            for i in range(root.size()):
                node = root.at(i)
                key = node.name()
                if node.isMat():
                    data[key] = node.mat()
            fs.release()
        else:
            print("Error: Unsupported calibration file format")
            return
        
        # Load images
        left_img = cv2.imread(left_image_path)
        right_img = cv2.imread(right_image_path)
        
        if left_img is None or right_img is None:
            print("Error: Could not load images")
            return
        
        # Get overlap masks
        overlap_mask_left = data.get('overlap_mask_left')
        overlap_mask_right = data.get('overlap_mask_right')
        
        if overlap_mask_left is None or overlap_mask_right is None:
            print("Error: Overlap masks not found in calibration file")
            return
        
        # Create colored overlay (green = overlap region)
        overlay_left = left_img.copy()
        overlay_right = right_img.copy()
        
        # Apply green tint to overlap regions
        overlay_left[overlap_mask_left > 0] = cv2.addWeighted(
            overlay_left[overlap_mask_left > 0], 0.7,
            np.array([0, 255, 0], dtype=np.uint8), 0.3, 0
        )
        overlay_right[overlap_mask_right > 0] = cv2.addWeighted(
            overlay_right[overlap_mask_right > 0], 0.7,
            np.array([0, 255, 0], dtype=np.uint8), 0.3, 0
        )
        
        # Create side-by-side visualization
        combined = np.hstack([overlay_left, overlay_right])
        
        # Create transparent shading visualization
        # Non-overlap regions get semi-transparent dark overlay
        shaded_left = left_img.copy().astype(np.float32)
        shaded_right = right_img.copy().astype(np.float32)
        
        # Darken non-overlap areas (multiply by 0.3 for 70% transparency effect)
        shaded_left[overlap_mask_left == 0] = shaded_left[overlap_mask_left == 0] * 0.3
        shaded_right[overlap_mask_right == 0] = shaded_right[overlap_mask_right == 0] * 0.3
        
        # Convert back to uint8
        shaded_left = shaded_left.astype(np.uint8)
        shaded_right = shaded_right.astype(np.uint8)
        
        # Create side-by-side with shading
        combined_shaded = np.hstack([shaded_left, shaded_right])
        
        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        color = (255, 255, 255)
        
        # Labels for green overlay version
        cv2.putText(combined, 'Left Camera', (20, 40), font, font_scale, color, thickness)
        cv2.putText(combined, 'Right Camera', (left_img.shape[1] + 20, 40), 
                   font, font_scale, color, thickness)
        cv2.putText(combined, 'Green = Overlap Region', (20, combined.shape[0] - 20), 
                   font, font_scale, (0, 255, 0), thickness)
        
        # Labels for shaded version
        cv2.putText(combined_shaded, 'Left Camera', (20, 40), font, font_scale, color, thickness)
        cv2.putText(combined_shaded, 'Right Camera', (shaded_left.shape[1] + 20, 40), 
                   font, font_scale, color, thickness)
        cv2.putText(combined_shaded, 'Bright = Overlap Region', (20, combined_shaded.shape[0] - 20), 
                   font, font_scale, color, thickness)
        
        # Stack both visualizations vertically
        final_visualization = np.vstack([combined, combined_shaded])
        
        # Display both versions
        cv2.imshow('Stereo Overlap Visualization', final_visualization)
        print("\nShowing overlap visualization:")
        print("  Top row: Green overlay marks overlap regions")
        print("  Bottom row: Transparent shading (bright = overlap, dark = non-overlap)")
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save visualizations
        os.makedirs('calibration', exist_ok=True)
        
        # Save combined version with both styles
        output_combined = 'calibration/stereo_overlap_visualization.png'
        cv2.imwrite(output_combined, final_visualization)
        print(f"\nSaved combined visualization to {output_combined}")
        
        # Save individual versions
        output_green = 'calibration/stereo_overlap_green.png'
        cv2.imwrite(output_green, combined)
        print(f"Saved green overlay version to {output_green}")
        
        output_shaded = 'calibration/stereo_overlap_shaded.png'
        cv2.imwrite(output_shaded, combined_shaded)
        print(f"Saved transparent shading version to {output_shaded}")


    def convert_calibration(self, input_file, output_format):
        """
        Convert existing calibration file to different format.
        
        Args:
            input_file: Path to existing calibration file (.npz or .yaml)
            output_format: Target format - 'npz', 'yaml', or 'both'
        """
        # Determine input format and base name
        if input_file.endswith('.npz'):
            base_name = input_file[:-4]
            data = dict(np.load(input_file))
            print(f"Loaded NPZ file: {input_file}")
        elif input_file.endswith('.yaml') or input_file.endswith('.yml'):
            base_name = input_file.rsplit('.', 1)[0]
            # Read YAML using OpenCV FileStorage
            fs = cv2.FileStorage(input_file, cv2.FILE_STORAGE_READ)
            data = {}
            
            # Read all nodes from the file
            root = fs.root()
            for i in range(root.size()):
                node = root.at(i)
                key = node.name()
                if node.isSeq():
                    # Sequence of matrices (rvecs, tvecs)
                    seq_data = []
                    for j in range(node.size()):
                        seq_data.append(node.at(j).mat())
                    data[key] = seq_data
                elif node.isMat():
                    data[key] = node.mat()
                elif node.isInt():
                    data[key] = node.real()
                elif node.isReal():
                    data[key] = node.real()
                elif node.isString():
                    data[key] = node.string()
            
            fs.release()
            print(f"Loaded YAML file: {input_file}")
        else:
            print(f"Error: Unsupported file format. Use .npz or .yaml files.")
            return
        
        # Save in requested format(s)
        self.save_calibration(base_name, data, output_format)
        print(f"Conversion complete: {input_file} -> {output_format}")


    def load_calibrations(self):
        """Load previously saved calibration data from files (supports .npz and .yaml)."""
        # Try to load left calibration
        for ext in ['.yaml', '.yml', '.npz']:
            left_file = f'calibration/left{ext}'
            if os.path.exists(left_file):
                print(f"Loading left calibration from {left_file}")
                if ext == '.npz':
                    self.left_calibration = dict(np.load(left_file))
                else:
                    self.left_calibration = self._load_yaml_calibration(left_file)
                break
        else:
            # Fallback to old location
            if os.path.exists('left.npz'):
                print("Loading left calibration from left.npz")
                self.left_calibration = dict(np.load('left.npz'))
        
        # Try to load right calibration
        for ext in ['.yaml', '.yml', '.npz']:
            right_file = f'calibration/right{ext}'
            if os.path.exists(right_file):
                print(f"Loading right calibration from {right_file}")
                if ext == '.npz':
                    self.right_calibration = dict(np.load(right_file))
                else:
                    self.right_calibration = self._load_yaml_calibration(right_file)
                break
        else:
            # Fallback to old location
            if os.path.exists('right.npz'):
                print("Loading right calibration from right.npz")
                self.right_calibration = dict(np.load('right.npz'))
        
        # Try to load fisheye calibrations
        for ext in ['.yaml', '.yml', '.npz']:
            left_fisheye_file = f'calibration/left_fisheye{ext}'
            if os.path.exists(left_fisheye_file):
                if ext == '.npz':
                    self.left_fisheye_calibration = dict(np.load(left_fisheye_file))
                else:
                    self.left_fisheye_calibration = self._load_yaml_calibration(left_fisheye_file)
                break
        else:
            if os.path.exists('left_fisheye.npz'):
                self.left_fisheye_calibration = dict(np.load('left_fisheye.npz'))
        
        for ext in ['.yaml', '.yml', '.npz']:
            right_fisheye_file = f'calibration/right_fisheye{ext}'
            if os.path.exists(right_fisheye_file):
                if ext == '.npz':
                    self.right_fisheye_calibration = dict(np.load(right_fisheye_file))
                else:
                    self.right_fisheye_calibration = self._load_yaml_calibration(right_fisheye_file)
                break
        else:
            if os.path.exists('right_fisheye.npz'):
                self.right_fisheye_calibration = dict(np.load('right_fisheye.npz'))
        
        # Try to load stereo calibration
        for ext in ['.yaml', '.yml', '.npz']:
            stereo_file = f'calibration/stereo_calibration{ext}'
            if os.path.exists(stereo_file):
                if ext == '.npz':
                    self.stereo_calibration = dict(np.load(stereo_file))
                else:
                    self.stereo_calibration = self._load_yaml_calibration(stereo_file)
                break
        else:
            if os.path.exists('stereo_calibration.npz'):
                self.stereo_calibration = dict(np.load('stereo_calibration.npz'))
    
    
    def _load_yaml_calibration(self, filepath):
        """Helper method to load calibration from YAML file."""
        fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
        data = {}
        
        # Read common calibration keys using direct access
        # String values
        camera_label = fs.getNode('camera_label')
        if not camera_label.empty():
            data['camera_label'] = camera_label.string()
        
        model = fs.getNode('model')
        if not model.empty():
            data['model'] = model.string()
        
        # Integer values
        image_width = fs.getNode('image_width')
        if not image_width.empty():
            data['image_width'] = int(image_width.real())
        
        image_height = fs.getNode('image_height')
        if not image_height.empty():
            data['image_height'] = int(image_height.real())
        
        # Matrix values - these are the critical ones
        matrix_keys = [
            'camera_matrix', 'dist_coeffs', 
            'camera_matrix_left', 'dist_coeffs_left',
            'camera_matrix_right', 'dist_coeffs_right',
            'R', 'T', 'E', 'F',
            'R1', 'R2', 'P1', 'P2', 'Q',
            'map1_left', 'map2_left', 'map1_right', 'map2_right',
            'overlap_mask_left', 'overlap_mask_right'
        ]
        
        for key in matrix_keys:
            node = fs.getNode(key)
            if not node.empty():
                mat_data = node.mat()
                if mat_data is not None:
                    data[key] = mat_data
        
        # Tuple values (ROI)
        roi_left = fs.getNode('roi_left')
        if not roi_left.empty():
            data['roi_left'] = tuple(int(roi_left.at(i).real()) for i in range(4))
        
        roi_right = fs.getNode('roi_right')
        if not roi_right.empty():
            data['roi_right'] = tuple(int(roi_right.at(i).real()) for i in range(4))
        
        # Sequence of matrices (rvecs, tvecs)
        seq_keys = ['rvecs', 'tvecs']
        for key in seq_keys:
            node = fs.getNode(key)
            if not node.empty() and node.isSeq():
                seq_data = []
                for j in range(node.size()):
                    item = node.at(j)
                    mat_data = item.mat()
                    if mat_data is not None:
                        seq_data.append(mat_data)
                if seq_data:
                    data[key] = seq_data
        
        fs.release()
        return data


    def rectify_images(self, left_image, right_image):
        """
        Rectify stereo image pair using stereo calibration data.
        
        Args:
            left_image: Left camera image
            right_image: Right camera image
            
        Returns:
            Tuple of (left_rectified, right_rectified) images
        """
        left_camera_matrix = self.stereo_calibration['camera_matrix_left']
        left_dist_coeffs = self.stereo_calibration['dist_coeffs_left']
        right_camera_matrix = self.stereo_calibration['camera_matrix_right']
        right_dist_coeffs = self.stereo_calibration['dist_coeffs_right']
        rotation = self.stereo_calibration['R']
        translation = self.stereo_calibration['T']

        image_size = left_image.shape[:2][::-1]

        # Stereo rectification
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            left_camera_matrix, left_dist_coeffs,
            right_camera_matrix, right_dist_coeffs,
            image_size, rotation, translation, alpha=self.rectification_alpha
        )

        # Compute the undistortion and rectification maps
        left_map1, left_map2 = cv2.initUndistortRectifyMap(
            left_camera_matrix, left_dist_coeffs, R1, P1, image_size, cv2.CV_16SC2
        )
        right_map1, right_map2 = cv2.initUndistortRectifyMap(
            right_camera_matrix, right_dist_coeffs, R2, P2, image_size, cv2.CV_16SC2
        )

        # Apply the rectification maps to the images
        left_rectified = cv2.remap(left_image, left_map1, left_map2, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_image, right_map1, right_map2, cv2.INTER_LINEAR)

        return left_rectified, right_rectified

    def capture_images(self, camera=0, name="left", **kwargs):
        """Capture images from a single camera."""
        from camera_capture import capture_images
        capture_images(self.marker_detector, camera, name, self.markers_required, **kwargs)
    
    def capture_stereo_images(self, **kwargs):
        """Capture stereo image pairs."""
        from camera_capture import capture_stereo_images
        capture_stereo_images(self.marker_detector, self.cameras["left"], self.cameras["right"], 
                            self.image_folders["stereo"], self.markers_required, **kwargs)
    
    def preview_stereo(self):
        """Preview stereo camera feed with rectification."""
        from camera_capture import preview_stereo
        fisheye_calibrations = {
            "left": self.left_fisheye_calibration,
            "right": self.right_fisheye_calibration
        }
        preview_stereo(self.marker_detector, fisheye_calibrations, self.cameras, self.rectification_alpha)
