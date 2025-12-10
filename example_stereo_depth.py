#!/usr/bin/env python3
"""
Example script demonstrating depth computation for divergent stereo cameras.

This script shows how to:
1. Load stereo calibration with overlap masks
2. Rectify stereo images
3. Compute disparity only in overlap region
4. Convert disparity to 3D point cloud
5. Visualize depth map with overlap mask applied
"""

import cv2
import numpy as np
import argparse


def load_stereo_calibration(calib_file):
    """Load stereo calibration from YAML or NPZ file."""
    if calib_file.endswith('.yaml') or calib_file.endswith('.yml'):
        fs = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)
        data = {}
        root = fs.root()
        for i in range(root.size()):
            node = root.at(i)
            key = node.name()
            if node.isMat():
                data[key] = node.mat()
        fs.release()
    elif calib_file.endswith('.npz'):
        data = dict(np.load(calib_file))
    else:
        raise ValueError("Calibration file must be .yaml or .npz")
    
    return data


def compute_depth_map(left_img_path, right_img_path, calib_file, 
                      num_disparities=16*5, block_size=15):
    """
    Compute depth map for divergent stereo cameras using overlap mask.
    
    Args:
        left_img_path: Path to left camera image
        right_img_path: Path to right camera image
        calib_file: Path to stereo calibration file
        num_disparities: Number of disparities (must be divisible by 16)
        block_size: Block size for matching (must be odd)
    
    Returns:
        disparity: Disparity map
        depth: Depth map in meters
        points_3d: 3D point cloud
        overlap_mask: Mask of valid overlap region
    """
    # Load calibration
    print("Loading calibration...")
    calib = load_stereo_calibration(calib_file)
    
    # Load images
    print("Loading images...")
    left_img = cv2.imread(left_img_path)
    right_img = cv2.imread(right_img_path)
    
    if left_img is None or right_img is None:
        raise ValueError("Could not load images")
    
    # Get calibration parameters
    map1_left = calib['map1_left']
    map2_left = calib['map2_left']
    map1_right = calib['map1_right']
    map2_right = calib['map2_right']
    Q = calib['Q']
    
    # Use marker-based overlap mask if available (most accurate), 
    # otherwise fall back to rectified, then original
    if 'overlap_mask_left_markers' in calib:
        overlap_mask_left = calib['overlap_mask_left_markers']
        print("Using marker-based overlap mask (most accurate)")
    elif 'overlap_mask_left_rect' in calib:
        overlap_mask_left = calib['overlap_mask_left_rect']
        print("Using rectified overlap mask")
    else:
        overlap_mask_left = calib['overlap_mask_left']
        print("Using original overlap mask")
    
    # Rectify images
    print("Rectifying images...")
    left_rectified = cv2.remap(left_img, map1_left, map2_left, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_img, map1_right, map2_right, cv2.INTER_LINEAR)
    
    # Convert to grayscale for stereo matching
    left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
    
    # Create stereo matcher
    print("Computing disparity map...")
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    # Compute disparity
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    
    # Apply overlap mask - set invalid regions to 0
    disparity_masked = disparity.copy()
    disparity_masked[overlap_mask_left == 0] = 0
    
    # Reproject to 3D
    print("Reprojecting to 3D...")
    points_3d = cv2.reprojectImageTo3D(disparity_masked, Q)
    
    # Extract depth (Z coordinate)
    depth = points_3d[:, :, 2]
    
    # Filter invalid depths
    depth[depth <= 0] = 0
    depth[depth > 10] = 0  # Filter out points beyond 10 meters
    
    return disparity_masked, depth, points_3d, overlap_mask_left


def visualize_depth(left_img_path, disparity, depth, overlap_mask):
    """Visualize disparity and depth maps."""
    # Load original image for overlay
    left_img = cv2.imread(left_img_path)
    
    # Normalize disparity for visualization
    disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    disp_vis[overlap_mask == 0] = 0  # Black out non-overlap regions
    
    # Normalize depth for visualization
    depth_vis = depth.copy()
    depth_vis[depth_vis == 0] = np.nan  # Set invalid to NaN
    valid_depth = depth_vis[~np.isnan(depth_vis)]
    if len(valid_depth) > 0:
        vmin, vmax = np.percentile(valid_depth, [5, 95])
        depth_vis = np.clip(depth_vis, vmin, vmax)
        depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
        depth_vis[overlap_mask == 0] = 0  # Black out non-overlap regions
    else:
        depth_vis = np.zeros_like(left_img)
    
    # Create overlay on original image
    overlay = cv2.addWeighted(left_img, 0.5, depth_vis, 0.5, 0)
    
    # Stack visualizations
    top_row = np.hstack([left_img, disp_vis])
    bottom_row = np.hstack([depth_vis, overlay])
    combined = np.vstack([top_row, bottom_row])
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(combined, 'Disparity (overlap only)', (left_img.shape[1] + 10, 30), 
                font, 1, (255, 255, 255), 2)
    cv2.putText(combined, 'Depth (m)', (10, left_img.shape[0] + 30), 
                font, 1, (255, 255, 255), 2)
    cv2.putText(combined, 'Depth Overlay', (left_img.shape[1] + 10, left_img.shape[0] + 30), 
                font, 1, (255, 255, 255), 2)
    
    # Display
    cv2.imshow('Depth Visualization (Black = outside overlap region)', combined)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save
    output_path = 'calibration/depth_visualization.png'
    cv2.imwrite(output_path, combined)
    print(f"Saved visualization to {output_path}")
    
    return combined


def print_statistics(depth, overlap_mask):
    """Print depth statistics."""
    valid_depth = depth[overlap_mask > 0]
    valid_depth = valid_depth[valid_depth > 0]
    
    if len(valid_depth) > 0:
        print("\nDepth Statistics (overlap region only):")
        print(f"  Valid pixels: {len(valid_depth)} / {np.sum(overlap_mask > 0)} "
              f"({100 * len(valid_depth) / np.sum(overlap_mask > 0):.1f}%)")
        print(f"  Min depth: {np.min(valid_depth):.3f} m")
        print(f"  Max depth: {np.max(valid_depth):.3f} m")
        print(f"  Mean depth: {np.mean(valid_depth):.3f} m")
        print(f"  Median depth: {np.median(valid_depth):.3f} m")
        print(f"  Std dev: {np.std(valid_depth):.3f} m")
    else:
        print("\nNo valid depth measurements found!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Compute and visualize depth map for divergent stereo cameras'
    )
    parser.add_argument('--left', type=str, required=True,
                        help='Path to left camera image')
    parser.add_argument('--right', type=str, required=True,
                        help='Path to right camera image')
    parser.add_argument('--calib', type=str, required=True,
                        help='Path to stereo calibration file (.yaml or .npz)')
    parser.add_argument('--num-disparities', type=int, default=80,
                        help='Number of disparities (must be divisible by 16)')
    parser.add_argument('--block-size', type=int, default=15,
                        help='Block size for matching (must be odd)')
    
    args = parser.parse_args()
    
    # Validate parameters
    if args.num_disparities % 16 != 0:
        print("Error: num_disparities must be divisible by 16")
        exit(1)
    if args.block_size % 2 == 0:
        print("Error: block_size must be odd")
        exit(1)
    
    # Compute depth map
    disparity, depth, points_3d, overlap_mask = compute_depth_map(
        args.left, args.right, args.calib,
        args.num_disparities, args.block_size
    )
    
    # Print statistics
    print_statistics(depth, overlap_mask)
    
    # Visualize
    visualize_depth(args.left, disparity, depth, overlap_mask)
    
    print("\nDepth computation complete!")
    print(f"3D points shape: {points_3d.shape}")
    print(f"To extract point cloud: points = points_3d[overlap_mask > 0]")
