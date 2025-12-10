# Divergent Stereo Camera Calibration Guide

This guide explains how to calibrate and work with divergent stereo cameras (cameras with non-parallel optical axes).

## Overview

Divergent stereo cameras have optical axes that point outward at an angle, creating a wider combined field of view but with a reduced overlap region compared to parallel stereo cameras. This is useful for:
- Wide-area surveillance
- Panoramic stereo vision
- Reduced blind spots in robotics

## Calibration Workflow

### 1. Individual Camera Calibration

First, calibrate each camera individually:

```bash
# Calibrate left camera
python main.py -a calibrate-left --output-format yaml

# Calibrate right camera
python main.py -a calibrate-right --output-format yaml
```

### 2. Stereo Calibration

Capture stereo image pairs with the calibration board visible in **both** cameras:

```bash
# Capture stereo pairs
python main.py -a capture-stereo

# Perform stereo calibration
python main.py -a stereo-calibrate --output-format yaml
```

The stereo calibration will automatically compute:
- **Extrinsic parameters** (R, T): Rotation and translation between cameras
- **Rectification maps**: For aligning image planes
- **Overlap masks**: Binary masks showing overlapping field of view
- **Vergence angle**: Angle between optical axes
- **Baseline distance**: Physical distance between camera centers

### 3. Understanding the Output

After stereo calibration, you'll see output like:

```
Overlap Analysis:
  Left camera overlap: 45.3% of image
  Right camera overlap: 42.8% of image
  Overlap region: 856x720 pixels
  Baseline distance: 0.065m
  Vergence angle: 25.40°
```

**Key Metrics:**
- **Overlap %**: Percentage of each camera's FOV that overlaps with the other
- **Vergence angle**: 0° = parallel, higher = more divergent
- **Baseline**: Distance between camera centers (affects depth accuracy)

## Using the Overlap Masks

### Saved Mask Files

Overlap masks are automatically saved to `calibration/overlap_mask_*.png`:

- `overlap_mask_left.png`: White pixels = visible in both cameras from left camera's view
- `overlap_mask_right.png`: White pixels = visible in both cameras from right camera's view

### Visualizing Overlap on Real Images

To see the overlap region overlaid on actual images:

```bash
python main.py -a visualize-overlap \
    --left-image images/stereo/stereo_left_01.jpg \
    --right-image images/stereo/stereo_right_01.jpg \
    --stereo-calib calibration/stereo_calibration.yaml
```

This creates `calibration/stereo_overlap_visualization.png` showing both images side-by-side with the overlap region highlighted in green.

## Accessing Calibration Data in Code

### Loading Stereo Calibration

```python
import cv2
import numpy as np

# Load YAML calibration
fs = cv2.FileStorage('calibration/stereo_calibration.yaml', cv2.FILE_STORAGE_READ)
camera_matrix_left = fs.getNode('camera_matrix_left').mat()
camera_matrix_right = fs.getNode('camera_matrix_right').mat()
dist_coeffs_left = fs.getNode('dist_coeffs_left').mat()
dist_coeffs_right = fs.getNode('dist_coeffs_right').mat()
R = fs.getNode('R').mat()  # Rotation from left to right
T = fs.getNode('T').mat()  # Translation from left to right
overlap_mask_left = fs.getNode('overlap_mask_left').mat()
overlap_mask_right = fs.getNode('overlap_mask_right').mat()
map1_left = fs.getNode('map1_left').mat()
map2_left = fs.getNode('map2_left').mat()
map1_right = fs.getNode('map1_right').mat()
map2_right = fs.getNode('map2_right').mat()
fs.release()

# Or load NPZ
data = np.load('calibration/stereo_calibration.npz')
R = data['R']
T = data['T']
```

### Using Rectification Maps

```python
# Rectify stereo images for depth computation
left_img = cv2.imread('left.jpg')
right_img = cv2.imread('right.jpg')

# Apply rectification
left_rectified = cv2.remap(left_img, map1_left, map2_left, cv2.INTER_LINEAR)
right_rectified = cv2.remap(right_img, map1_right, map2_right, cv2.INTER_LINEAR)

# Now corresponding points are on the same horizontal scanline
```

### Masking to Overlap Region Only

```python
# Process only the overlapping region
left_overlap_only = cv2.bitwise_and(left_img, left_img, mask=overlap_mask_left)
right_overlap_only = cv2.bitwise_and(right_img, right_img, mask=overlap_mask_right)

# For stereo matching, restrict to overlap
stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)
disparity = stereo.compute(
    cv2.cvtColor(left_overlap_only, cv2.COLOR_BGR2GRAY),
    cv2.cvtColor(right_overlap_only, cv2.COLOR_BGR2GRAY)
)
```

## Optimizing for Divergent Cameras

### Adjusting Rectification Alpha

The `rectification_alpha` parameter (default 0.95) controls FOV vs black borders:
- `alpha=0`: Crop to valid pixels only (smaller FOV, no black borders)
- `alpha=1`: Keep all pixels (larger FOV, black borders in invalid regions)
- `alpha=0.95`: Recommended balance for divergent cameras

Edit in `config.ini`:
```ini
[calibration]
rectification_alpha = 0.95
```

### Capturing Better Stereo Pairs

For divergent cameras:
1. **Place board at varying distances** (0.5m - 2m for typical setups)
2. **Cover the overlap region thoroughly** - move the board around the overlapping FOV
3. **Capture at different angles** - tilt board in various orientations
4. **Aim for 20-30 good pairs** - more pairs = better calibration

### Computing Depth in Overlap Region

```python
# After rectification, compute disparity
disparity = stereo_matcher.compute(left_rectified, right_rectified)

# Convert to depth (only valid in overlap region)
# Q matrix converts disparity to 3D points
points_3d = cv2.reprojectImageTo3D(disparity, Q)

# Mask to overlap region
points_3d_masked = points_3d.copy()
points_3d_masked[overlap_mask_left == 0] = 0

# Now you have 3D coordinates in overlap region
```

## Extrinsic Parameters Explained

### Rotation Matrix (R)
3×3 matrix describing rotation from left to right camera:
```python
# Convert to axis-angle or Euler angles
rvec, _ = cv2.Rodrigues(R)
angle = np.linalg.norm(rvec)
axis = rvec / angle
print(f"Rotation: {np.degrees(angle):.2f}° around axis {axis.flatten()}")
```

### Translation Vector (T)
3×1 vector giving position of right camera relative to left:
```python
baseline = np.linalg.norm(T)  # Distance between cameras
direction = T / baseline       # Direction from left to right
print(f"Baseline: {baseline:.3f}m")
print(f"Direction: {direction.flatten()}")
```

## Troubleshooting

### Low Overlap Percentage (<30%)
- Cameras may be too divergent for stereo vision
- Consider adjusting camera angles physically
- Use wider baseline if possible

### High Reprojection Error
- Ensure board is visible clearly in both cameras
- Capture more images in the overlap region
- Check for motion blur or poor lighting

### Masks Look Wrong
- Verify individual camera calibrations are accurate
- Check that stereo pairs are truly synchronized
- Ensure R and T matrices are reasonable (baseline ~5-15cm typical)

## Advanced Usage

### Custom Overlap Computation
```python
from calibration import Calibration

calib = Calibration.from_config(config)
overlap_mask_left, overlap_mask_right, info = calib.compute_overlap_mask(
    K1=camera_matrix_left,
    D1=dist_coeffs_left,
    K2=camera_matrix_right,
    D2=dist_coeffs_right,
    R=R,
    T=T,
    image_size=(1920, 1080)
)

print(f"Vergence angle: {info['vergence_angle']:.2f}°")
print(f"Overlap width: {info['overlap_width']}px")
```

### Programmatic Visualization
```python
calib.visualize_stereo_overlap(
    'images/stereo/stereo_left_01.jpg',
    'images/stereo/stereo_right_01.jpg',
    'calibration/stereo_calibration.yaml'
)
```

## References

- [OpenCV Stereo Calibration](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga91018d80e2a93ade37539f01e6f07de5)
- [Stereo Rectification](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6)
- [Disparity Map Computation](https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html)
