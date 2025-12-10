Neave Engineering Camera Calibration Utilities
==

A modular camera calibration system for stereo camera setups using ArUco markers and PiCamera2.

This is primarily for calibrating cameras on Raspberry Pi based devices, it should work for others for pure calibration but the capture parts will assume you've a Pi camera attached.

## Project Structure

```
├── main.py                    # Entry point with CLI and config support
├── calibration.py             # Core calibration logic
├── camera_capture.py          # PiCamera2 capture functions
├── image_annotation.py        # ArUco marker detection
├── config.ini                 # Configuration file
└── README.md                  # This file
```

## Installation

If your device will be single purpose and you don't need to worry about dependency hell, have a look at Jeff Geerling's [post](https://www.jeffgeerling.com/blog/2023/how-solve-error-externally-managed-environment-when-installing-pip3) to disable the library safety mechanism that's enabled by default.

Create a venv:
```bash
python3 -m venv env
source env/bin/activate
```

And follow [this guide](https://forums.raspberrypi.com/viewtopic.php?t=361758) to using PiCamera2 in a venv.

## Configuration

All settings can be configured via `config.ini`:

### Camera Settings
- Camera device indices (left/right)
- Exposure time and gain
- Image size and transformations

### ArUco Marker Settings
- Marker dictionary type
- Board dimensions
- Square and marker lengths
- Detection parameters

### Calibration Settings
- Rectification alpha
- Fisheye balance
- Output file paths

### Preview Settings
- Margins and scale factors

## Usage

### Setup

Before running calibration operations, set up your environment:

```bash
# Create a sample configuration file
python main.py --action create-config

# Create the folder structure for storing images
python main.py --action setup-folders
```

You can also create a custom config file:

```bash
python main.py --config my_config.ini --action create-config
python main.py --config my_config.ini --action setup-folders
```

### Using Command Line Interface

The calibration pipeline can be run using different actions:

```bash
# Capture images from left camera
python main.py --action capture-left

# Capture images from right camera
python main.py --action capture-right

# Capture stereo image pairs
python main.py --action capture-stereo

# Calibrate left camera
python main.py --action calibrate-left

# Calibrate left camera with fisheye model
python main.py --action calibrate-left --fisheye

# Calibrate right camera
python main.py --action calibrate-right

# Perform stereo calibration
python main.py --action stereo-calibrate

# Visualize overlap region for divergent cameras
python main.py --action visualize-overlap \
    --left-image images/stereo/stereo_left_01.jpg \
    --right-image images/stereo/stereo_right_01.jpg \
    --stereo-calib calibration/stereo_calibration.yaml

# Preview stereo feed
python main.py --action preview
```

### Divergent Stereo Cameras

If you're using cameras with divergent (non-parallel) optical axes, see the **[Divergent Stereo Camera Guide](DIVERGENT_STEREO.md)** for:
- Computing overlap masks showing the overlapping field of view
- Analyzing vergence angle and baseline distance
- Optimizing stereo rectification for divergent configurations
- Visualizing overlap regions on actual images

### Using Custom Configuration

Specify a different config file:

```bash
python main.py --config my_config.ini --action capture-stereo
```

### Programmatic Usage

```python
from calibration import Calibration
import configparser

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

# Create calibration instance
calibrate = Calibration.from_config(config)

# Load existing calibrations
calibrate.load_calibrations()

# Perform operations
calibrate.capture_stereo_images()
calibrate.stereo_calibrate()
```

## Calibration Workflow

1. **Setup** - Initialize your environment:
   ```bash
   python main.py --action create-config
   python main.py --action setup-folders
   ```
2. **Configure** - Edit `config.ini` with your camera and marker settings
3. **Capture** - Collect calibration images:
   - Individual cameras: `capture-left`, `capture-right`
   - Stereo pairs: `capture-stereo`
4. **Calibrate** - Process images:
   - Individual: `calibrate-left`, `calibrate-right`
   - Stereo: `stereo-calibrate`
5. **Verify** - Preview results with `preview`

## Output Files

Calibration data is saved in your chosen format (NPZ, YAML, or both):
- `left.npz/.yaml` - Left camera calibration
- `right.npz/.yaml` - Right camera calibration
- `left_fisheye.npz/.yaml` - Left camera fisheye calibration
- `right_fisheye.npz/.yaml` - Right camera fisheye calibration
- `stereo_calibration.npz/.yaml` - Stereo calibration data with extrinsics

### Stereo Calibration Output

The stereo calibration file includes:
- **Intrinsics**: Camera matrices and distortion coefficients for both cameras
- **Extrinsics**: Rotation (R) and translation (T) between cameras
- **Rectification**: Maps for aligning image planes (R1, R2, P1, P2, Q)
- **Overlap Masks**: Binary masks showing overlapping field of view
- **Metadata**: Vergence angle, baseline distance, overlap statistics

For divergent cameras, additional masks are saved:
- `calibration/overlap_mask_left.png` - Left camera overlap region
- `calibration/overlap_mask_right.png` - Right camera overlap region
- `calibration/stereo_overlap_visualization.png` - Visual overlay (when using visualize-overlap) 


