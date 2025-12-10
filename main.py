#!/usr/bin/env python3

"""
Main entry point for camera calibration utilities.
Imports and uses the modular calibration components.
"""

import argparse
import configparser
import os


def create_sample_config(output_path='config.ini'):
    """
    Create a sample configuration file with default settings.
    
    Args:
        output_path: Path where the config file should be created
    """
    config = configparser.ConfigParser()
    
    config['cameras'] = {
        '# Camera device indices': '',
        'left': '1',
        'right': '0'
    }
    
    config['capture'] = {
        '# Camera capture settings': '',
        'exposure_time': '250000',
        'analogue_gain': '1.0',
        'capture_interval': '2',
        'image_size_width': '1920',
        'image_size_height': '1080',
        '# Camera transformations': '',
        'hflip': '1',
        'vflip': '1'
    }
    
    config['aruco'] = {
        '# ArUco marker detection settings': '',
        'dictionary': 'DICT_6X6_100',
        'marker_dimensions_width': '8',
        'marker_dimensions_height': '5',
        'square_length': '0.05',
        'marker_length': '0.037',
        'marker_colour_b': '0',
        'marker_colour_g': '0',
        'marker_colour_r': '255',
        '# Marker detection parameters': '',
        'corner_refinement': 'CORNER_REFINE_SUBPIX',
        'markers_required_ratio': '0.9'
    }
    
    config['calibration'] = {
        '# Calibration settings': '',
        'rectification_alpha': '0.95',
        '# Fisheye calibration': '',
        'fisheye': 'true',
        'fisheye_balance': '1.0',
        '# Stereo calibration criteria': '',
        'stereo_max_iter': '100',
        'stereo_epsilon': '1e-5',
        '# Stereo calibration flags (for divergent cameras with minimal overlap)': '',
        'stereo_fix_intrinsic': 'true',
        'stereo_fix_focal_length': 'true',
        'stereo_fix_principal_point': 'true',
        '# Output format': '',
        'output_format': 'npz',
        '# Performance': '',
        'max_images': '30'
    }
    
    config['paths'] = {
        '# Image folder paths': '',
        'left_images': 'images/left',
        'right_images': 'images/right',
        'stereo_images': 'images/stereo',
        'saves_folder': 'images/saves',
        'calibration_output': 'calibration',
        '# Calibration output files': '',
        'left_calibration': 'left.npz',
        'right_calibration': 'right.npz',
        'left_fisheye_calibration': 'left_fisheye.npz',
        'right_fisheye_calibration': 'right_fisheye.npz',
        'stereo_calibration': 'stereo_calibration.npz'
    }
    
    config['preview'] = {
        '# Preview settings': '',
        'left_margin': '965',
        'right_margin': '931',
        'preview_scale': '3'
    }
    
    with open(output_path, 'w') as configfile:
        config.write(configfile)
    
    print(f"Sample configuration file created: {output_path}")


def create_folder_structure(config_path='config.ini'):
    """
    Create the folder structure based on config file settings.
    
    Args:
        config_path: Path to the configuration file
    """
    config = configparser.ConfigParser()
    config.read(config_path)
    
    folders = [
        config.get('paths', 'left_images'),
        config.get('paths', 'right_images'),
        config.get('paths', 'stereo_images'),
        config.get('paths', 'saves_folder'),
        config.get('paths', 'calibration_output')
    ]
    
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")
        else:
            print(f"Folder already exists: {folder}")
    
    print("Folder structure setup complete!")


def load_config(config_path='config.ini'):
    """
    Load configuration from INI file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        ConfigParser object with loaded settings
    """
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


def create_calibration_from_config(config):
    """
    Create a Calibration instance with settings from config file.
    
    Args:
        config: ConfigParser object with configuration
        
    Returns:
        Configured Calibration instance
    """
    # Create calibration instance with marker dimensions from config
    marker_dims = (
        config.getint('aruco', 'marker_dimensions_width'),
        config.getint('aruco', 'marker_dimensions_height')
    )
    
    calibration = Calibration(
        marker_dimensions=marker_dims,
        square_length=config.getfloat('aruco', 'square_length'),
        marker_length=config.getfloat('aruco', 'marker_length')
    )
    
    # Set camera indices
    calibration.cameras['left'] = config.getint('cameras', 'left')
    calibration.cameras['right'] = config.getint('cameras', 'right')
    
    # Set image folders
    calibration.image_folders['left'] = config.get('paths', 'left_images')
    calibration.image_folders['right'] = config.get('paths', 'right_images')
    calibration.image_folders['stereo'] = config.get('paths', 'stereo_images')
    
    # Set rectification alpha
    calibration.rectification_alpha = config.getfloat('calibration', 'rectification_alpha')
    
    # Set marker color
    calibration.marker_detector.marker_colour = (
        config.getint('aruco', 'marker_colour_b'),
        config.getint('aruco', 'marker_colour_g'),
        config.getint('aruco', 'marker_colour_r')
    )
    
    return calibration


def get_capture_settings(config):
    """
    Extract capture settings from config.
    
    Args:
        config: ConfigParser object with configuration
        
    Returns:
        Dictionary with capture settings
    """
    return {
        'exposure_time': config.getint('capture', 'exposure_time'),
        'analogue_gain': config.getfloat('capture', 'analogue_gain'),
        'capture_interval': config.getint('capture', 'capture_interval'),
        'image_size': (
            config.getint('capture', 'image_size_width'),
            config.getint('capture', 'image_size_height')
        ),
        'hflip': config.getint('capture', 'hflip'),
        'vflip': config.getint('capture', 'vflip')
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Camera Calibration Utility')
    parser.add_argument('-c', '--config', type=str, default='config.ini',
                        help='Path to configuration file (default: config.ini)')
    parser.add_argument('-a', '--action', type=str,
                        choices=['capture-left', 'capture-right', 'capture-stereo', 
                                'calibrate-left', 'calibrate-right', 'stereo-calibrate', 
                                'preview', 'create-config', 'setup-folders', 'convert-calibration',
                                'visualize-overlap'],
                        help='Action to perform')
    parser.add_argument('-f', '--fisheye', action='store_true',
                        help='Use fisheye calibration model')
    parser.add_argument('--output-format', type=str, choices=['npz', 'yaml', 'both'],
                        help='Calibration output format (overrides config file)')
    parser.add_argument('--input-file', type=str,
                        help='Input calibration file for conversion (.npz or .yaml)')
    parser.add_argument('--left-image', type=str,
                        help='Left camera image for overlap visualization')
    parser.add_argument('--right-image', type=str,
                        help='Right camera image for overlap visualization')
    parser.add_argument('--stereo-calib', type=str,
                        help='Stereo calibration file for overlap visualization')
    parser.add_argument('--max-images', type=int,
                        help='Maximum number of images to use for calibration (default: 30)')
    
    args = parser.parse_args()
    
    # Handle utility actions that don't require full setup
    if args.action == 'create-config':
        create_sample_config(args.config)
        exit(0)
    
    if args.action == 'convert-calibration':
        if not args.input_file:
            parser.error('--input-file is required for convert-calibration action')
        if not args.output_format:
            parser.error('--output-format is required for convert-calibration action')
        if not os.path.exists(args.input_file):
            print(f"Error: Input file not found: {args.input_file}")
            exit(1)
        
        # Import calibration module for conversion
        from calibration import Calibration
        calib = Calibration(marker_dimensions=(8, 5), square_length=0.05, marker_length=0.037)
        calib.convert_calibration(args.input_file, args.output_format)
        exit(0)
    
    if args.action == 'setup-folders':
        if not os.path.exists(args.config):
            print(f"Config file not found: {args.config}")
            print("Create one first with: python main.py --action create-config")
            exit(1)
        create_folder_structure(args.config)
        exit(0)
    
    # Require action for calibration operations
    if args.action is None:
        parser.error("--action is required")
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        print("Create one first with: python main.py --action create-config")
        exit(1)
    
    # Import calibration module only when needed (requires picamera2)
    from calibration import Calibration
    
    config = load_config(args.config)
    
    # Create calibration instance
    calibrate = Calibration.from_config(config)
    calibrate.load_calibrations()
    
    # Get capture settings
    capture_settings = get_capture_settings(config)
    
    # Determine output format
    if args.output_format:
        save_format = args.output_format
    else:
        save_format = config.get('calibration', 'output_format', fallback='npz')
    
    # Determine max images
    if args.max_images:
        max_images = args.max_images
    else:
        max_images = config.getint('calibration', 'max_images', fallback=30)
    
    # Determine fisheye mode (CLI flag overrides config)
    if args.fisheye:
        use_fisheye = True
    else:
        use_fisheye = config.getboolean('calibration', 'fisheye', fallback=False)
    
    # Execute requested action
    if args.action == 'capture-left':
        calibrate.capture_images(
            calibrate.cameras['left'], 
            'left',
            **capture_settings
        )
    elif args.action == 'capture-right':
        calibrate.capture_images(
            calibrate.cameras['right'], 
            'right',
            **capture_settings
        )
    elif args.action == 'capture-stereo':
        calibrate.capture_stereo_images(**capture_settings)
    elif args.action == 'calibrate-left':
        calibrate.calibrate('left', fisheye=use_fisheye, save_format=save_format, max_images=max_images)
    elif args.action == 'calibrate-right':
        calibrate.calibrate('right', fisheye=use_fisheye, save_format=save_format, max_images=max_images)
    elif args.action == 'stereo-calibrate':
        calibrate.stereo_calibrate(save_format=save_format)
    elif args.action == 'visualize-overlap':
        if not args.left_image or not args.right_image or not args.stereo_calib:
            print("Error: --left-image, --right-image, and --stereo-calib are required for visualize-overlap")
            exit(1)
        calibrate.visualize_stereo_overlap(args.left_image, args.right_image, args.stereo_calib)
    elif args.action == 'preview':
        calibrate.preview_stereo()
