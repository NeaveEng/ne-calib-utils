#!/usr/bin/env python3

"""
Utility script to batch rename images for use with calibration tools.
Renames numbered images to the expected format for capture sessions.
"""

import argparse
import os
import re
import shutil
from pathlib import Path
from datetime import datetime


def get_image_timestamp(filepath):
    """
    Get timestamp from image file's creation time.
    Uses the earlier of creation time or modification time for cross-platform compatibility.
    
    Args:
        filepath: Path to the image file
        
    Returns:
        Integer timestamp (seconds since epoch)
    """
    stat = os.stat(filepath)
    # On Windows, st_ctime is creation time
    # On Unix, st_ctime is last metadata change time
    # Use st_birthtime if available (macOS), otherwise use the earlier of ctime/mtime
    if hasattr(stat, 'st_birthtime'):
        # macOS
        return int(stat.st_birthtime)
    else:
        # Windows uses st_ctime for creation, Unix uses it for metadata changes
        # To be safe, use the earlier of the two
        return int(min(stat.st_ctime, stat.st_mtime))


def get_numbered_images(directory, pattern=None):
    """
    Get all numbered image files from a directory.
    
    Args:
        directory: Path to the directory containing images
        pattern: Optional regex pattern to match filenames (default: looks for numbered files)
        
    Returns:
        List of (filepath, number) tuples sorted by number
    """
    if pattern is None:
        # Default pattern matches files like: 0.png, 1.jpg, image_0.png, IMG_001.jpg, etc.
        pattern = r'.*?(\d+)\.(png|jpg|jpeg|bmp)$'
    
    image_files = []
    for file in os.listdir(directory):
        match = re.match(pattern, file, re.IGNORECASE)
        if match:
            number = int(match.group(1))
            filepath = os.path.join(directory, file)
            image_files.append((filepath, number, file))
    
    # Sort by number
    image_files.sort(key=lambda x: x[1])
    return image_files


def rename_for_single_camera(directory, camera_name, dry_run=False, move_to=None):
    """
    Rename images for single camera calibration using file timestamps.
    
    Args:
        directory: Path to the directory containing images
        camera_name: Name of the camera ('left' or 'right')
        dry_run: If True, only show what would be renamed without making changes
        move_to: Optional destination directory to move renamed files to
    """
    images = get_numbered_images(directory)
    
    if not images:
        print(f"No numbered images found in {directory}")
        return
    
    print(f"Found {len(images)} images to rename")
    
    # Create destination directory if needed
    if move_to and not dry_run and not os.path.exists(move_to):
        os.makedirs(move_to)
        print(f"Created directory: {move_to}")
    
    for filepath, number, original_name in images:
        timestamp = get_image_timestamp(filepath)
        extension = Path(filepath).suffix
        new_name = f"captured_{camera_name}_frame_{timestamp}{extension}"
        
        # Determine destination path
        if move_to:
            new_path = os.path.join(move_to, new_name)
        else:
            new_path = os.path.join(directory, new_name)
        
        if dry_run:
            file_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            action = "Move" if move_to else "Rename"
            dest = new_path if move_to else new_name
            print(f"  {action}: {original_name} -> {dest} (file time: {file_time})")
        else:
            shutil.move(filepath, new_path)
            action = "Moved" if move_to else "Renamed"
            print(f"{action}: {original_name} -> {new_name}")
    
    if dry_run:
        print("\nDry run complete. Run without --dry-run to apply changes.")


def rename_for_stereo(left_dir, right_dir, dry_run=False, move_to=None):
    """
    Rename images for stereo calibration (matching left/right pairs) using file timestamps.
    
    Args:
        left_dir: Path to the directory containing left camera images
        right_dir: Path to the directory containing right camera images
        dry_run: If True, only show what would be renamed without making changes
        move_to: Optional destination directory to move renamed files to (both left and right)
    """
    left_images = get_numbered_images(left_dir)
    right_images = get_numbered_images(right_dir)
    
    if not left_images or not right_images:
        print(f"Images not found in both directories")
        return
    
    if len(left_images) != len(right_images):
        print(f"Warning: Different number of images in left ({len(left_images)}) and right ({len(right_images)}) directories")
        print(f"Will process {min(len(left_images), len(right_images))} pairs")
    
    num_pairs = min(len(left_images), len(right_images))
    print(f"Found {num_pairs} image pairs to rename")
    
    # Create destination directory if needed
    if move_to and not dry_run and not os.path.exists(move_to):
        os.makedirs(move_to)
        print(f"Created directory: {move_to}")
    
    for i in range(num_pairs):
        left_filepath, left_num, left_original = left_images[i]
        right_filepath, right_num, right_original = right_images[i]
        
        # Use the left camera's timestamp for the pair
        timestamp = get_image_timestamp(left_filepath)
        
        left_ext = Path(left_filepath).suffix
        right_ext = Path(right_filepath).suffix
        
        new_left_name = f"captured_left_frame_{timestamp}{left_ext}"
        new_right_name = f"captured_right_frame_{timestamp}{right_ext}"
        
        # Determine destination paths
        if move_to:
            new_left_path = os.path.join(move_to, new_left_name)
            new_right_path = os.path.join(move_to, new_right_name)
        else:
            new_left_path = os.path.join(left_dir, new_left_name)
            new_right_path = os.path.join(right_dir, new_right_name)
        
        if dry_run:
            file_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            action = "Move" if move_to else "Rename"
            print(f"  Pair {i+1} (file time: {file_time}):")
            if move_to:
                print(f"    Left:  {action} {left_original} -> {new_left_path}")
                print(f"    Right: {action} {right_original} -> {new_right_path}")
            else:
                print(f"    Left:  {left_original} -> {new_left_name}")
                print(f"    Right: {right_original} -> {new_right_name}")
        else:
            shutil.move(left_filepath, new_left_path)
            shutil.move(right_filepath, new_right_path)
            action = "Moved" if move_to else "Renamed"
            print(f"{action} pair {i+1}:")
            print(f"  Left:  {left_original} -> {new_left_name}")
            print(f"  Right: {right_original} -> {new_right_name}")
    
    if dry_run:
        print("\nDry run complete. Run without --dry-run to apply changes.")


def copy_and_rename(source_dir, dest_dir, camera_name=None, dry_run=False):
    """
    Copy images from source to destination and rename them using file timestamps.
    
    Args:
        source_dir: Path to source directory
        dest_dir: Path to destination directory
        camera_name: Name of the camera ('left' or 'right'), if None uses 'left'
        dry_run: If True, only show what would be done
    """
    if camera_name is None:
        camera_name = 'left'
    
    images = get_numbered_images(source_dir)
    
    if not images:
        print(f"No numbered images found in {source_dir}")
        return
    
    if not os.path.exists(dest_dir) and not dry_run:
        os.makedirs(dest_dir)
        print(f"Created directory: {dest_dir}")
    
    print(f"Found {len(images)} images to copy and rename")
    
    for filepath, number, original_name in images:
        timestamp = get_image_timestamp(filepath)
        extension = Path(filepath).suffix
        new_name = f"captured_{camera_name}_frame_{timestamp}{extension}"
        new_path = os.path.join(dest_dir, new_name)
        
        if dry_run:
            file_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  {original_name} -> {new_name} (file time: {file_time})")
        else:
            shutil.copy2(filepath, new_path)
            print(f"Copied: {original_name} -> {new_name}")
    
    if dry_run:
        print("\nDry run complete. Run without --dry-run to apply changes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Batch rename images for calibration tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Rename images in-place for single camera
  python rename_images.py --mode single --dir images/left --camera left

  # Rename stereo pairs (requires matching numbered files in both dirs)
  python rename_images.py --mode stereo --left-dir source/left --right-dir source/right

  # Copy and rename from source to destination
  python rename_images.py --mode copy --source source/images --dest images/left --camera left

  # Dry run to preview changes
  python rename_images.py --mode single --dir images/left --camera left --dry-run
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                        choices=['single', 'stereo', 'copy'],
                        help='Renaming mode: single camera, stereo pairs, or copy-and-rename')
    parser.add_argument('--dir', type=str,
                        help='Directory containing images (for single mode)')
    parser.add_argument('--left-dir', type=str,
                        help='Directory containing left camera images (for stereo mode)')
    parser.add_argument('--right-dir', type=str,
                        help='Directory containing right camera images (for stereo mode)')
    parser.add_argument('--source', type=str,
                        help='Source directory (for copy mode)')
    parser.add_argument('--dest', type=str,
                        help='Destination directory (for copy mode)')
    parser.add_argument('--camera', type=str, default='left',
                        choices=['left', 'right'],
                        help='Camera name for single/copy modes (default: left)')
    parser.add_argument('--move-to', type=str,
                        help='Destination directory to move renamed files to (creates if needed)')
    parser.add_argument('--pattern', type=str,
                        help='Custom regex pattern to match image filenames (default: auto-detect numbered files)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview changes without actually renaming files')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.dir:
            parser.error("--dir is required for single mode")
        if not os.path.exists(args.dir):
            print(f"Error: Directory not found: {args.dir}")
            exit(1)
        rename_for_single_camera(args.dir, args.camera, args.dry_run, args.move_to)
    
    elif args.mode == 'stereo':
        if not args.left_dir or not args.right_dir:
            parser.error("--left-dir and --right-dir are required for stereo mode")
        if not os.path.exists(args.left_dir):
            print(f"Error: Left directory not found: {args.left_dir}")
            exit(1)
        if not os.path.exists(args.right_dir):
            print(f"Error: Right directory not found: {args.right_dir}")
            exit(1)
        rename_for_stereo(args.left_dir, args.right_dir, args.dry_run, args.move_to)
    
    elif args.mode == 'copy':
        if not args.source or not args.dest:
            parser.error("--source and --dest are required for copy mode")
        if not os.path.exists(args.source):
            print(f"Error: Source directory not found: {args.source}")
            exit(1)
        copy_and_rename(args.source, args.dest, args.camera, args.dry_run)
