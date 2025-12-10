#!/usr/bin/env python3

import cv2
import numpy as np


class MarkerDetector:
    """Class for ArUco marker detection and annotation."""
    
    def __init__(self, marker_dimensions=(8, 5)):
        """
        Initialize the ArUco marker detector.
        
        Args:
            marker_dimensions: Tuple of (width, height) for the Charuco board
        """
        self.marker_dimensions = marker_dimensions
        self.marker_colour = (0, 0, 255)  # GBR so red
        self.init_aruco_board()
        
    def init_aruco_board(self):
        """Initialize ArUco dictionary, detector parameters, and Charuco board."""
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
        self.parameters = cv2.aruco.DetectorParameters()
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        self.board = cv2.aruco.CharucoBoard(self.marker_dimensions, 0.05, 0.037, self.dictionary)

        self.markers_total = self.marker_dimensions[0] * self.marker_dimensions[1] // 2 
        self.markers_required = self.markers_total * 0.9
        print(f"markers total: {self.markers_total}, markers required: {self.markers_required}")
    
    def create_board(self, square_length, marker_length):
        """
        Create a new Charuco board with custom dimensions.
        
        Args:
            square_length: Length of the square side in meters
            marker_length: Length of the marker side in meters
            
        Returns:
            CharucoBoard object
        """
        return cv2.aruco.CharucoBoard(self.marker_dimensions, square_length, marker_length, self.dictionary)

    def detect_markers(self, frame):
        """
        Detect ArUco markers in a frame.
        
        Args:
            frame: Input image frame (RGB format)
            
        Returns:
            Tuple of (corners, ids, rejectedImgPoints)
        """
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Detect Aruco markers in the image
        corners, ids, rejectedImgPoints = self.detector.detectMarkers(gray)
        return corners, ids, rejectedImgPoints

    def show_markers(self, frame, corners, ids, scale):
        """
        Annotate frame with detected markers.
        
        Args:
            frame: Input image frame
            corners: Detected marker corners
            ids: Detected marker IDs
            scale: Scale factor for resizing the output frame
            
        Returns:
            Annotated frame with markers drawn
        """
        if corners is not None and len(corners) > self.markers_required:
            # Draw the detected markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor=self.marker_colour)
            
        # Display the frame
        frame = cv2.resize(frame, (frame.shape[1] // scale, frame.shape[0] // scale))
        return frame

    def init_fisheye_maps(self, img, balance, calibration):
        """
        Initialize fisheye undistortion maps.
        
        Args:
            img: Input image to get dimensions from
            balance: Balance parameter for fisheye calibration
            calibration: Calibration data containing K and D matrices
            
        Returns:
            Tuple of (map1, map2) for cv2.remap
        """
        img_dim = img.shape[:2][::-1]  
        DIM = (1920, 1080)
        
        scaled_K = calibration["K"] * img_dim[0] / DIM[0]  
        scaled_K[2][2] = 1.0  
        
        D = calibration["D"]
        
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D,
            img_dim, np.eye(3), balance=balance)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3),
            new_K, img_dim, cv2.CV_16SC2)

        return map1, map2
