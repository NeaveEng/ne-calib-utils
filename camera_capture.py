#!/usr/bin/env python3

import cv2
from picamera2 import Picamera2
import libcamera
import time


def P_controller(Kp: float = 0.05, setpoint: float = 0, measurement: float = 0, output_limits=(-10000, 10000)):
    """Proportional controller for camera synchronization."""
    e = setpoint - measurement
    P = Kp * e

    output_value = P

    # output and limit if output_limits set
    lower, upper = output_limits
    if (upper is not None) and (output_value > upper):
        return upper
    elif (lower is not None) and (output_value < lower):
        return lower
    return output_value


def capture_metadata(left_cam, right_cam, debug=False):
    """Capture metadata from both cameras and calculate synchronization parameters."""
    metadata_picam2a = left_cam.capture_metadata()
    metadata_picam2b = right_cam.capture_metadata()

    timestamp_picam2a = metadata_picam2a["SensorTimestamp"] / 1000  #  convert ns to µs because all other values are in µs
    timestamp_picam2b = metadata_picam2b["SensorTimestamp"] / 1000  #  convert ns to µs because all other values are in µs
    timestamp_delta = timestamp_picam2b - timestamp_picam2a

    controller_output_frameduration_delta = int(P_controller(0.05, 0, timestamp_delta, (-10000, 10000)))
    control_out_frameduration = int(metadata_picam2a["FrameDuration"] + controller_output_frameduration_delta)  # sync to a, so use that for ref

    if debug:
        print("Cam A: SensorTimestamp: ", timestamp_picam2a, " FrameDuration: ", metadata_picam2a["FrameDuration"])
        print("Cam B: SensorTimestamp: ", timestamp_picam2b, " FrameDuration: ", metadata_picam2b["FrameDuration"])
        print("SensorTimestampDelta: ", round(timestamp_delta / 1000, 1), "ms")
        print("FrameDurationDelta: ", controller_output_frameduration_delta, "new FrameDurationLimit: ", control_out_frameduration)

    with right_cam.controls as ctrl:
        # set new FrameDurationLimits based on P_controller output.
        ctrl.FrameDurationLimits = (control_out_frameduration, control_out_frameduration)
    return timestamp_picam2a, timestamp_picam2b, round(timestamp_delta / 1000, 1)


def capture_images(marker_detector, camera=0, name="left", markers_required=18, 
                  exposure_time=250000, analogue_gain=1.0, capture_interval=2,
                  image_size=(1920, 1080), hflip=1, vflip=1):
    """Capture images from a single camera with ArUco marker detection."""
    # Initialize the Picamera2
    picam = Picamera2(camera)

    cam_config = picam.create_video_configuration(
        main={
                "size": image_size
            },
        controls={
                "ExposureTime": exposure_time, 
                "AnalogueGain": analogue_gain,
                "AwbEnable": 0,
                "AeEnable": 0
                }
        )

    cam_config["transform"] = libcamera.Transform(hflip=hflip, vflip=vflip)
    picam.configure(cam_config)
    
    picam.start()
    print(f"Camera {name}/{camera} started")

    last_capture_time = time.time()
    capture_count = 0
        
    while (True):
        frame = picam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        og_frame = frame.copy()
                                    
        corners, ids, _ = marker_detector.detect_markers(frame)
        if(corners is not None and len(corners) > 0):
            print(f"Corners: {len(corners)}, IDs: {len(ids)}")
        
            if ids is not None and len(ids) >= markers_required:
                if(time.time() - last_capture_time > capture_interval):
                    last_capture_time = time.time()
                    filename = f"images/{name}/captured_left_frame_{int(last_capture_time)}.png"
                    cv2.imwrite(filename, og_frame)
                    capture_count += 1                                            
            
            og_frame = marker_detector.show_markers(og_frame, corners, ids, scale=1)    

        cv2.putText(og_frame, f"Captured: {capture_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)                
        cv2.imshow(f'Camera {name}', og_frame)
               

        keycode = cv2.waitKeyEx(1)
            
        # Break the loop on 'q' key press
        if keycode == 113:
            break
        
        if keycode == 32:
            timestamp = int(time.time())
            cv2.imwrite(f"images/saves/{name}_capture_{timestamp}.png", og_frame)
            print("Images captured")
            
    # Release the Picamera2 and close windows
    picam.stop()
    cv2.destroyAllWindows()


def capture_stereo_images(marker_detector, left=1, right=0, image_folder="images/stereo", 
                         markers_required=18, exposure_time=250000, analogue_gain=1.0, 
                         capture_interval=2, image_size=(1920, 1080), hflip=1, vflip=1):
    """Capture synchronized stereo images from two cameras."""
    # Initialize the Picamera2
    left_picam = Picamera2(left)
    right_picam = Picamera2(right)

    left_cam_config = left_picam.create_video_configuration(
        main={
                "size": image_size
            },
        controls={
                "ExposureTime": exposure_time, 
                "AnalogueGain": analogue_gain,
                "AwbEnable": 0,
                "AeEnable": 0
                }
        )

    left_cam_config["transform"] = libcamera.Transform(hflip=hflip, vflip=vflip)
    left_picam.configure(left_cam_config)
    
    right_cam_config = right_picam.create_video_configuration(
        main={
                "size": image_size
            },
        controls={
                "ExposureTime": exposure_time, 
                "AnalogueGain": analogue_gain,
                "AwbEnable": 0,
                "AeEnable": 0
                }
        )

    right_cam_config["transform"] = libcamera.Transform(hflip=hflip, vflip=vflip)
    right_picam.configure(right_cam_config)
    
    left_picam.start()
    right_picam.start()
    print(f"Cameras started")

    last_capture_time = time.time()
    capture_count = 0
        
    # TODO: Add frame sync for stereo capture
    while (True):
        left_frame = left_picam.capture_array()
        left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
        left_og_frame = left_frame.copy()
        
        right_frame = right_picam.capture_array()
        right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)
        right_og_frame = right_frame.copy()
                                                
        left_corners, left_ids, _ = marker_detector.detect_markers(left_frame)
        right_corners, right_ids, _ = marker_detector.detect_markers(right_frame)
        
        if(left_corners is not None and len(left_corners) > 0) and (right_corners is not None and len(right_corners) > 0):
            print(f"Left corners: {len(left_corners)}, IDs: {len(left_ids)}, Right corners: {len(right_corners)}, IDs: {len(right_ids)}")
        
            if (left_ids is not None and len(left_ids) >= markers_required) and (right_ids is not None and len(right_ids) >= markers_required):
                if(time.time() - last_capture_time > capture_interval):
                    last_capture_time = time.time()
                    filename = f"{image_folder}/captured_left_frame_{int(last_capture_time)}.png"
                    cv2.imwrite(filename, left_og_frame)
                    
                    filename = f"{image_folder}/captured_right_frame_{int(last_capture_time)}.png"
                    cv2.imwrite(filename, right_og_frame)
                    capture_count += 1                                            
        
            left_og_frame = marker_detector.show_markers(left_og_frame, left_corners, left_ids, 1)    
            right_og_frame = marker_detector.show_markers(right_og_frame, right_corners, right_ids, 1)
        
        combined_frame = cv2.hconcat([left_og_frame, right_og_frame])                
        combined_frame = cv2.resize(combined_frame, (combined_frame.shape[1] // 2, combined_frame.shape[0] // 2))
        cv2.putText(combined_frame, f"Captured: {capture_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(f'Stereo Camera ', combined_frame)
               
        keycode = cv2.waitKeyEx(1)
            
        # Break the loop on 'q' key press
        if keycode == 113:
            break
        
        if keycode == 32:
            timestamp = int(time.time())
            cv2.imwrite(f"images/saves/left_capture_{timestamp}.png", left_og_frame)
            cv2.imwrite(f"images/saves/right_capture_{timestamp}.png", right_og_frame)
            print("Images captured")
            
    # Release the Picamera2 and close windows
    left_picam.stop()
    right_picam.stop()
    cv2.destroyAllWindows()


def preview_stereo(marker_detector, fisheye_calibrations, cameras, rectification_alpha=0.95):
    """Preview stereo camera feed with optional rectification."""
    # Initialize the Picamera2
    left_picam = Picamera2(cameras["left"])
    right_picam = Picamera2(cameras["right"])

    exposure_time = 250000
    # exposure_time = 7500
    analogue_gain = 1.0

    left_cam_config = left_picam.create_video_configuration(
        main={
                "size": (1920, 1080)
            },
        controls={
                "ExposureTime": exposure_time, 
                "AnalogueGain": analogue_gain,
                "AwbEnable": 0,
                "AeEnable": 0
                }
        )

    left_cam_config["transform"] = libcamera.Transform(hflip=1, vflip=1)
    left_picam.configure(left_cam_config)
    
    right_cam_config = right_picam.create_video_configuration(
        main={
                "size": (1920, 1080)
            },
        controls={
                "ExposureTime": exposure_time, 
                "AnalogueGain": analogue_gain,
                "AwbEnable": 0,
                "AeEnable": 0
                }
        )

    right_cam_config["transform"] = libcamera.Transform(hflip=1, vflip=1)
    right_picam.configure(right_cam_config)
    
    left_picam.start()
    right_picam.start()
    print(f"Cameras started")
    
    left_margin = 965
    right_margin = 931

    left_maps = [None, None]
    right_maps = [None, None]
    
    rectify = True
            
    while (True):
        left_frame = left_picam.capture_array()
        right_frame = right_picam.capture_array()
        
        left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
        right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)
        
        if(left_maps[0] is None):
            left_maps = marker_detector.init_fisheye_maps(left_frame, 1.0, fisheye_calibrations["left"])
            
        if(right_maps[0] is None):
            right_maps = marker_detector.init_fisheye_maps(right_frame, 1.0, fisheye_calibrations["right"])
        
        left_og_frame = left_frame.copy()
        right_og_frame = right_frame.copy()
                    
        if(rectify == True):
            left_undist_image = cv2.remap(left_frame, left_maps[0], left_maps[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            right_undist_image = cv2.remap(right_frame, right_maps[0], right_maps[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            
            # left_rectified, right_rectified = self.rectify_images(left_frame, right_frame)
            height, width = left_undist_image.shape[:2]

            # undist_image = cv2.remap(img, left_map1, left_map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            # cv2.line(left_rectified,(width - left_margin - 2,0),(width - left_margin - 2, height),(255,0,0),3)
                            
            combined_frame = cv2.hconcat([left_undist_image, right_undist_image])
            # combined_frame = cv2.hconcat([left_rectified[0:height, 0:width-left_margin], right_rectified[0:height, right_margin:width]])
            
            combined_frame = cv2.resize(combined_frame, (combined_frame.shape[1] // 3, combined_frame.shape[0] // 3))
            cv2.imshow('Rectified Images', combined_frame)
        
        else:                
            # left_undistorted = undistort_image(left_frame, left_calibration["camera_matrix"], left_calibration["dist_coeffs"], 0)
            # 720, 1280
            # cv2.line(left_undistorted,(margin,0),(margin, 720),(255,0,0),2)
            # right_undistorted = undistort_image(right_frame, right_calibration["camera_matrix"], right_calibration["dist_coeffs"], 1)
            # cv2.line(right_undistorted,(1280 - margin, 0),(1280 - margin, 720),(255,0,0),2)
            # combined_frame = cv2.hconcat([left_undistorted[0:720, 0:left_margin], right_undistorted[0:720, 1280 - right_margin:1280]])
            # combined_frame = cv2.hconcat([left_frame[0:720, 0:left_margin], right_frame[0:720, 1280 - right_margin:1280]])
            combined_frame = cv2.hconcat([left_frame, right_frame])
            
            # combined_frame = stitcher.stitch([left_undistorted, right_undistorted])
            cv2.imshow('Undistorted Images', combined_frame)
        
        keycode = cv2.waitKeyEx(1)
        
        # Break the loop on 'q' key press
        if keycode == 113:
            break
        
        if keycode == 44:
            rectification_alpha -= 0.001
            print(f"Rectification alpha: {rectification_alpha}")
        
        if keycode == 46:
            rectification_alpha += 0.001
            print(f"Rectification alpha: {rectification_alpha}")
            
        margin_adjustment = 1
        if keycode == 65364:
            left_margin += margin_adjustment
            print(f"Left margin: {left_margin}")
            
        if keycode == 65362:
            left_margin -= margin_adjustment
            print(f"Left margin: {left_margin}")
        
        if keycode == 65363:
            right_margin += margin_adjustment
            print(f"Right margin: {right_margin}")
            
        if keycode == 65361:
            right_margin -= margin_adjustment
            print(f"Right margin: {right_margin}")
