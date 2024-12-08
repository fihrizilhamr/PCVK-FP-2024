import cv2
import argparse
import csv
import string
import easyocr
import ast
import pandas as pd
import numpy as np
import os

from contextlib import redirect_stdout
from scipy.interpolate import interp1d
from sort.sort import *
from ultralytics import YOLO







def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10):
    cv2.rectangle(img, top_left, bottom_right, color, thickness)
    return img





def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='./sample.mp4', help='source')
    parser.add_argument('--vehicle-detect', type=str, default='./yolov8n.pt', help='vehicle detect model.pt path(s)')
    parser.add_argument('--roi-detect', type=str, default='./license_plate_detector.pt', help='roi detect model.pt path(s)')
    parser.add_argument('--vehicle-conf', type=float, default=0.0, help='vehicle detection confidence threshold')
    parser.add_argument('--roi-conf', type=float, default=0.0, help='roi detection confidence threshold')
    parser.add_argument('--frame-limit', type=int, default=240, help='limit processed frames')
    parser.add_argument('--output', type=str, default='./output_sample.mp4', help='path to save output video')
    
    return parser


def get_car_id(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, *_ = license_plate

    for vehicle in vehicle_track_ids:
        # print(vehicle)
        vx1, vy1, vx2, vy2, vehicle_id = vehicle
        # center
        # if vx1 <= (x2-x1)/2 <= vx2 and vy1 <= (y2-y1)/2 <= vy2: 
        #     return vehicle
        # top left
        if vx1 <= x1 <= vx2 and vy1 <= y1 <= vy2: 
            return vehicle 
        # top right
        # if vx1 <= x2 <= vx2 and vy1 <= y1 <= vy2:
        #     return vehicle
        # bottom left
        # if vx1 <= x1 <= vx2 and vy1 <= y2 <= vy2:
        #     return vehicle
        # bottom right
        # if vx1 <= x2 <= vx2 and vy1 <= y2 <= vy2:
        #     return vehicle
        

    return -1, -1, -1, -1, -1

def find_matching_detection(vehicles, tracked_bbox, vehicle_detections):
    # Convert to numpy arrays for vectorized comparisons
    tracked_bbox = np.array(tracked_bbox[:4])
    detections = np.array([d[:4] for d in vehicle_detections])
    
    
    # Find rows that match the tracked_bbox
    matches = np.all(np.abs(detections - tracked_bbox) <= 8, axis=1)
    if np.sum(matches) > 2:
        print(tracked_bbox)
        print(detections)
        print(matches)
    
    # If a match is found, return the corresponding vehicle type
    if np.any(matches):
        matched_index = np.argmax(matches)  # Get the index of the first match
        class_id = int(vehicle_detections[matched_index][5])
        return vehicles.get(class_id)
    
    return "unknown"


import logging
logging.getLogger('ultralytics').setLevel(logging.ERROR)

def detect_license(opt):
    results = {}
    vehicles_tracker = Sort()

    # Load models
    vehicle_detector = YOLO(opt.vehicle_detect, verbose=False)
    roi_detector = YOLO(opt.roi_detect, verbose=False)

    # Open video source
    cap = cv2.VideoCapture(opt.source)
    vehicles = {2: "car", 3: "motorbike", 5: "bus", 7: "truck"}
    frame_nmr = 0

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(opt.output, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_nmr > opt.frame_limit:
            break
        
        results[frame_nmr] = {}

        # Detect vehicles
        detections = vehicle_detector(frame)[0]

        # Detect license plates
        license_plate_detections = roi_detector(frame)[0]

        print(f"Processing frame number {frame_nmr}: found {len(detections)} vehicle{'s' if len(detections) != 1 else ''} and {len(license_plate_detections)} license plate{'s' if len(license_plate_detections) != 1 else ''}")

        vehicle_detections = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles and score > opt.vehicle_conf:  # Adjust confidence threshold as needed
                vehicle_detections.append([x1, y1, x2, y2, score, class_id])
                
        # print(vehicle_detections)
        # Update tracker
        tracked_vehicles = vehicles_tracker.update(np.array(vehicle_detections))
        # print(tracked_vehicles)

        for license_plate in license_plate_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            if score < opt.roi_conf:  # Adjust confidence threshold as needed
                continue

            # Assign license plate to a tracked vehicle
            vx1, vy1, vx2, vy2, vehicle_id = get_car_id(license_plate, tracked_vehicles)

            vehicle_class = find_matching_detection(vehicles, [vx1, vy1, vx2, vy2], vehicle_detections)

            if vehicle_id != -1:
                # Draw bounding boxes
                draw_border(frame, (int(x1), int(y1)), (int(x2), int(y2)), color=(255, 0, 0))
                draw_border(frame, (int(vx1), int(vy1)), (int(vx2), int(vy2)), color=(0, 255, 0))

                # Masukkan preprocess di sini

                # Save results
                results[frame_nmr][vehicle_id] = {
                    "vehicle": {"bbox": [vx1, vy1, vx2, vy2], "type": vehicle_class},
                    "license_plate": {"bbox": [x1, y1, x2, y2]}
                }

        # Write frame to video
        out.write(frame)
        frame_nmr += 1

    # Release resources
    cap.release()
    out.release()
    print(f"Processed video saved to {opt.output}")
    print(results)



if __name__ == '__main__':
    opt = make_parser().parse_args()
    detect_license(opt)

