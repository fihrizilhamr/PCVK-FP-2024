import cv2
import argparse
import csv
import string
import easyocr
import ast
import pandas as pd
import numpy as np
import os
import re

from datetime import datetime


from time import time, perf_counter
from contextlib import redirect_stdout
from scipy.interpolate import interp1d
from sort.sort import *
from ultralytics import YOLO

char_to_int = { 'O': '0',
                'I': '1',
                'J': '3',
                'A': '4',
                'G': '6',
                'S': '5'}

int_to_char = { '0': 'O',
                '1': 'I',
                '3': 'J',
                '4': 'A',
                '6': 'G',
                '5': 'S'}

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to enhance text
    _, thresholded = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)

    # save_path = f"preprocessed_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

    # if save_path:
    #     cv2.imwrite(save_path, thresholded)

    return thresholded


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10):
    cv2.rectangle(img, top_left, bottom_right, color, thickness)
    return img


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='./sample.mp4', help='source')
    parser.add_argument('--vehicle-detect', type=str, default='./yolov8n.pt', help='vehicle detect model.pt path(s)')
    parser.add_argument('--roi-detect', type=str, default='./license_plate_detector.pt', help='roi detect model.pt path(s)')
    parser.add_argument('--vehicle-conf', type=float, default=0.0, help='vehicle detection confidence threshold')
    parser.add_argument('--roi-conf', type=float, default=0.5, help='roi detection confidence threshold')
    parser.add_argument('--frame-limit', type=int, default=1800, help='limit processed frames')
    parser.add_argument('--output', type=str, default='./output_sample.mp4', help='path to save output video')
    parser.add_argument('--output-data', type=str, default='./data.csv', help='path to save output data')
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

def licenseFormatCheck(text):
    if not (len(text) >= 7 and len(text) <= 9):
        return False
    
    # Regex untuk format plat nomor: 1-2 huruf, diikuti 1-4 angka, diikuti 1-3 huruf
    # pattern = r'^[A-Z]{1,2}\d{1,4}[A-Z]{1,3}$'
    return True


def formatLicense(text):
    # format plat nomor agar sesuai dengan menkonversi karakter
    license_plate_ = ''
    for char in text:
        # Konversi karakter sesuai pemetaan
        if char in int_to_char:
            license_plate_ += int_to_char[char]
        elif char in char_to_int:
            license_plate_ += char_to_int[char]
        else:
            license_plate_ += char  # Jika tidak ada dalam pemetaan, gunakan karakter asli

    return license_plate_


def readLicensePlateString(license_plate_crop_img):
    # membaca string plat nomor menggunakan OCR
    reader = easyocr.Reader(['en'], gpu=True)
    detections = reader.readtext(license_plate_crop_img)

    for detection in detections:
        bbox, text, score = detection

        text = text.upper().replace(' ', '')

        if licenseFormatCheck(text):
            return formatLicense(text), score

    return None, None

def saveToCSV(results, output_path):
    # Save the results to a CSV format for visualization
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{},{}\n'.format(
            'frame', 'vehicle_id', 'vehicle_bbox', 'vehicle_type',
            'license_plate_bbox', 'license_plate_bbox_score', 
            'license_number', 'license_number_score'
        ))

        for frame_nmr, vehicles in results.items():
            for vehicle_id, vehicle_data in vehicles.items():
                if 'vehicle' in vehicle_data and 'license_plate' in vehicle_data:
                    vehicle = vehicle_data['vehicle']
                    license_plate = vehicle_data['license_plate']

                    # Extract the required fields
                    vehicle_bbox = '[{} {} {} {}]'.format(
                        vehicle['bbox'][0], vehicle['bbox'][1], 
                        vehicle['bbox'][2], vehicle['bbox'][3]
                    )
                    vehicle_type = vehicle.get('type', 'unknown')

                    lp_bbox = '[{} {} {} {}]'.format(
                        license_plate['bbox'][0], license_plate['bbox'][1], 
                        license_plate['bbox'][2], license_plate['bbox'][3]
                    )
                    lp_bbox_score = license_plate.get('bbox_score', 0.0)
                    lp_text = license_plate.get('text', 'N/A')
                    lp_text_score = license_plate.get('text_score', 0.0)

                    # Write to CSV
                    f.write('{},{},{},{},{},{},{},{}\n'.format(
                        frame_nmr, vehicle_id, vehicle_bbox, vehicle_type,
                        lp_bbox, lp_bbox_score, lp_text, lp_text_score
                    ))
        f.close()


import logging
logging.getLogger('ultralytics').setLevel(logging.ERROR)
logging.getLogger('easyocr').setLevel(logging.ERROR)
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
        if not ret or frame_nmr >= opt.frame_limit:
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

        # Update tracker
        tracked_vehicles = vehicles_tracker.update(np.array(vehicle_detections))

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

                # Crop License plate untuk process deteksi string
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # Preprocessing
                pre_prosessed_image = preprocess_image(license_plate_crop)
               

                # Baca license plate number
                license_plate_text, license_plate_text_score = readLicensePlateString(pre_prosessed_image)

                # Jika text ditemukan, tampilkan di layar
                if license_plate_text is not None:
                    # Tampilkan teks di atas bounding box plat nomor
                    text_position = (int(x1), int(y1) - 10)  # Posisi teks sedikit di atas bounding box
                    cv2.putText(
                        frame,
                        license_plate_text,  # Teks plat nomor
                        text_position,
                        cv2.FONT_HERSHEY_SIMPLEX,  # Jenis font
                        0.7,  # Ukuran font
                        (0, 255, 0),  # Warna teks (hijau)
                        2,  # Ketebalan garis teks
                        cv2.LINE_AA  # Jenis garis
                    )

                # Simpan hasil ke dictionary untuk logging atau CSV
                    results[frame_nmr][vehicle_id] = {'vehicle': {'bbox': [vx1, vy1, vx2, vy2], "type": vehicle_class},
                                                        'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': license_plate_text,
                                                                        'bbox_score': score,
                                                                        'text_score': license_plate_text_score}}

        # Write frame to video
        out.write(frame)
        frame_nmr += 1

    # Release resources
    cap.release()
    out.release()
    print(f"Processed video saved to {opt.output}")
    return results

if __name__ == '__main__':
    opt = make_parser().parse_args()
    t1 = perf_counter()
    result = detect_license(opt)
    t2 = perf_counter()
    print(f"Processing time: {t2 - t1} seconds")
    print(f"Averaging {opt.frame_limit/(t2 - t1)} fps")
    print(f"Result{result}")
    saveToCSV(result, opt.output_data)