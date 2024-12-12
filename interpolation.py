import csv
import argparse
import ast
import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='./sample.mp4', help='source')
    parser.add_argument('--source-data', type=str, default='./data.csv', help='data source')
    parser.add_argument('--output', type=str, default='./interpolated_output_sample.mp4', help='path to save output video')
    parser.add_argument('--output-data', type=str, default='./interpolated_data.csv', help='path to save output data')
    return parser

def interpolate_bounding_boxes(data):
    frame_numbers = np.array([int(row['frame']) for row in data])
    vehicle_ids = np.array([int(float(row['vehicle_id'])) for row in data])
    vehicle_bboxes = np.array([list(map(float, row['vehicle_bbox'][1:-1].split())) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])

    interpolated_data = []
    unique_car_ids = np.unique(vehicle_ids)
    for car_id in unique_car_ids:

        frame_numbers_ = [p['frame'] for p in data if int(float(p['vehicle_id'])) == int(float(car_id))]
        # print(frame_numbers_, car_id)

        car_mask = vehicle_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []

        first_frame_number = car_frame_numbers[0]
        last_frame_number = car_frame_numbers[-1]

        for i in range(len(vehicle_bboxes[car_mask])):
            frame_number = car_frame_numbers[i]
            car_bbox = vehicle_bboxes[car_mask][i]
            license_plate_bbox = license_plate_bboxes[car_mask][i]

            if i > 0:
                prev_frame_number = car_frame_numbers[i-1]
                prev_car_bbox = car_bboxes_interpolated[-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

                if frame_number - prev_frame_number > 1:
                    # Interpolate missing frames' bounding boxes
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
                    interpolated_car_bboxes = interp_func(x_new)
                    interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
                    interpolated_license_plate_bboxes = interp_func(x_new)

                    car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
                    license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])

            car_bboxes_interpolated.append(car_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)

        for i in range(len(car_bboxes_interpolated)):
            frame_number = first_frame_number + i
            row = {}
            row['frame'] = str(frame_number)
            row['vehicle_id'] = str(car_id)
            row['vehicle_bbox'] = ' '.join(map(str, car_bboxes_interpolated[i]))
            row['license_plate_bbox'] = ' '.join(map(str, license_plate_bboxes_interpolated[i]))

            if str(frame_number) not in frame_numbers_:
                # Imputed row, set the following fields to '0'
                row['license_plate_bbox_score'] = '0'
                row['license_number'] = '0'
                row['license_number_score'] = '0'
                row['vehicle_type'] = 'Unknown'  # Or any default placeholder
            else:
                # Original row, retrieve values from the input data if available
                original_row = [p for p in data if int(p['frame']) == frame_number and int(float(p['vehicle_id'])) == int(float(car_id))][0]
                row['license_plate_bbox_score'] = original_row['license_plate_bbox_score'] if 'license_plate_bbox_score' in original_row else '0'
                row['license_number'] = original_row['license_number'] if 'license_number' in original_row else '0'
                row['license_number_score'] = original_row['license_number_score'] if 'license_number_score' in original_row else '0'
                row['vehicle_type'] = original_row['vehicle_type'] if 'vehicle_type' in original_row else 'Unknown'


            interpolated_data.append(row)

    return interpolated_data

def start_interpolating(opt):
    # Load the CSV file
    with open(opt.source_data, 'r') as file:
        reader = csv.DictReader(file)
        data = list(reader)

    # Interpolate missing data
    interpolated_data = interpolate_bounding_boxes(data)

    # Write updated data to a new CSV file
    header = ['frame', 'vehicle_id', 'vehicle_bbox', 'vehicle_type',
            'license_plate_bbox', 'license_plate_bbox_score', 
            'license_number', 'license_number_score']
    with open(opt.output_data, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(interpolated_data)

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10):
    cv2.rectangle(img, top_left, bottom_right, color, thickness)
    return img

def start_visualizing(opt):
    
    results = pd.read_csv(opt.output_data, encoding='ISO-8859-1')
    results['license_number_score'] = pd.to_numeric(results['license_number_score'], errors='coerce')


    # load video
    video_path = opt.source
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Specify the codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(opt.output, fourcc, fps, (width, height))

    license_plate = {}
    for car_id in np.unique(results['vehicle_id']):
        max_ = np.amax(results[results['vehicle_id'] == car_id]['license_number_score'])
        license_plate[car_id] = {'license_crop': None,
                                'license_plate_number': results[(results['vehicle_id'] == car_id) &
                                                                (results['license_number_score'] == max_)]['license_number'].iloc[0]}
        cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['vehicle_id'] == car_id) &
                                                (results['license_number_score'] == max_)]['frame'].iloc[0])
        ret, frame = cap.read()

        x1, y1, x2, y2 = ast.literal_eval(results[(results['vehicle_id'] == car_id) &
                                                (results['license_number_score'] == max_)]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))

        license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
        license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

        license_plate[car_id]['license_crop'] = license_crop


    frame_nmr = -1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # read frames
    ret = True
    while ret:
        ret, frame = cap.read()
        frame_nmr += 1
        if ret:
            df_ = results[results['frame'] == frame_nmr]
            for row_indx in range(len(df_)):
                # draw car
                car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['vehicle_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25)

                text_position = (int(car_x1), int(car_y1) - 30)  # Posisi teks sedikit di atas bounding box
                # print(df_.iloc[row_indx]['vehicle_type'])
                cv2.putText(
                    frame,
                    str(df_.iloc[row_indx]['vehicle_type']), 
                    text_position,
                    cv2.FONT_HERSHEY_SIMPLEX,  # Jenis font
                    2,  # Ukuran font
                    (0, 255, 0),  # Warna teks (hijau)
                    2,  # Ketebalan garis teks
                    cv2.LINE_AA  # Jenis garis
                )


                # draw license plate
                x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

                # crop license plate
                license_crop = license_plate[df_.iloc[row_indx]['vehicle_id']]['license_crop']

                H, W, _ = license_crop.shape

                try:
                    # Coordinates for the license crop
                    top_left = (int((car_x2 + car_x1 - W) / 2), int(car_y1) - H - 100)
                    bottom_right = (int((car_x2 + car_x1 + W) / 2), int(car_y1) - 100)

                    # Adding the cropped license plate
                    frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :] = license_crop

                    # Adding a white block behind the text
                    text_top_left = (top_left[0], top_left[1] - 300)
                    text_bottom_right = (bottom_right[0], top_left[1])
                    frame[text_top_left[1]:text_bottom_right[1], text_top_left[0]:text_bottom_right[0], :] = (255, 255, 255)

                    # Adding a blue border around the license plate region
                    cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 10)  # Blue border, thickness = 10

                    # Adding a blue border around the text background
                    cv2.rectangle(frame, text_top_left, text_bottom_right, (255, 0, 0), 10)

                    # Adding the license plate number as text
                    (text_width, text_height), _ = cv2.getTextSize(
                        license_plate[df_.iloc[row_indx]['vehicle_id']]['license_plate_number'],
                        cv2.FONT_HERSHEY_SIMPLEX,
                        4.3,
                        17)

                    text_x = int((car_x2 + car_x1 - text_width) / 2)
                    text_y = int(car_y1 - H - 250 + (text_height / 2))

                    cv2.putText(frame,
                                license_plate[df_.iloc[row_indx]['vehicle_id']]['license_plate_number'],
                                (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                4.3,
                                (0, 0, 0),
                                17)


                except:
                    pass

            out.write(frame)
            frame = cv2.resize(frame, (1280, 720))

            # cv2.imshow('frame', frame)
            # cv2.waitKey(0)

    out.release()
    cap.release()


if __name__ == '__main__':
    opt = make_parser().parse_args()
    start_interpolating(opt)
    start_visualizing(opt)    