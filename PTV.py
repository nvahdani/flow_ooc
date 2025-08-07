import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trackpy as tp
from scipy.signal import medfilt
import os
from tqdm import tqdm
import json
from scipy.signal import savgol_filter

# Define Constants
DEFAULT_PIXEL_SIZE_MM = 1.539e-6    # Default pixel size in mm
DEFAULT_THRESHOLD = 145             # Default threshold for binarization

SETTINGS_DIR = "settings"
os.makedirs(SETTINGS_DIR, exist_ok=True)

def rotate_frame(frame, angle):
    """
    Rotate a frame counterclockwise by a given angle in degrees around its center.

    :param frame: The frame to rotate.
    :param angle: The angle in degrees to rotate.
    :return: The rotated frame.
    """
    (h, w) = frame.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def create_mask(frame_shape, mask_left, mask_right, mask_top, mask_bottom):
    """
    Create a mask from the given frame shape.

    :param frame_shape: The shape of the frame.
    :param mask_left: The left edge of the mask.
    :param mask_right: The right edge of the mask.
    :param mask_top: The top edge of the mask.
    :param mask_bottom: The bottom edge of the mask.
    :return: The mask.
    """
    height, width = frame_shape
    mask = np.ones((height, width), dtype=np.uint8) * 255
    x_min = int(mask_left / 100 * width)
    x_max = int(width - (mask_right / 100) * width)
    y_min = int(mask_top / 100 * height)
    y_max = int(height - (mask_bottom / 100) * height)
    mask[:, :x_min] = 0
    mask[:, x_max:] = 0
    mask[:y_min, :] = 0
    mask[y_max:, :] = 0
    return mask


def calibrate_pixel_size(image, known_distance_mm):
    """
    Calculate the pixel size of the given image.

    :param image: The image to calibrate.
    :param known_distance_mm: The known distance in mm.
    :return: The pixel size of the image.
    """
    clone = image.copy()
    display = clone.copy()
    points = []

    def click_event(event, x, y, flags, param):
        """
        Called when a mouse button is clicked.

        :param event: The mouse event.
        :param x: The x coordinate of the mouse.
        :param y: The y coordinate of the mouse.
        :param flags: The mouse event.
        :param param: The parameter.
        :return: None
        """
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
            if len(points) == 2:
                cv2.line(display, points[0], points[1], (0, 255, 0), 2)
            cv2.imshow("Calibration", display)

    cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration", 800, 600)
    cv2.imshow("Calibration", display)
    cv2.setMouseCallback("Calibration", click_event)
    cv2.waitKey(0)
    cv2.destroyWindow("Calibration")

    if len(points) == 2:
        dist_pixels = np.linalg.norm(np.array(points[0]) - np.array(points[1]))
        pixel_size_mm = known_distance_mm / dist_pixels
        print(f"Calibrated pixel size: {pixel_size_mm:.6e} mm/pixel")
        return pixel_size_mm
    else:
        print("Calibration failed. Using default pixel size.")
        return DEFAULT_PIXEL_SIZE_MM


def interactive_mask_and_threshold(video_path, angle):
    """
    Interactively mask and threshold.

    :param video_path: The path to the video file.
    :param angle: The angle in degrees to rotate.
    :return: The mask and threshold.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cv2.namedWindow('Interactive Frame')
    cv2.resizeWindow('Interactive Frame', 1440, 360)

    cv2.createTrackbar('Frame', 'Interactive Frame', 0, total_frames - 1, lambda x: None)
    cv2.createTrackbar('Threshold', 'Interactive Frame', 127, 255, lambda x: None)
    cv2.createTrackbar('Left Mask (%)', 'Interactive Frame', 37, 100, lambda x: None)
    cv2.createTrackbar('Right Mask (%)', 'Interactive Frame', 52, 100, lambda x: None)
    cv2.createTrackbar('Top Mask (%)', 'Interactive Frame', 19, 100, lambda x: None)
    cv2.createTrackbar('Bottom Mask (%)', 'Interactive Frame', 19, 100, lambda x: None)

    while True:
        frame_number = cv2.getTrackbarPos('Frame', 'Interactive Frame')
        threshold_value = cv2.getTrackbarPos('Threshold', 'Interactive Frame')
        mask_left = cv2.getTrackbarPos('Left Mask (%)', 'Interactive Frame')
        mask_right = cv2.getTrackbarPos('Right Mask (%)', 'Interactive Frame')
        mask_top = cv2.getTrackbarPos('Top Mask (%)', 'Interactive Frame')
        mask_bottom = cv2.getTrackbarPos('Bottom Mask (%)', 'Interactive Frame')

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            break
        frame = rotate_frame(frame, angle)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = create_mask(gray.shape, mask_left, mask_right, mask_top, mask_bottom)
        masked = cv2.bitwise_and(gray, gray, mask=mask)
        _, binary = cv2.threshold(masked, threshold_value, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=3)
        binary = cv2.bitwise_and(binary, mask)

        combined = np.hstack([
            cv2.resize(frame, (480, 360)),
            cv2.resize(cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR), (480, 360)),
            cv2.resize(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), (480, 360)),
        ])
        cv2.imshow('Interactive Frame', combined)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:
            break
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()
    cv2.destroyAllWindows()
    return threshold_value, mask_left, mask_right, mask_top, mask_bottom

def interactive_diameter_test_and_select(frame, mask_left, mask_right, mask_top, mask_bottom, diameters=[5, 7, 9, 11, 13], minmass=100):
    """
    Test the diameter of the given frame.

    :param frame: The frame to test.
    :param mask_left: The left edge of the mask.
    :param mask_right: The right edge of the mask.
    :param mask_top: The top edge of the mask.
    :param mask_bottom: The bottom edge of the mask.
    :param diameters: The list of diameters to test.
    :param minmass: The minimum mass to test.
    :return: The diameter of the given frame.
    """
    mask = create_mask(frame.shape, mask_left, mask_right, mask_top, mask_bottom)
    frame = cv2.bitwise_and(frame, mask)

    selected = []
    for d in diameters:
        features = tp.locate(frame, diameter=d, minmass=minmass)
        print(f"Diameter {d}: {len(features)} particles found")
        tp.annotate(features, frame)
        plt.title(f"Diameter = {d}. Press Y to keep, any other key to skip.")
        plt.show()

        keep = input(f"Keep diameter {d}? (y/n): ").strip().lower()
        if keep == 'y':
            selected.append(d)

    return selected

def process_video(video_path, threshold_value, mask_left, mask_right, mask_top, mask_bottom,
                  pixel_size_mm, angle, diameters=[9], minmass=5):
    """
    Process a video.

    :param video_path: The path to the video file.
    :param threshold_value: The threshold value.
    :param mask_left: The left edge of the mask.
    :param mask_right: The right edge of the mask.
    :param mask_top: The top edge of the mask.
    :param mask_bottom: The bottom edge of the mask.
    :param diameters: The list of diameters to test.
    :param minmass: The minimum mass to test.
    :return: The processed video.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, first_frame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    first_frame = rotate_frame(first_frame, angle)
    mask = create_mask(first_frame.shape[:2], mask_left, mask_right, mask_top, mask_bottom)
    masked_height, masked_width = np.count_nonzero(mask, axis=0).max(), np.count_nonzero(mask, axis=1).max()
    effective_width_mm = masked_width * pixel_size_mm
    effective_length_mm = masked_height * pixel_size_mm

    frames = []
    for frame_idx in tqdm(range(n_frames), desc="Reading video frames"):
        ret, frame = cap.read()
        if not ret:
            break
        frame = rotate_frame(frame, angle)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=3)
        mask = create_mask(gray.shape, mask_left, mask_right, mask_top, mask_bottom)
        masked = cv2.bitwise_and(binary, mask)
        frames.append(masked)

    cap.release()
    all_results = []
    for d in diameters:
        print(f"Tracking with diameter {d}")
        features = []
        for i, frame in enumerate(tqdm(frames, desc=f"Tracking d={d}", unit="frame")):
            f = tp.locate(frame, diameter=d, minmass=minmass)
            if f is not None and len(f) > 0:
                f['frame'] = i
                features.append(f)
        features = pd.concat(features, ignore_index=True)
        search_ranges = features['frame'].apply(lambda t: max(3, 15 - 0.02 * t)).to_numpy()
        linked = tp.link_df(features, search_range=search_ranges, memory=5)
        linked = tp.filter_stubs(linked, threshold=11)
        linked['x_real'] = linked['x'] * pixel_size_mm
        linked['y_real'] = linked['y'] * pixel_size_mm
        linked['dt'] = 1 / fps
        area_mm2 = effective_width_mm * effective_length_mm
        for pid, traj in linked.groupby('particle'):
            if 'frame' in traj.index.names:
                traj = traj.reset_index(drop=True)
            traj = traj.sort_values('frame')
            dx = traj['x_real'].diff()
            dy = traj['y_real'].diff()
            dt = traj['dt']
            speed = np.sqrt(dx**2 + dy**2) / dt
            flow_ul_min = speed * area_mm2 * 60
            all_results.append(pd.DataFrame({'time_min': traj['frame'] / fps / 60, 'flow_ul_min': flow_ul_min, 'diameter': d}))
    return pd.concat(all_results)
    search_ranges = features['frame'].apply(lambda t: max(3, 15 - 0.02 * t)).to_numpy()
    linked = tp.link_df(features, search_range=search_ranges, memory=5)
    linked = tp.filter_stubs(linked, threshold=11)

    linked['x_real'] = linked['x'] * pixel_size_mm
    linked['y_real'] = linked['y'] * pixel_size_mm
    linked['dt'] = 1 / fps

    area_mm2 = (effective_width_mm) * (effective_length_mm)
    results = []
    for pid, traj in tqdm(linked.groupby('particle'), desc="Computing velocities"):
        if 'frame' in traj.index.names:
            traj = traj.reset_index(drop=True)
        traj = traj.sort_values('frame')
        dx = traj['x_real'].diff()
        dy = traj['y_real'].diff()
        dt = traj['dt']
        speed = np.sqrt(dx**2 + dy**2) / dt
        flow_ul_min = speed * area_mm2 * 60
        results.append(pd.DataFrame({'time_min': traj['frame'] / fps / 60, 'flow_ul_min': flow_ul_min}))

    return pd.concat(results)

def smooth_data(data, kernel_size=29):
    """
    Smooth data uses a kernel of size 29 with the medilt function of Scipy performing a median filter.
    Taking the velocity measurements of every particle tracked as an array. Applying a median filter with 29 array entries.

    :param data: data to smooth
    :param kernel_size: size of kernel
    :return: smoothed data
    """
    if len(data) < kernel_size:
        kernel_size = len(data) if len(data) % 2 == 1 else len(data) - 1
        if kernel_size < 3:
            return data  # too short for meaningful smoothing
    return medfilt(data, kernel_size=kernel_size)

def save_settings(name, settings):
    """
    Save settings to a JSON file.

    :param name: The name of the file.
    :param settings: The settings to save.
    :return: None
    """
    with open(os.path.join(SETTINGS_DIR, f"{name}.json"), 'w') as f:
        json.dump(settings, f)


def load_settings(name):
    """
    Load settings from a JSON file.

    :param name: The name of the settings file.
    :return: None
    """
    try:
        path = os.path.join(SETTINGS_DIR, f"{name}.json")
        if os.path.getsize(path) == 0:
            print(f"Warning: Settings file '{path}' is empty. Skipping.")
            return None
        with open(path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Failed to load settings for {name}: {e}")
        return None


def run(video_path, angle, threshold, ml, mr, mt, mb, pixel_size_mm, diameters, output_path=None):
    """
    Run the tracking algorithm.

    :param video_path: path to video file
    :param angle: angle in degrees
    :param threshold: threshold in degrees
    :param ml: number of pixel from the middle to the left border
    :param mr: number of pixel from the middle to the right border
    :param mt: number of pixel from the middle to the top border
    :param mb: number of pixel from the middle to the bottom border
    :param pixel_size_mm: pixel size in millimeters
    :param diameters: diameters in millimeters
    :return: None
    """
    print("Tracking particles...")
    data = process_video(video_path, threshold, ml, mr, mt, mb, pixel_size_mm, angle, diameters=diameters)
    grouped = data.groupby('time_min')['flow_ul_min'].mean().reset_index()
    for d in data['diameter'].unique():
        d_data = data[data['diameter'] == d]
        grouped = d_data.groupby('time_min')['flow_ul_min'].mean().reset_index()
        smoothed = smooth_data(grouped['flow_ul_min'].values)
        plt.plot(grouped['time_min'], smoothed, label=f'Diameter {d}')
    plt.xlabel('Time [min]')
    plt.ylabel('Flow Rate (µL/min)')
    plt.title('Smoothed Flow Rate per Diameter')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(grouped['time_min'], smoothed, label='Smoothed Flow Velocity (µL/min)', color='red')
    plt.xlabel('Time [min]')
    plt.ylabel('Flow Rate (µL/min)')
    plt.title('Smoothed Particle Velocity Over Time')
    plt.grid(True)
    plt.legend()
    plt.show()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pd.DataFrame({
            'Time (min)': grouped['time_min'],
            'Smoothed Flow Rate (µL/min)': smoothed
        }).to_excel(output_path, index=False)
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    video_configs = [
        ("Video/Video1.2025-05-06-12-25-05.AVI", 95, "Data/Video/Video1/flow_velocity.xlsx"),
        ("Video/Video2.2025-05-06-13-01-49.AVI", 90, "Data/Video/Video2/flow_velocity.xlsx"),
        ("Video/Video3.2025-05-06-13-56-47.AVI", 92, "Data/Video/Video3/flow_velocity.xlsx"),
        ("Video/Video4.2025-05-06-14-33-09.AVI", 93, "Data/Video/Video4/flow_velocity.xlsx"),
    ]

    video_settings = []
    for video_path, angle, save_path in video_configs:
        name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n--- Preparing settings for {name} with angle {angle} ---")

        settings = load_settings(name)
        if settings is None:
            result = interactive_mask_and_threshold(video_path, angle)
            if result is None:
                continue
            threshold, ml, mr, mt, mb = result

            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cv2.namedWindow("Select Calibration Frame", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Select Calibration Frame", 800, 600)
            cv2.createTrackbar("Frame", "Select Calibration Frame", 0, total_frames - 1, lambda x: None)

            while True:
                frame_idx = cv2.getTrackbarPos("Frame", "Select Calibration Frame")
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rot = rotate_frame(frame, angle)
                cv2.imshow("Select Calibration Frame", frame_rot)
                key = cv2.waitKey(30) & 0xFF
                if key == 13:
                    break
                elif key == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    continue

            cap.release()
            cv2.destroyAllWindows()
            pixel_size_mm = calibrate_pixel_size(frame_rot, known_distance_mm=0.5)
            gray = cv2.cvtColor(frame_rot, cv2.COLOR_BGR2GRAY)
            diameters = interactive_diameter_test_and_select(gray, ml, mr, mt, mb)
            if not diameters:
                print("No diameters selected. Defaulting to [9].")
                diameters = [9]

            settings = {
                "threshold": threshold, "ml": ml, "mr": mr, "mt": mt, "mb": mb,
                "pixel_size_mm": pixel_size_mm, "selected_diameters": diameters
            }
            save_settings(name, settings)
        else:
            threshold = settings["threshold"]
            ml = settings["ml"]
            mr = settings["mr"]
            mt = settings["mt"]
            mb = settings["mb"]
            pixel_size_mm = settings["pixel_size_mm"]
            diameters = settings.get("selected_diameters", [9])

        video_settings.append((video_path, angle, threshold, ml, mr, mt, mb, pixel_size_mm, diameters, save_path))

    for (video_path, angle, threshold, ml, mr, mt, mb, pixel_size_mm, diameters, save_path) in video_settings:
        run(video_path, angle, threshold, ml, mr, mt, mb, pixel_size_mm, diameters, output_path=save_path)