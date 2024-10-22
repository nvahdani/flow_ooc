import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

flowrate_calculation_ul_per_min = [
    7.938378156, 7.842708635, 7.749317582, 7.658124558, 7.569052866, 7.48202934,
    7.396984136, 7.313850552, 7.232564852, 7.1530661, 7.075296013, 6.999198813,
    6.924721097, 6.851811712, 6.780421634, 6.710503865, 6.642013322, 6.574906747,
    6.509142611, 6.444681031, 6.381483687, 6.31951375, 6.258735804, 6.199115787,
    6.140620918, 6.083219646, 6.026881586, 5.971577469, 5.91727909, 5.863959263,
    5.811591771, 5.760151325, 5.709613524, 5.659954817, 5.611152464, 5.563184502,
    5.516029714, 5.469667596, 5.424078328, 5.379242745, 5.335142308, 5.291759086,
    5.249075723, 5.207075418, 5.165741906, 5.125059434, 5.085012738, 5.045587032,
    5.006767982, 4.968541693, 4.930894691, 4.893813907, 4.857286662, 4.821300653,
    4.785843939, 4.750904928, 4.716472364, 4.682535314, 4.649083158, 4.616105579,
    4.583592547, 4.551534315, 4.519921408, 4.48874461, 4.457994958, 4.427663734,
    4.397742455, 4.368222865, 4.33909693, 4.310356828, 4.281994942, 4.254003855,
    4.226376343, 4.199105368, 4.172184071, 4.145605772, 4.119363955, 4.093452272,
    4.067864531, 4.042594695, 4.017636877, 3.992985333, 3.968634459, 3.944578788,
    3.920812985, 3.897331842, 3.874130273, 3.851203317, 3.828546126, 3.806153967,
    3.784022217, 3.762146359, 3.74052198, 3.71914477, 3.698010514, 3.677115095,
    3.656454485, 3.63602475, 3.615822041, 3.595842595, 3.57608273, 3.556538847,
    3.537207425, 3.518085016, 3.49916825, 3.480453828, 3.461938519, 3.443619163,
    3.425492665, 3.407555997, 3.389806191, 3.372240343, 3.354855608, 3.337649198,
    3.320618385, 3.303760494, 3.287072903, 3.270553047, 3.254198408, 3.238006521,
    3.221974967, 3.206101377, 3.190383429, 3.174818843, 3.159405387, 3.144140869,
    3.129023142, 3.114050098, 3.09921967, 3.08452983, 3.069978588, 3.055563992,
    3.041284127, 3.027137112, 3.013121101, 2.999234284, 2.985474883, 2.971841151,
    2.958331375, 2.944943872, 2.93167699, 2.918529106, 2.905498625, 2.892583983,
    2.879783641, 2.867096088, 2.854519842, 2.842053442, 2.829695457, 2.817444478,
    2.805299121, 2.793258027, 2.781319858, 2.7694833, 2.757747062, 2.746109873,
    2.734570486, 2.723127671, 2.711780222, 2.700526952, 2.689366693, 2.678298297,
    2.667320634, 2.656432593, 2.64563308, 2.634921022, 2.624295358, 2.61375505,
    2.603299071, 2.592926415, 2.582636089, 2.572427116, 2.562298537, 2.552249404,
    2.542278788, 2.53238577, 2.52256945, 2.512828938, 2.503163359, 2.493571853,
    2.484053571, 2.474607677, 2.46523335, 2.455929778, 2.446696164, 2.437531721,
    2.428435676, 2.419407265, 2.410445737, 2.401550351, 2.392720378, 2.383955099,
    2.375253805, 2.366615799, 2.358040392, 2.349526907, 2.341074674, 2.332683037,
    2.324351344, 2.316078957, 2.307865243, 2.299709582, 2.29161136, 2.283569972,
    2.275584822, 2.267655322, 2.259780892, 2.251960961, 2.244194965, 2.236482347,
    2.22882256, 2.221215062, 2.21365932, 2.206154807, 2.198701005, 2.1912974,
    2.183943487, 2.176638769, 2.169382752, 2.162174952, 2.155014889, 2.14790209,
    2.14083609, 2.133816428, 2.12684265, 2.119914307, 2.113030956, 2.106192161,
    2.09939749, 2.092646519, 2.085938826, 2.079273997, 2.072651622, 2.066071297,
    2.059532622, 2.053035204, 2.046578654, 2.040162586, 2.033786621, 2.027450385,
    2.021153507, 2.014895622, 2.008676368, 2.00249539, 1.996352334, 1.990246854,
    1.984178604, 1.978147246, 1.972152444, 1.966193867, 1.960271188, 1.954384082,
    1.948532231, 1.942715319, 1.936933034, 1.931185068, 1.925471115, 1.919790875,
    1.914144051, 1.908530348, 1.902949476, 1.897401148, 1.89188508, 1.88640099,
    1.880948603, 1.875527644, 1.870137842, 1.864778929, 1.859450641, 1.854152715,
    1.848884892, 1.843646918, 1.838438539, 1.833259505, 1.828109568, 1.822988484,
    1.817896011,1.812831911, 1.807795946, 1.802787883, 1.797807491, 1.792854541, 1.787928806,
1.783030064, 1.778158092, 1.773312672, 1.768493587, 1.763700624, 1.758933571,
1.754192217, 1.749476356, 1.744785783, 1.740120294, 1.73547969, 1.730863771,
1.726272341, 1.721705207, 1.717162174, 1.712643054, 1.708147657, 1.703675798,
1.699227293, 1.694801957, 1.690399612, 1.686020079, 1.68166318, 1.67732874,
1.673016587, 1.668726549, 1.664458456, 1.660212141, 1.655987436, 1.651784178,
1.647602203, 1.643441351, 1.639301461, 1.635182376, 1.631083939, 1.627005996,
1.622948392, 1.618910977, 1.6148936, 1.610896112, 1.606918365, 1.602960215,
1.599021516, 1.595102126, 1.591201902, 1.587320705, 1.583458395, 1.579614836,
1.575789891, 1.571983424, 1.568195303, 1.564425395, 1.560673569, 1.556939696,
1.553223646, 1.549525293, 1.54584451, 1.542181172, 1.538535156, 1.53490634,
1.531294601, 1.527699819, 1.524121876, 1.520560653, 1.517016033, 1.513487901,
1.509976141, 1.506480641, 1.503001286, 1.499537967, 1.496090572, 1.492658991,
1.489243116, 1.48584284, 1.482458055, 1.479088657, 1.47573454, 1.472395601,
1.469071737, 1.465762846, 1.462468828, 1.459189581, 1.455925008, 1.452675009,
1.449439487, 1.446218347, 1.443011491, 1.439818826, 1.436640257, 1.433475692,
1.430325037, 1.427188202, 1.424065095, 1.420955627, 1.417859708, 1.414777251,
1.411708168, 1.408652371, 1.405609774, 1.402580293, 1.399563843, 1.396560339,
1.393569699, 1.390591841, 1.387626681, 1.38467414, 1.381734137, 1.378806592,
1.375891426, 1.372988561, 1.37009792, 1.367219424, 1.364352998, 1.361498567,
1.358656054, 1.355825385, 1.353006487, 1.350199286, 1.34740371, 1.344619686,
1.341847143, 1.339086011, 1.336336218, 1.333597696, 1.330870374, 1.328154185,
1.325449061, 1.322754933, 1.320071735, 1.317399402, 1.314737865, 1.312087062,
1.309446926, 1.306817393, 1.304198401, 1.301589884, 1.298991782, 1.29640403,
1.293826569, 1.291259336, 1.288702271, 1.286155313, 1.283618403, 1.281091481,
1.278574489, 1.276067367, 1.273570059, 1.271082506, 1.268604652, 1.266136439,
1.263677813, 1.261228716, 1.258789094, 1.256358891, 1.253938055, 1.251526529,
1.249124261, 1.246731198, 1.244347286, 1.241972474, 1.239606708, 1.237249939,
1.234902114, 1.232563183, 1.230233095, 1.2279118, 1.225599249, 1.223295391,
1.221000179, 1.218713564, 1.216435497, 1.214165931, 1.211904818, 1.209652111,
1.207407763, 1.205171728, 1.202943959, 1.200724411, 1.198513039, 1.196309798,
1.194114641, 1.191927526, 1.189748409, 1.187577244, 1.185413989, 1.183258602,
1.181111037, 1.178971255, 1.176839211, 1.174714865, 1.172598174, 1.170489098,
1.168387594, 1.166293624, 1.164207145, 1.162128119, 1.160056505, 1.157992263,
1.155935355, 1.153885741, 1.151843382, 1.149808241, 1.147780279, 1.145759457,
1.143745739, 1.141739087, 1.139739464, 1.137746832, 1.135761156, 1.133782399,
1.131810525, 1.129845498, 1.127887283, 1.125935843, 1.123991145, 1.122053153,
1.120121832, 1.118197148, 1.116279067, 1.114367555, 1.112462579, 1.110564104,
1.108672099, 1.106786528, 1.104907361, 1.103034564, 1.101168105, 1.099307952,
1.097454072, 1.095606435, 1.093765009, 1.091929762, 1.090100664, 1.088277683,
1.08646079, 1.084649952, 1.082845142, 1.081046327, 1.079253479, 1.077466568,
1.075685564, 1.073910438, 1.072141161, 1.070377705, 1.06862004, 1.066868138,
1.065121971, 1.06338151, 1.061646728, 1.059917597, 1.05819409, 1.056476178,
1.054763836, 1.053057035, 1.051355749, 1.049659951, 1.047969615, 1.046284714,
1.044605223, 1.042931115, 1.041262364, 1.039598944, 1.037940831, 1.036287999,
1.034640422, 1.032998076, 1.031360936, 1.029728976, 1.028102173, 1.026480503,
1.02486394, 1.02325246, 1.021646041, 1.020044657, 1.018448286, 1.016856903,
1.015270486, 1.013689012, 1.012112456, 1.010540797, 1.008974012, 1.007412077,
1.005854971, 1.00430267, 1.002755154, 1.001212399, 0.999674385, 0.998141088,
0.996612487, 0.995088562, 0.993569289,0.992054649, 0.99054462, 0.98903918, 0.98753831, 0.986041988, 0.984550193,
0.983062906, 0.981580105, 0.980101771, 0.978627883, 0.977158421, 0.975693366,
0.974232697, 0.972776395, 0.97132444, 0.969876813, 0.968433495, 0.966994466,
0.965559707, 0.9641292, 0.962702925, 0.961280864, 0.959862997, 0.958449308,
0.957039776, 0.955634384, 0.954233113, 0.952835946, 0.951442864, 0.95005385,
0.948668886, 0.947287954, 0.945911036, 0.944538115, 0.943169174, 0.941804195,
0.940443161, 0.939086055, 0.93773286, 0.93638356, 0.935038137, 0.933696575,
0.932358857, 0.931024966, 0.929694887, 0.928368603, 0.927046097, 0.925727354,
0.924412358, 0.923101092, 0.921793541, 0.920489689, 0.91918952, 0.917893019,
0.91660017, 0.915310958, 0.914025368, 0.912743384, 0.911464991, 0.910190174,
0.908918918, 0.907651208, 0.906387029, 0.905126367, 0.903869207, 0.902615535
]

# Kalman filter initialization
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
# Initial transitionMatrix and processNoiseCov
kalman.transitionMatrix = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01

# Function to dynamically adjust the Kalman filter based on velocity
def adjust_kalman_based_on_velocity(velocity):
    if velocity > 3:  # If particles are moving fast
        kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1  # Increase noise for fast motion
    else:  # If particles are slowing down
        kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 0.9, 0],  # Slow down motion
            [0, 0, 0, 0.9]
        ], np.float32)
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03  # Lower noise for slow motion

    # Function to predict the next position using Kalman Filter
def predict_next_position( ):
    prediction = kalman.predict()
    return int(prediction[0]), int(prediction[1])

# Update the Kalman filter with a new measurement
def update_kalman(current_position):
    measurement = np.array([[np.float32(current_position[0])], [np.float32(current_position[1])]])
    kalman.correct(measurement)

def calculate_flow_rate_limit_flow_rate(dimension_of_flow: str, length_um: float, width_um: float, pixel_mm: float,
                                        prev_position: tuple, current_position: tuple, frames_per_seconds: int,
                                        prev_flow_rate: float, max_change: float):
    dx = current_position[0] - prev_position[0]
    dy = current_position[1] - prev_position[1]
    distance_in_pixels = np.sqrt(dx ** 2 + dy ** 2)
    distance_in_meters = distance_in_pixels * pixel_mm
    velocity_m_per_s = distance_in_meters * frames_per_seconds

    area_m2 = (length_um * 1e-6) * (width_um * 1e-6)
    current_flow_rate_m3_s = velocity_m_per_s * area_m2
    current_flow_rate = current_flow_rate_m3_s

    if dimension_of_flow == 'microliter per seconds':
        current_flow_rate = current_flow_rate_m3_s * 1e9
        if abs(current_flow_rate - prev_flow_rate) > max_change:
            return prev_flow_rate

    elif dimension_of_flow == 'microliter per minute':
        current_flow_rate = current_flow_rate_m3_s * 1e9 * 60

        if abs(current_flow_rate - prev_flow_rate) > max_change:
            return prev_flow_rate
    return current_flow_rate

pixel_size_mm = 1.539e-6
max_change_threshold = 40
time_values = []
velocity_values = []
average_velocity_values = []
flow_rate_values = []
current_velocities = []
second_counter = 0
velocity_pixel_s_list = []
previous_velocities = []
previous_flow_rate = []
previous_positions = []
frame_number = 0
frames = []
maps = []

# Load the video
video_path = 'Video/241001dilution4-method2.2024-10-01-15-19-36.AVI'
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()
# Get total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Get frame rate per second
fps = cap.get(cv2.CAP_PROP_FPS)

width_video = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height_video = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

video_flowrate = 'Data/Flow_rate_particle.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_flowrate = cv2.VideoWriter(video_flowrate, fourcc, fps, (width_video, height_video))
video_mask = 'Data/Mask_Flow_rate_particle.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_mask = cv2.VideoWriter(video_mask, fourcc, fps, (width_video, height_video))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create a mask that excludes the middle part of the frame
    mask_left = np.ones(gray_frame.shape, dtype=np.uint8) * 255
    # Define the rectangle coordinates for the middle region
    start_x_left = 0#int(binary_frame.shape[1] / 2) - 0
    end_x_left = 1300#int(binary_frame.shape[1] / 2) + 0
    cv2.rectangle(mask_left, (start_x_left, 0), (end_x_left, gray_frame.shape[0]), (0, 0, 0),
                  -1)  # Black rectangle to exclude middle

    # Apply the left mask
    masked_frame_left = cv2.bitwise_and(gray_frame, mask_left)

    # Create a mask that excludes the middle part of the frame
    mask_right = np.ones(gray_frame.shape, dtype=np.uint8) * 255  # Create a white mask
    # Define the rectangle coordinates for the middle region
    start_x_right = 1400
    end_x_right = 2045
    cv2.rectangle(mask_right, (start_x_right, 0), (end_x_right, gray_frame.shape[0]), (0, 0, 0),
                  -1)

    # Apply the mask
    masked_frame = cv2.bitwise_and(masked_frame_left, mask_right)

    # Apply binary threshold and erosion
    _, binary_frame = cv2.threshold(masked_frame, 15, 255, cv2.THRESH_BINARY)
    kernel = np.ones((9,9), np.uint8)

    # Apply closing to fill small holes inside the particles
    closing = cv2.morphologyEx(binary_frame, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    current_positions = []

    for cnt in contours:

        area = cv2.contourArea(cnt)
        if area < 35:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            current_positions.append((cx, cy))

            update_kalman((cx, cy))

            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    if not current_positions:
        predicted_position = predict_next_position()
        cv2.circle(frame, predicted_position, 5, (0, 255, 255), -1)

    if previous_positions:
        if len(previous_flow_rate) < len(previous_positions):
            previous_flow_rate = [0.0] * len(previous_positions)

        for i, current_pos in enumerate(current_positions):
            if i < len(previous_positions):
                prev_pos = previous_positions[i]
                prev_flow = previous_flow_rate[i]
                current_flowrate = calculate_flow_rate_limit_flow_rate(dimension_of_flow='microliter per minute',length_um=500, width_um=200, pixel_mm=1.539e-6,prev_position=prev_pos, current_position=current_pos, frames_per_seconds=fps, prev_flow_rate=prev_flow, max_change=max_change_threshold)
                previous_flow_rate[i] = current_flowrate

                cv2.putText(frame, f'v={current_flowrate:.2f}ul/min', (current_pos[0] + 10, current_pos[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                current_time = frame_number / fps
                time_values.append(current_time)
                flow_rate_values.append(current_flowrate)

    previous_positions = current_positions
    frames.append(frame)
    maps.append(closing)
    out_flowrate.write(frame)
    out_mask.write(closing)
    time_values.append(len(frames) / fps)
    flow_rate_values.append(np.mean(previous_flow_rate) if previous_flow_rate else 0)

cap.release()
out_flowrate.release()
out_mask.release()

def nothing( ):
    pass

cv2.namedWindow('Video with Flow Rate')
cv2.resizeWindow('Video with Flow Rate', 640, 480)
cv2.createTrackbar('Frame', 'Video with Flow Rate', 0, len(frames) - 1, nothing)
cv2.namedWindow('Video with Binary Mask')
cv2.resizeWindow('Video with Binary Mask', 640, 480)

while True:
    frame_index = cv2.getTrackbarPos('Frame', 'Video with Flow Rate')
    frame = frames[frame_index]
    cv2.imshow('Video with Flow Rate', frame)

    frame_mask = maps[frame_index]
    cv2.imshow('Video with Binary Mask', frame_mask)

    # Press 'q' to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

final_time_values = []
final_flow_rate_values = []

previous_time = 0
previous_flow_rate = None


with open('Data/flow_rate_data_raw.csv', 'w', newline='') as csvfile:
    fieldnames = ['Time (s)', 'Flow Rate (uL/min)']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for time, flow_rate in zip(time_values, flow_rate_values):

        if time == 0.0:
            time = previous_time
        else:
            previous_time = time

        final_time_values.append(time)
        final_flow_rate_values.append(flow_rate)
        writer.writerow({'Time (s)': time, 'Flow Rate (uL/min)': flow_rate})

file_path = 'Data/flow_rate_data_raw.csv'
data = pd.read_csv(file_path)
time_median = [0.0]
flow_median = [0.0]
temp_flow = []
prev_time = None
median_flow_rate = data.groupby('Time (s)')['Flow Rate (uL/min)'].median().reset_index()


for index, row in data.iterrows():
    current_time = row['Time (s)']
    current_flow = row['Flow Rate (uL/min)']

    if prev_time is None:
        prev_time = current_time
    elif current_time == prev_time:
        if current_flow != 0.0 or current_flow != 0.0:
            temp_flow.append(current_flow)
    else:
        if current_flow != 0 or current_flow != 0.0:
            median_flow = np.nanmedian(temp_flow)
            flow_median.append(median_flow)
            time_median.append(prev_time)
        temp_flow = [current_flow]
        prev_time = current_time

if temp_flow:
    med_flow = np.nanmedian(temp_flow)
    time_median.append(prev_time)
    flow_median.append(med_flow)

mean_time_sec = []
mean_flow_sec = []
temp_mean_flow_sec = []
index = 1
rest = len(flow_median)%5
for row in range(0,len(flow_median)-rest):
    if row%5 == 0:
        mean_time_sec.append(index)
        index += 1
        mean_flow_sec.append(np.nanmedian(temp_mean_flow_sec))
        temp_mean_flow_min = []
    temp_mean_flow_sec.append(flow_median[row])

mean_time_min = []
mean_flow_min = []
temp_mean_flow_min = []
index = 0

for row in range(0,len(mean_flow_sec)):
    if row%60 == 0:
        mean_time_min.append(index)
        index += 60
        mean_flow_min.append(np.nanmedian(temp_mean_flow_min))
        temp_mean_flow_min = []
    temp_mean_flow_min.append(mean_flow_sec[row])


with open('Data/median_flow_rate_data.csv', 'w', newline='') as csvfile:
    fieldnames = ['Time (s)', 'Flow Rate (uL/min)']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for time, flow_rate in zip(mean_time_sec, mean_flow_sec):
        writer.writerow({'Time (s)': time, 'Flow Rate (uL/min)': flow_rate})

# Plot the median flow rate and the smoothed curve
plt.figure(figsize=(10, 6))

# Plot the median flow rate
plt.plot(mean_time_sec, mean_flow_sec, label='Flow Rate [ul/min]', marker='o', linestyle='-', color='blue',alpha=0.6)

plt.plot(mean_time_sec, flowrate_calculation_ul_per_min[:len(mean_time_sec)], label='Calculated Flow Rate [ul/min]', color='red')


plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Flow Rate (uL/min)', fontsize=12)
plt.title('Median Flow Rate Over Time with Smoothing', fontsize=14)
plt.grid(True)
plt.legend()

plt.show()

