from picamera2 import Picamera2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import cv2
import tflite_runtime.interpreter as tflite
import RPi.GPIO as GPIO
import time, datetime, csv
from collections import deque, Counter

# Panel boundary config
PANEL_LENGTH = 60.0  # cm
PANEL_WIDTH = 60.0   # cm

# Robot position tracking (start at center or any (0,0) reference)
robot_x = PANEL_LENGTH / 2
robot_y = PANEL_WIDTH / 2

# For demo, assume every move_forward = +5cm on y, move_backward = -5cm, left/right on x-axis
MOVE_STEP = 5.0  # cm

# Prediction smoothing config
VOTE_LEN = 5
pred_history = deque(maxlen=VOTE_LEN)

# GPIO Assign
IN1, IN2, IN3, IN4 = 17, 18, 22, 23
BRUSH_IN1, BRUSH_IN2 = 5, 6
PUMP_IN3, PUMP_IN4 = 13, 19
TRIG, ECHO = 20, 21

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for pin in [IN1, IN2, IN3, IN4, BRUSH_IN1, BRUSH_IN2, PUMP_IN3, PUMP_IN4]:
    GPIO.setup(pin, GPIO.OUT)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

pwm_wheel_1 = GPIO.PWM(IN1, 100)
pwm_wheel_2 = GPIO.PWM(IN3, 100)
pwm_brush = GPIO.PWM(BRUSH_IN1, 100)
pwm_pump = GPIO.PWM(PUMP_IN3, 100)
pwm_wheel_1.start(0)
pwm_wheel_2.start(0)
pwm_brush.start(0)
pwm_pump.start(0)

def stop_all():
    pwm_wheel_1.ChangeDutyCycle(0)
    pwm_wheel_2.ChangeDutyCycle(0)
    pwm_brush.ChangeDutyCycle(0)
    pwm_pump.ChangeDutyCycle(0)
    for pin in [IN1, IN2, IN3, IN4, BRUSH_IN1, BRUSH_IN2, PUMP_IN3, PUMP_IN4]:
        GPIO.output(pin, GPIO.LOW)

def wheels_forward(speed=50):
    global robot_y
    # Move only if boundary not exceeded
    if robot_y + MOVE_STEP <= PANEL_WIDTH:
        GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
        pwm_wheel_1.ChangeDutyCycle(speed); pwm_wheel_2.ChangeDutyCycle(speed)
        robot_y += MOVE_STEP
        time.sleep(0.7)
        stop_all()
    else:
        stop_all()
def wheels_backward(speed=50):
    global robot_y
    if robot_y - MOVE_STEP >= 0:
        GPIO.output(IN1, GPIO.LOW); GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN3, GPIO.LOW); GPIO.output(IN4, GPIO.HIGH)
        pwm_wheel_1.ChangeDutyCycle(speed); pwm_wheel_2.ChangeDutyCycle(speed)
        robot_y -= MOVE_STEP
        time.sleep(0.7)
        stop_all()
    else:
        stop_all()
def wheels_left(speed=50):
    global robot_x
    if robot_x - MOVE_STEP >= 0:
        GPIO.output(IN1, GPIO.LOW); GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
        pwm_wheel_1.ChangeDutyCycle(speed); pwm_wheel_2.ChangeDutyCycle(speed)
        robot_x -= MOVE_STEP
        time.sleep(0.7)
        stop_all()
    else:
        stop_all()
def wheels_right(speed=50):
    global robot_x
    if robot_x + MOVE_STEP <= PANEL_LENGTH:
        GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.LOW); GPIO.output(IN4, GPIO.HIGH)
        pwm_wheel_1.ChangeDutyCycle(speed); pwm_wheel_2.ChangeDutyCycle(speed)
        robot_x += MOVE_STEP
        time.sleep(0.7)
        stop_all()
    else:
        stop_all()
def brush_on(speed=70):
    GPIO.output(BRUSH_IN1, GPIO.HIGH); GPIO.output(BRUSH_IN2, GPIO.LOW)
    pwm_brush.ChangeDutyCycle(speed)
def brush_off():
    pwm_brush.ChangeDutyCycle(0); GPIO.output(BRUSH_IN1, GPIO.LOW); GPIO.output(BRUSH_IN2, GPIO.LOW)
def pump_on(speed=70):
    GPIO.output(PUMP_IN3, GPIO.HIGH); GPIO.output(PUMP_IN4, GPIO.LOW)
    pwm_pump.ChangeDutyCycle(speed)
def pump_off():
    pwm_pump.ChangeDutyCycle(0); GPIO.output(PUMP_IN3, GPIO.LOW); GPIO.output(PUMP_IN4, GPIO.LOW)
def get_distance():
    GPIO.output(TRIG, False)
    time.sleep(0.05)
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)
    pulse_start = time.time(); pulse_end = time.time()
    while GPIO.input(ECHO) == 0: pulse_start = time.time()
    while GPIO.input(ECHO) == 1: pulse_end = time.time()
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    return round(distance, 2)

interpreter = tflite.Interpreter(model_path='solar_panel_cleaner.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
H, W = input_details[0]['shape'][1], input_details[0]['shape'][2]
labels = ["Clean", "Dusty", "Bird-drop", "Electrical-damage", "Physical-Damage"]
label_to_action = {
    "Clean": "No cleaning required",
    "Dusty": "Dry cleaning required",
    "Bird-drop": "Bird dropping dry and wet cleaning required",
    "Electrical-damage": "Maintenance required",
    "Physical-Damage": "Maintenance required"
}

picam2 = Picamera2()
picam2.start()

root = tk.Tk()
root.title("Solar Panel Live ML + GPIO + Panel Boundary + Smoothing")
panel = tk.Label(root)
panel.pack()
info = tk.Label(root, font=("Arial", 16))
info.pack()

# Time Series Log List
time_series_log = []

def update():
    dist = get_distance()
    frame = picam2.capture_array()
    img = cv2.resize(frame, (W, H))
    if img.shape[2] == 4: img = img[:, :, :3]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = np.expand_dims(img_rgb.astype(np.float32), axis=0)
    if input_details[0]['dtype'] == np.int8: x = x.astype(np.int8)
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    pred_index = np.argmax(output)
    raw_pred_label = labels[pred_index]

    # Smoothing: majority vote over last VOTE_LEN predictions
    pred_history.append(raw_pred_label)
    pred_label = Counter(pred_history).most_common(1)[0][0]

    action = label_to_action.get(pred_label, "Unknown")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time_series_log.append([now, pred_label, action, dist, robot_x, robot_y])

    info_text = (f"Prediction: {pred_label} (Smoothed)\n"
                 f"Action: {action}\n"
                 f"Distance: {dist:.1f} cm\n"
                 f"Position: X={robot_x:.1f}, Y={robot_y:.1f} cm")
    info.config(text=info_text)
    img_disp = ImageTk.PhotoImage(Image.fromarray(frame))
    panel.configure(image=img_disp)
    panel.image = img_disp

    # GPIO + BOUNDARY logic
    if dist and dist < 20:
        wheels_backward(speed=40)
        stop_all()
        root.after(500, update)
        return

    # All movement only within panel boundary
    if pred_label == "Clean":
        stop_all(); brush_off(); pump_off()
    elif pred_label == "Dusty":
        wheels_forward(speed=40); brush_on(speed=50); pump_off()
    elif pred_label == "Bird-drop":
        wheels_forward(speed=40); brush_on(speed=50); pump_on(speed=50)
    elif pred_label == "Electrical-damage":
        wheels_left(speed=40); stop_all(); brush_off(); pump_off()
    elif pred_label == "Physical-Damage":
        wheels_right(speed=40); stop_all(); brush_off(); pump_off()

    root.after(100, update)

def on_close():
    with open("panel_timeseries_log_withxy.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Prediction", "Action", "Distance_cm", "Robot_X_cm", "Robot_Y_cm"])
        writer.writerows(time_series_log)
    stop_all(); brush_off(); pump_off(); GPIO.cleanup(); picam2.stop(); root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
update()
root.mainloop()
