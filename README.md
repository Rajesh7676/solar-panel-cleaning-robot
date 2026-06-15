# Solar Panel Cleaning and Maintenance Robot

Selected for KSCST State-Level Exhibition 2025-26, Karnataka State Council for Science and Technology, IISc Bengaluru.

## Problem

Solar panels lose efficiency over time due to dust accumulation, bird droppings, and physical or electrical damage. Manual inspection across large installations is slow and expensive, and minor damage often goes unnoticed until output drops significantly.

## What this project does

This is an autonomous robot that moves across a solar panel, continuously captures images, and classifies the panel's condition using a CNN model running on a Raspberry Pi. Based on the classification, it decides whether to clean the panel (dry brushing or wet cleaning) or stop and flag the area for maintenance.

## System Flow

1. Camera captures a frame of the panel surface
2. CNN model classifies the frame into one of five categories
3. Predictions are smoothed over the last few frames to avoid acting on a single noisy reading
4. Based on the smoothed prediction, the robot either:
   - moves on (panel is clean)
   - runs the brush for dust
   - runs brush and pump for bird droppings
   - stops and reports maintenance required for damage
5. Robot position, distance travelled, and predictions are logged continuously for later analysis

## Hardware

- Raspberry Pi 4B
- Pi Camera v2 (8MP)
- Two 12V DC gear motors with L298N driver for movement
- 12V diaphragm pump for wet cleaning
- Brush motor for dry cleaning
- Ultrasonic sensor for obstacle/edge detection
- 12V 7Ah solar-rechargeable battery

## Model

- CNN, MobileNet-based, converted to TensorFlow Lite for on-device inference
- Around 60ms per frame on the Raspberry Pi 4
- Trained on roughly 1500 images across five classes: Clean, Dusty, Bird-drop, Electrical-damage, Physical-Damage
- Predictions are smoothed using a majority vote over the last five frames, which reduces false triggers from single bad frames

## Decision Logic

| Prediction | Action |
|---|---|
| Clean | Continue moving, no cleaning |
| Dusty | Dry brush only |
| Bird-drop | Brush plus pump for 5 seconds |
| Electrical-damage | Stop motors, move clear of the spot, show maintenance alert |
| Physical-Damage | Stop motors, move clear of the spot, show maintenance alert |

## Safety and movement logic

The robot tracks its own position (X, Y) within the panel boundary and won't drive itself off the edge. An ultrasonic sensor checks for obstacles and reverses the robot if something is too close. When damage is detected, the robot moves slightly away from that spot before stopping, so it doesn't keep sitting on a damaged area.

## Logging and analysis

Every cycle, the system logs the time, prediction, action taken, distance travelled, and current position to a CSV file. A separate analysis script reads this log and generates:

- a plot of distance travelled over time
- a histogram of distance values
- summary statistics (mean, median, min, max, standard deviation)

This was used to study the robot's movement patterns over a full run.

## Background

This was originally a four-person final year B.E. project (AI and ML branch), but the hardware design, assembly, model training, software integration, and deployment were done individually. The hardware went through three iterations - Arduino, then ESP32, then Raspberry Pi 4 - before settling on the final design. One of the more time-consuming issues was getting the Pi camera working after a libcamera update broke compatibility with the OV5647 sensor, which took close to a month to resolve.

A paper based on this work was presented at an IFERP conference.

## Tech stack

Python, TensorFlow Lite, OpenCV, Picamera2, Tkinter, RPi.GPIO, Pandas, Matplotlib
