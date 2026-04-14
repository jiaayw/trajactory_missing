Project **SIGHTBOT**
CV-Based Obstacle Detection & Marker Navigation Robot

1. Product Overview
1.1 Objective
Build a vision-based mobile robot that can:
Navigate using camera-based perception
Avoid obstacles in real time
Detect and approach a target object (yellow cube)
Confirm destination via visual marker (ArUco / QR / barcode)
Stop safely at the destination

1.2 System Summary
The system is a modular robotics pipeline:
Pi Camera → CV Processing (Raspberry Pi) → Decision Logic → Serial → Arduino Uno → Motor Driver → Motors

2. Goals & Success Criteria
2.1 Primary Goals
Real-time obstacle avoidance
Reliable detection of yellow target object
Marker-based destination confirmation
Stable movement control via Arduino
Safe stopping behavior
2.2 Success Criteria
The system is successful if:
Robot moves correctly using Pi commands
Obstacles are avoided in basic environments
Yellow cube is consistently detected
Marker is correctly identified
Robot approaches and stops at target
System fails safely under errors

3. System Architecture
3.1 Hardware Components
Component
Role
Raspberry Pi 4
Vision + decision making
Pi Camera
Visual input
Arduino Uno R3
Motor control
L298N Driver
Motor actuation
4 DC Motors
Locomotion (2-channel grouped)
LEDs (diffused)
Controlled lighting
Battery Pack
Motor power

Optional:
Ultrasonic sensor (for precise stopping)

3.2 Software Components
Module
Function
Camera Stream
Capture frames
Obstacle Detector
Detect obstacles
Yellow Detector
Find target cube
Marker Reader
Identify destination
State Machine
Control behavior
Serial Controller
Send commands to Arduino


4. Functional Requirements
4.1 Motion Control
The robot must support:
Command
Action
F
Move forward
B
Move backward
L
Turn left
R
Turn right
S
Stop


4.2 Obstacle Detection
Use classical CV (no ML)
Analyze lower half of image
Split into left/center/right zones
Detect blockage and avoid accordingly

4.3 Target Detection
Detect yellow cube using HSV thresholding
Identify largest valid contour
Track centroid for steering

4.4 Marker Confirmation
Detect marker on cube (ArUco / QR / barcode)
Confirm target identity before final approach

4.5 Navigation Behavior
Condition
Action
Target centered
Move forward
Target left
Turn left
Target right
Turn right
Obstacle ahead
Avoid
Target close
Stop


5. Non-Functional Requirements
5.1 Performance
Real-time processing (~10–20 FPS acceptable)
Stable behavior over speed
Minimal oscillation
5.2 Reliability
Works under controlled lighting
Robust to minor variations in environment
5.3 Safety
Arduino stops motors on timeout
Pi stops robot on error conditions

6. Lighting Requirements
6.1 LED Setup
Diffused LEDs mounted near camera
Even illumination across scene
Avoid direct glare
6.2 Camera Configuration
Fixed exposure
Fixed white balance
Consistent brightness

7. Motor System Design
7.1 Configuration
4 motors grouped into 2 channels:
Left side (2 motors)
Right side (2 motors)
7.2 Control Model
Differential drive system

8. Communication Protocol
8.1 Interface
Serial communication (USB)
8.2 Format
Single-character commands with newline
Example:
F\n
L\n
S\n

9. System Behavior (State Machine)
States:
IDLE
SEARCH_TARGET
APPROACH_TARGET
CONFIRM_MARKER
AVOID_OBSTACLE
STOP_AT_TARGET
FAILSAFE

10. Calibration Requirements
No machine learning is used.
System requires manual calibration:
HSV color thresholds (yellow detection)
Obstacle sensitivity thresholds
Target size for stopping condition

11. Constraints
11.1 Technical Constraints
Single RGB camera (no depth sensing)
Arduino Uno memory limitations
Limited processing power (Raspberry Pi)
11.2 Environmental Constraints
Works best in controlled lighting
Sensitive to reflections and shadows

12. Risks & Limitations
Known Limitations
No precise depth estimation
Marker detection may fail at distance
Performance affected by lighting changes
Risks
False obstacle detection
Drift due to motor imbalance
Camera blur during motion

13. Testing Plan
Phase 1: Hardware Validation
Motor control test
Serial communication test
Phase 2: Vision Validation
Camera feed
Yellow detection
Marker detection
Phase 3: Integration
Combine CV + motion
Tune thresholds
Phase 4: Full System Test
Autonomous navigation
Obstacle avoidance + target reaching

14. Future Enhancements
Ultrasonic sensor integration
ArUco marker upgrade
Wheel encoders
PID motor control
SLAM / mapping (advanced)

15. Deliverables
Arduino motor control script (.ino)
Raspberry Pi Python codebase
Configuration file
Documentation (setup + calibration)
Test scripts

16. Key Design Principle
Keep the system simple, modular, and explainable.
This is a baseline robotics system, not a production-grade autonomous platform.

17. Summary
This project demonstrates a complete robotics pipeline:
perception (camera + CV)
decision-making (rules + state machine)
actuation (Arduino + motors)
The system is designed to be:
easy to build
easy to debug
easy to extend

Final Note
This PRD defines a working, end-to-end robotics system without machine learning, relying on controlled environments and rule-based perception.
