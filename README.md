# GestureIQ — AI-Powered Virtual Mouse using OpenCV & MediaPipe

GestureIQ is a real-time gesture-controlled virtual mouse system built using Python, OpenCV, MediaPipe, and PyAutoGUI.

The project enables touchless human-computer interaction through hand gesture recognition using a webcam. Users can control cursor movement, perform clicks, and trigger system actions entirely using hand gestures.

---

# Features

- Real-time hand tracking using MediaPipe
- Gesture-based cursor movement
- Left click gesture
- Right click gesture
- Double click gesture
- Screenshot trigger gesture
- Cursor smoothing for stable tracking
- Kalman Filter-based movement prediction
- FPS monitoring
- Real-time UI feedback system

---

# Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- PyAutoGUI

---

# Project Architecture

Webcam Feed  
↓  
OpenCV Frame Processing  
↓  
MediaPipe Hand Landmark Detection  
↓  
Gesture Recognition Logic  
↓  
Kalman Filter Prediction + Cursor Smoothing  
↓  
Mouse Control using PyAutoGUI  

---

# Gesture Controls

| Gesture | Action |
|----------|---------|
| All fingers pinched | Cursor Movement |
| Index finger up | Left Click |
| Middle finger up | Right Click |
| Pinky finger up | Double Click |
| Ring finger up | Screenshot |

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/your-username/GestureIQ.git
cd GestureIQ
