# Real-Time Expression Mapping & Digital Avatar
### **A Multi-Cascade Implementation of Facial Landmark Detection & State Management**

## Project Overview
This project is a high-accuracy computer vision pipeline developed to map human facial expressions onto a digital avatar of a monkey in real-time. The system utilizes coordinate-based detection to translate facial gestures into animated responses.

## Technical Stack
* **Language:** Python
* **Computer Vision:** OpenCV (utilizing multiple Haar Cascade Classifiers)
* **Data Processing:** NumPy (Landmark coordinate mathematics)
* **Packaging:** PyInstaller (Standalone Executable generation with `resource_path` handling)

## Key Technical Features

### **1. Stability Buffer (Confirmation Logic)**
To solve the common "chatter" or visual jitter found in basic detection, I engineered a **CONFIRMATION state-management buffer**. By assigning specific frame-count thresholds (e.g., `smile: 5`, `turned_head: 6`), the system requires consistent detection across multiple frames before transitioning the avatar’s state. This results in a significantly more stable and professional user experience.

### **2. Simplified Logic Hierarchy**
Using **Object-Oriented Programming (OOP)** principles, the system manages a simultaneous multi-cascade pipeline. The logic follows a strict priority hierarchy:
1. **Frontal Smile Detection**: Triggers the 'smile' state.
2. **Eye Detection**: If a face is present but eyes are not detected (closed), it triggers the 'think' state.
3. **Profile Detection**: If no frontal face is found, the system defaults to 'turned_head'.

### **3. Optimized Performance & Deployment**
* **Multi-Cascade Support:** Simultaneously runs `frontal`, `profile`, `smile`, and `eye` classifiers to resolve expression states.
* **Resource Management:** Implemented a `resource_path` helper function to ensure asset paths (images/cascades) remain valid after compilation into a standalone executable.

## Repository Structure
* `src/main.py`: Core implementation scripts and detection logic.
* `cascades/`: XML files for Haar Cascade Classifiers.
* `images/`: Custom digital avatar assets ('neutral', 'smile', 'think', 'turned_head').
* `main.spec`: Configuration for PyInstaller standalone deployment.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/vatsss2/monkey-expression-live-camera-filter.git](https://github.com/vatsss2/monkey-expression-live-camera-filter.git)

2. **install dependencies**
   ```bash
   pip install -r requirements.txt

3. **Run the application**
   ```bash
   python src/main.py
