# Real-Time Expression Mapping & Digital Avatar
### **A Technical Implementation of Facial Landmark Detection & State Management**

## Project Overview
This project is a high-accuracy computer vision pipeline developed to map human facial expressions onto a digital avatar of a monkey in real-time.
## Technical Stack
* **Language:** Python
* **Computer Vision:** OpenCV (Real-time video processing)
* **Data Processing:** NumPy (Landmark coordinate mathematics)
* **Packaging:** PyInstaller (Standalone Executable generation)

## Key Technical Features

### **1. Threshold Counter Logic (Jitter Reduction)**
To solve the common "chatter" or visual jitter found in basic landmark detection, I engineered a **custom state-management buffer**. By utilizing a Threshold Counter, the system requires consistent detection across multiple frames before transitioning the avatar’s state, resulting in a significantly more stable and professional user experience.

### **2. Priority-Based Logic Hierarchy**
the system manages conflicting facial detections. For example, the logic prioritizes profile-view obstruction detection over "closed-eye" states to prevent accidental expression triggers during lateral head movement.

### **3. Optimized Performance & Accuracy**
* **Precision:** Achieved **95%+ detection accuracy** through fine-tuned confidence thresholds.
* **Performance:** Optimized the Python pipeline to maintain real-time frame rates on standard consumer webcams.

## Repository Structure
* `src/`: Core implementation scripts and detection logic.
* `assets/`: Landmark mapping guides and digital avatar assets.
* `dist/`: Standalone executable produced via PyInstaller for testing without a Python environment.

## 🔧 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/vatsss2/monkey-expression-live-camera-filter.git](https://github.com/vatsss2/monkey-expression-live-camera-filter.git)
