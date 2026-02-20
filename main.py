import cv2
import os
import sys
import numpy as np

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def main():
    TEST_MODE = False

    CONFIRMATION = {
        'smile': 5, 
        'think': 3, 
        'turned_head': 6, 
        'neutral': 2
    }

    print("Loading cascade classifiers...")
    cascades = {
        'frontal': cv2.CascadeClassifier(resource_path(os.path.join('cascades', 'haarcascade_frontalface_default.xml'))),
        'profile': cv2.CascadeClassifier(resource_path(os.path.join('cascades', 'haarcascade_profileface.xml'))),
        'smile': cv2.CascadeClassifier(resource_path(os.path.join('cascades', 'haarcascade_smile.xml'))),
        'left_eye': cv2.CascadeClassifier(resource_path(os.path.join('cascades', 'haarcascade_lefteye_2splits.xml'))),
        'right_eye': cv2.CascadeClassifier(resource_path(os.path.join('cascades', 'haarcascade_righteye_2splits.xml')))
    }
    if any(c.empty() for c in cascades.values()):
        sys.exit("Could not load one or more cascade classifiers.")

    print("Initializing webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): sys.exit("Cannot open camera.")
    success, frame = cap.read()
    if not success: sys.exit("Could not read from webcam.")
    frame_h, frame_w, _ = frame.shape

    print("Loading avatar images...")
    avatar_images = {}
    try:
        image_names = ['neutral', 'smile', 'think', 'turned_head']
        for name in image_names:
            path = resource_path(os.path.join('images', f'{name}.png'))
            img = cv2.imread(path)
            if img is None: raise FileNotFoundError(f"{name}.png not found")
            avatar_images[name] = cv2.resize(img, (frame_w, frame_h))
    except Exception as e:
        sys.exit(f"Could not load avatar images. Error: {e}")

    print("\n Application started! Press 'q' to quit.")
    current_state = 'neutral'
    counters = {name: 0 for name in image_names}
    
    while True:
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        frontal_faces = cascades['frontal'].detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        profile_faces = cascades['profile'].detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        
        expression_this_frame = None

        if len(frontal_faces) > 0:
            (x, y, w, h) = sorted(frontal_faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            if TEST_MODE: cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            face_roi_gray = gray_frame[y:y+h, x:x+w]

            smiles = cascades['smile'].detectMultiScale(face_roi_gray, scaleFactor=1.7, minNeighbors=22)
            left_eyes = cascades['left_eye'].detectMultiScale(face_roi_gray, scaleFactor=1.2, minNeighbors=10)
            right_eyes = cascades['right_eye'].detectMultiScale(face_roi_gray, scaleFactor=1.2, minNeighbors=10)
            
            has_smile = len(smiles) > 0
            has_no_eyes = len(left_eyes) == 0 and len(right_eyes) == 0
            has_both_eyes = len(left_eyes) > 0 and len(right_eyes) > 0

            if has_smile:
                expression_this_frame = 'smile'
            elif has_no_eyes:
                expression_this_frame = 'think'
            elif has_both_eyes:
                expression_this_frame = 'neutral'
        elif len(profile_faces) > 0:
            expression_this_frame = 'turned_head'
        if expression_this_frame is not None:
            for state in counters:
                counters[state] = counters[state] + 1 if state == expression_this_frame else 0
        new_state = current_state
        for state, threshold in CONFIRMATION.items():
            if counters[state] > threshold:
                new_state = state
                break
        current_state = new_state
             
        avatar_frame = avatar_images[current_state]

        if TEST_MODE:
            output_frame = np.hstack((frame, avatar_frame))
        else:
            output_frame = avatar_frame

        cv2.imshow('Digital Avatar', output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    main()
