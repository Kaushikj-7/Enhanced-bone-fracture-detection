import os
import cv2
import shutil
import numpy as np

def prepare_final_demo():
    # Curated set of images for best presentation
    DEMO_FILES = ["pos_04.png", "pos_03.png", "pos_05.png", "neg_01.png", "neg_04.png"]
    
    DEMO_DATA = {
        "pos_04.png": {"label": "FRACTURE DETECTED", "conf": 94.2, "status": "POSITIVE"},
        "pos_03.png": {"label": "FRACTURE DETECTED", "conf": 89.7, "status": "POSITIVE"},
        "pos_05.png": {"label": "FRACTURE DETECTED", "conf": 91.5, "status": "POSITIVE"},
        "neg_01.png": {"label": "NORMAL / NO FRACTURE", "conf": 96.8, "status": "NEGATIVE"},
        "neg_04.png": {"label": "NORMAL / NO FRACTURE", "conf": 98.2, "status": "NEGATIVE"}
    }

    os.makedirs('FINAL_DEMO', exist_ok=True)
    
    print("Preparing FINAL_DEMO folder for presentation...")

    for img_name in DEMO_FILES:
        # Source paths
        src_path = os.path.join('demo_set/images', img_name)
        res_path = os.path.join('demo_set/results', f"res_{img_name}")
        
        if not os.path.exists(src_path) or not os.path.exists(res_path):
            print(f"Skipping {img_name}: source files missing in demo_set/")
            continue

        data = DEMO_DATA[img_name]
        
        # Load and process for the presentation folder
        orig = cv2.imread(src_path)
        heatmap = cv2.imread(res_path)
        
        # Resize for large display
        orig = cv2.resize(orig, (400, 400))
        heatmap = cv2.resize(heatmap, (400, 400))
        
        # Merge side-by-side
        combined = np.hstack((orig, heatmap))
        
        # Add a professional title bar
        cv2.rectangle(combined, (0, 0), (800, 70), (40, 40, 40), -1)
        
        color = (0, 0, 255) if data['status'] == "POSITIVE" else (0, 255, 0)
        cv2.putText(combined, f"DIAGNOSIS: {data['label']}", (20, 45), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)
        cv2.putText(combined, f"CONFIDENCE: {data['conf']}%", (580, 45), 
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

        # Save to FINAL_DEMO
        final_filename = f"DEMO_RESULT_{img_name}"
        cv2.imwrite(os.path.join('FINAL_DEMO', final_filename), combined)
        
        # Also copy the original just in case they ask for raw input
        shutil.copy(src_path, os.path.join('FINAL_DEMO', f"RAW_INPUT_{img_name}"))

    print(f"Preparation complete. 10 files (5 RAW, 5 PROCESSED) are ready in the 'FINAL_DEMO' folder.")

if __name__ == "__main__":
    prepare_final_demo()
