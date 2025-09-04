import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import os

# image path (use raw string or forward slashes)
image_path = r'data\test2.png'  # ✅ Correct way

if not os.path.exists(image_path):
    raise FileNotFoundError(f"❌ Image not found at: {image_path}")

# Read image with OpenCV
img = cv2.imread(image_path)

if img is None:
    raise ValueError("❌ Failed to load image. Check path or file integrity.")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Detect text (you can pass either img or image_path)
text_ = reader.readtext(img)  

threshold = 0.25

# Draw bounding boxes and text
for t in text_:
    bbox, text, score = t
    print(f"Detected: '{text}' (score: {score:.2f})")

    if score > threshold:
        # Convert bbox points to integers
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))

        # Draw bounding box
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

        # Put detected text
        cv2.putText(img, text, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 2)

# Show result
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
