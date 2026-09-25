!pip install -q face_recognition opencv-python-headless fpdf2

import cv2
import os
import numpy as np
import face_recognition
import pandas as pd
from datetime import datetime
from google.colab import files
import matplotlib.pyplot as plt

print("="*50)
print("Running on CPU - HOG Model")
print("="*50)

attendance_file = "Attendance.csv"
with open(attendance_file, 'w') as f:
    f.write("Name,Time,Date,Status\n")
print(f"Created {attendance_file}")

def mark_attendance(name):
    with open(attendance_file, 'r+') as f:
        lines = f.readlines()
        name_list = [line.split(',')[0] for line in lines]
        if name not in name_list:
            now = datetime.now()
            time_str = now.strftime('%H:%M:%S')
            date_str = now.strftime('%d/%m/%Y')
            f.writelines(f"\n{name},{time_str},{date_str},Present")
            print(f"✅ Attendance Marked: {name} at {time_str}")

                     print("\n📌 STEP 1: Upload KNOWN face images")
print("Rename files like: Ramesh.jpg, Priya.jpg before upload")
uploaded_known = files.upload()

known_encodings = []
known_names = []

for file_name in uploaded_known.keys():
    name = os.path.splitext(file_name)[0] # File name = Person name
    print(f"\nProcessing {file_name}...")

    # Load image
    img = face_recognition.load_image_file(file_name)
    # Encode - HOG model (CPU fast)
    encodings = face_recognition.face_encodings(img)

    if len(encodings) > 0:
        known_encodings.append(encodings[0])
        known_names.append(name)
        print(f"✅ Learned: {name}")
    else:
        print(f"❌ No face found in {file_name} - Use clear front photo")

print(f"\nTotal Faces Learned: {known_names}")

print("\n📌 STEP 2: Upload TEST image (group photo for attendance)")
uploaded_test = files.upload()
test_path = list(uploaded_test.keys())[0]

# STEP 5: FACE RECOGNITION - CPU (HOG)
print("\n--- Detecting Faces on CPU (HOG) ---")
test_image = face_recognition.load_image_file(test_path)

# HOG = CPU model, fast and light
face_locations = face_recognition.face_locations(test_image, model="hog")
face_encodings_test = face_recognition.face_encodings(test_image, face_locations)

print(f"Total Faces Detected: {len(face_locations)}")

# Draw boxes
test_image_cv = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings_test):
    # Compare
    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
    face_distances = face_recognition.face_distance(known_encodings, face_encoding)

    name = "Unknown"
    if len(face_distances) > 0:
        best_index = np.argmin(face_distances)
        if matches[best_index]:
            name = known_names[best_index]
            mark_attendance(name)

    # Draw
    cv2.rectangle(test_image_cv, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.rectangle(test_image_cv, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
    cv2.putText(test_image_cv, name, (left+4, bottom-6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0), 1)

# Show output
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(test_image_cv, cv2.COLOR_BGR2RGB))
plt.title(f"Attendance Result - {len(face_locations)} Faces")
plt.axis('off')
plt.show()
                     print("\n--- FINAL ATTENDANCE SHEET ---")
try:
    df = pd.read_csv(attendance_file)
    print(df)
    df.to_csv("Final_Attendance.csv", index=False)
    files.download("Final_Attendance.csv")
except:
    print("No known faces matched")
