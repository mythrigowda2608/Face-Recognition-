📌 Overview
This project implements an AI-powered attendance system using face recognition. It leverages the HOG (Histogram of Oriented Gradients) model for CPU-based face detection, making it lightweight and efficient. The system allows you to upload known face images (students/employees), then process a group photo to automatically mark attendance.
⚙️ Requirements
pip install -q face_recognition opencv-python-headless fpdf2
🚀 Workflow
Step 1: Initialize Attendance File
Name,Time,Date,Status
Step 2: Upload Known Faces
Upload clear, front-facing images of individuals.

Rename files as Name.jpg (e.g., Ramesh.jpg, Priya.jpg).

The system encodes these faces and stores them for recognition.

Step 3: Upload Test Image
Upload a group photo containing multiple faces.

The system detects faces using the HOG model and compares them with known encodings.

Step 4: Mark Attendance
If a match is found:

Attendance is marked with Name, Time, Date, Status=Present.

Duplicate entries are avoided.

Unknown faces are labeled as "Unknown".
Step 5: Visualization
Bounding boxes and names are drawn on the group photo.

The processed image is displayed using matplotlib.

Step 6: Final Attendance Sheet
Attendance is saved in Final_Attendance.csv.

File is available for download in Colab.
📂 Input & Output
Inputs
Known face images: Individual photos (e.g., Ramesh.jpg, Priya.jpg).

Test image: Group photo for attendance marking.

Outputs
Processed group photo: With bounding boxes and names.

Attendance CSV files:

Attendance.csv → Raw attendance log.

Final_Attendance.csv → Final sheet for download.
