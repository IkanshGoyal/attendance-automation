import cv2
import face_recognition
import json
import os
import numpy as np 
from datetime import datetime 
import pickle

def markAttendance (name):
    with open( 'Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList. append(entry[0])
        if name not in nameList:
            now = datetime. now()
            time = now.strftime( '%I:%M:%S:%p')
            date = now. strftime ('%d-%B-%Y')
            f.writelines(f'\n{name}, {time}, {date}')

# Load known faces from a JSON file
with open("known_faces.json", "r") as json_file:
    known_faces = json.load(json_file)

known_face_encodings = list(known_faces.values())  # List of face encodings
known_face_names = list(known_faces.keys())  # List of corresponding names

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 represents the default camera, change to the appropriate camera index if needed

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame from BGR (OpenCV ordering) to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Locate faces in the frame using face_recognition
    face_locations = face_recognition.face_locations(rgb_frame, model='hog')

    # If a face is detected, process it
    for i, face_location in enumerate(face_locations):
        face_encoding = face_recognition.face_encodings(rgb_frame, [face_location])[0]

        # Compare the current encoding to known encodings
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"  # Default name if no match is found

        for j, is_match in enumerate(matches):
            if is_match:
                name = known_face_names[j]
                markAttendance(name)
                exit()

        # Draw a rectangle and label on the frame
        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Face Recognition", frame)

    # Exit the loop by pressing 'q'
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
