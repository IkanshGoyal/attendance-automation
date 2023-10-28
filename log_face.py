import cv2
import face_recognition
import json

known_faces = {}
known_face_encodings = []
known_face_names = []

name = ""

try:
    with open("known_faces.json", "r") as json_file:
        known_faces = json.load(json_file)
except FileNotFoundError:
    known_faces = {}

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
    if face_locations:
        face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]

        # Check if the face matches any known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            matched_idx = matches.index(True)
            name = known_face_names[matched_idx]

        # Draw a rectangle and label on the frame
        for top, right, bottom, left in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Face Recognition", frame)

    # Capture and log an image when the 's' key is pressed
    key = cv2.waitKey(1)
    if key == ord('s'):
        name = input("Enter the name of the person: ")
        known_faces[name] = face_encoding.tolist()  # Convert the encoding to a list
        known_face_names.append(name)
        known_face_encodings.append(face_encoding)

    # Exit the loop by pressing 'q'
    if key == ord('q'):
        break

# Save the known_faces dictionary to a JSON file
with open("known_faces.json", "w") as json_file:
    json.dump(known_faces, json_file)

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
