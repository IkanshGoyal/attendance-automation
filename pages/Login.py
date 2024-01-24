import cv2
import face_recognition
import json
import os
import numpy as np 
from datetime import datetime 
import streamlit as st

st.title("Attendance Web App")
st.subheader("Login Page", divider = "grey")


def markAttendance (name, id):
    with open( 'Attendance.csv', 'r+') as f:
        myDataList = f.readlines()

        now = datetime.now()
        time = now.strftime( '%I:%M:%S:%p')
        date = now.strftime ('%d-%B-%Y')

        f.writelines(f'\n{name}, {id}, {time}, {date}')

        str = "Attendance marked, " + name
        st.success(str)


def att():
    with open("known_faces.json", "r") as json_file:
        known_faces = json.load(json_file)

    known_face_encodings = list(known_faces.values())  
    known_face_names = list(known_faces.keys())  

    cap = cv2.VideoCapture(0)  

    while True:
        ret, frame = cap.read()
        f = 0

        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame, model='hog')

        for face_location in face_locations:
            face_encoding = face_recognition.face_encodings(rgb_frame, [face_location])[0]

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown" 
            found = False

            for j, is_match in enumerate(matches):
                if is_match:
                    name = known_face_names[j]
                    n, user_id = name.split(",")
                    markAttendance(n, user_id)
                    found = True
                    f = 1
                    break

            if not found:
                st.error("User not found")
                f = 1

        if f == 1:
            break

    cap.release()
    cv2.destroyAllWindows()


user_name = st.text_input("Full Name")
user_id = st.text_input("College Id")
submit_btn = st.button("Submit")
att_btn = st.button("Recognise")

if submit_btn:
    markAttendance(user_name, user_id)

if att_btn:
    att()