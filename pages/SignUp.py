import cv2
import face_recognition
import json
import os
import numpy as np 
from datetime import datetime 
import streamlit as st
import csv

st.title("Attendance Web App")
st.subheader("Sign-Up Page", divider = "grey")

user_name = st.text_input("Full Name", placeholder="Enter Name...")
user_id = st.text_input("College Id", placeholder="Enter College Id...")
user_roll = st.text_input("Class Roll No", placeholder="Enter Class Roll No...")
user_sec = st.text_input("Section", placeholder="Enter Section...")
user_year = st.number_input("Year", value=0, placeholder="Enter Year...")

submit_btn = st.button('Submit')
run = st.button('Log Face')


def log_face(user_name, user_id):
    known_faces = {}
    known_face_encodings = []
    known_face_names = []
    name = ""

    try:
        with open("known_faces.json", "r") as json_file:
            known_faces = json.load(json_file)
    except FileNotFoundError:
        known_faces = {}

    cap = cv2.VideoCapture(0)  

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame, model='hog')

        if face_locations:
            face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                matched_idx = matches.index(True)
                name = known_face_names[matched_idx]

            for top, right, bottom, left in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        FRAME_WINDOW.image(frame)

        if face_locations:
            try:
                name = "" + user_name + "," + user_id
                known_faces[name] = face_encoding.tolist()  
                known_face_names.append(name)
                known_face_encodings.append(face_encoding)

                try:
                    with open("known_faces.json", "w") as json_file:
                        json.dump(known_faces, json_file)
                        str = "Face Logged, " + user_name
                        st.success(str)
                except Exception as e:
                    st.error(f"Error writing to JSON file: {str(e)}")

                cap.release()
                cv2.destroyAllWindows()
                return
            except Exception as e:
                st.error(f"Error while saving face: {str(e)}")
    

def log_user(user_name, user_id, user_roll, user_sec, user_year):
    new_data = f"{user_name}, {user_id}, {user_roll}, {user_sec}, {user_year}\n"

    if not os.path.isfile('Users.csv'):
        with open('Users.csv', 'w') as f:
            f.write("Name, ID, Roll, Section, Year\n")
            f.write(new_data)
    else:
        with open('Users.csv', 'a') as f:
            f.write(new_data)


if submit_btn:
    st.success("You are now logged in!")
    log_user(user_name, user_id, user_roll, user_sec, user_year)

FRAME_WINDOW = st.image([])

if run:
    with st.container():
        st.write("Webcam Live Feed")
        log_face(user_name, user_id)