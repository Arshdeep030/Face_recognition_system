from tkinter import *
from tkinter import ttk
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2 
import numpy as np
import mysql.connector
from time import strftime
from datetime import datetime
from datetime import datetime, timedelta
import json
# Global dictionary to track the last attendance time for each person
last_attendance_time = {}

class FaceRecognition:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1366x768+0+0")
        self.root.title("Face Recognition System")

        title_lbl = Label(
            self.root,
            text="💻 FACE RECOGNITION 💻",
            font=("times new roman", 28, "bold"),
            bg="white",
            fg="#2E8B57",
            relief="groove",
            bd=3,
            padx=10,
            pady=5
        )
        title_lbl.place(x=0, y=0, width=1366, height=42)

        # Left Image
        img_left = Image.open("images/face_detector1.jpg")
        img_left = img_left.resize((660, 610), Image.Resampling.LANCZOS)
        self.photoimg_left = ImageTk.PhotoImage(img_left)

        f_lbl = Label(self.root, image=self.photoimg_left)
        f_lbl.place(x=0, y=42, width=660, height=610)

        # Right Image
        img_right = Image.open("images/facial_recognition_right_image.jpg")
        img_right = img_right.resize((750, 610), Image.Resampling.LANCZOS)
        self.photoimg_right = ImageTk.PhotoImage(img_right)

        f_lbl = Label(self.root, image=self.photoimg_right)
        f_lbl.place(x=620, y=42, width=750, height=610)

        # Face Recognition Button
        btn = Button(f_lbl, text="Face Recognition", cursor="hand2", command=self.face_recog,
                     font=("times new roman", 14, "bold"), bg="darkgreen", fg="white")
        btn.place(x=270, y=540, width=200, height=35)



    def show_notification(self, message):
        # Function to display a notification
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        messagebox.showinfo("Attendance Status", message)
        root.destroy()

    def mark_attendance(self, d, n, s, r):
        current_time = datetime.now()  # Ensure current_time is always assigned
        
        # Check if the person has marked attendance before
        if d in last_attendance_time:
            last_time = last_attendance_time[d]
            time_diff = current_time - last_time

            # If the time difference is less than 30 minutes, prevent marking attendance
            if time_diff < timedelta(minutes=30):
                print(f"Cannot mark attendance for {d} again. Last attendance was {time_diff} ago.")
                return  # Exit the function without adding a new row
            else:
                print(f"Allowed to mark attendance for {d}, 30 minutes have passed.")
        
        # Proceed with adding the new attendance only if it's not already recorded in the file
        with open("attendance.csv", "a+", newline="\n") as f:
            f.seek(0)
            myDataList = f.readlines()
            name_list = [line.split(",")[0] for line in myDataList]  # Extract names from the file

            if d not in name_list:
                dtString = current_time.strftime("%H:%M:%S")
                date = current_time.strftime("%d/%m/%Y")
                f.write(f"{d},{n},{s},{r},{dtString},{date},Present\n")
                
                # Update last attendance time after marking attendance
                last_attendance_time[d] = current_time  # This updates the time for this person
                print(f"Attendance for {d} marked successfully at {dtString} on {date}.")
                self.show_notification(f"Attendance for {n} ({d}) marked successfully!")  # Updated line
            else:
                print(f"Attendance for {d} already recorded. Skipping.")


    def face_recog(self):
        def draw_boundary(img, classifier, clf):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = classifier.detectMultiScale(gray, 1.1, 10)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                id, confidence = clf.predict(gray[y:y+h, x:x+w])
                confidence = 100 - (confidence / 3)

                conn = mysql.connector.connect(
                    host="localhost", user="root", password="sidhu1234@deep", database="face_recognizer"
                )
                my_cursor = conn.cursor()
                my_cursor.execute(f"SELECT student_id, student_name, roll_no, department FROM student WHERE student_id={id}")
                result = my_cursor.fetchone()
                conn.close()

                if result and confidence > 60:
                    d, n, s, r = result
                    self.mark_attendance(d, n, s, r)
                    cv2.putText(img, f"ID: {d}", (x, y - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(img, f"Name: {n}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(img, f"Roll No: {s}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                else:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(img, "Unknown", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        if faceCascade.empty():
            print("Error loading haarcascade_frontalface_default.xml")
            return

        clf = cv2.face.LBPHFaceRecognizer_create()
        try:
            clf.read("classifier.xml")
        except cv2.error:
            print("Error loading classifier.xml")
            return

        video_capture = cv2.VideoCapture(0)

        while True:
            ret, img = video_capture.read()
            if not ret:
                print("Error: Unable to read video stream")
                break

            draw_boundary(img, faceCascade, clf)
            cv2.imshow("Face Recognition", img)

            # Allow 'q' key or clicking close button to exit properly
            if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Face Recognition", cv2.WND_PROP_VISIBLE) < 1:
                break

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = Tk()
    FaceRecognition(root)
    root.mainloop()
