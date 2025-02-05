from tkinter import * 
from tkinter import ttk
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2 
import numpy as np
import mysql.connector
from time import strftime
from datetime import datetime, timedelta
import csv

# Global dictionary to track the last attendance time for each person
last_attendance_time = {}

def update_attendance_count(student_id):
    conn = mysql.connector.connect(
        host="localhost", user="root", password="sidhu1234@deep", database="face_recognizer"
    )
    my_cursor = conn.cursor()

    # Fetch current attendance count
    my_cursor.execute(f"SELECT attendance_count FROM student WHERE student_id = {student_id}")
    result = my_cursor.fetchone()

    # If attendance_count is None, set it to 0
    current_count = result[0] if result and result[0] is not None else 0
    new_count = current_count + 1

    # Update the attendance count in the database
    my_cursor.execute(f"UPDATE student SET attendance_count = {new_count} WHERE student_id = {student_id}")
    conn.commit()
    conn.close()

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
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Attendance Status", message)
        root.destroy()

    def mark_attendance(self, d, n, s, r):
        current_time = datetime.now()
        
        if d in last_attendance_time:
            last_time = last_attendance_time[d]
            time_diff = current_time - last_time
            if time_diff < timedelta(minutes=30):
                print(f"Cannot mark attendance for {d} again. Last attendance was {time_diff} ago.")
                return

        update_attendance_count(d)
        
        with open("attendance.csv", "a+", newline="\n") as f:
            f.seek(0)
            myDataList = f.readlines()
            name_list = [line.split(",")[0] for line in myDataList]

            dtString = current_time.strftime("%H:%M:%S")
            date = current_time.strftime("%d/%m/%Y")
            
            if d not in name_list:
                f.write(f"{d},{n},{s},{r},{dtString},{date},Present\n")

        last_attendance_time[d] = current_time
        print(f"Attendance for {d} marked successfully at {dtString} on {date}.")
        self.show_notification(f"Attendance for {n} ({d}) marked successfully!")

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

        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.read("classifier.xml")

        video_capture = cv2.VideoCapture(0)
        while True:
            ret, img = video_capture.read()
            if not ret:
                break
            draw_boundary(img, faceCascade, clf)
            cv2.imshow("Face Recognition", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = Tk()
    FaceRecognition(root)
    root.mainloop()
