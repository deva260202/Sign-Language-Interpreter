import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, Label
from PIL import Image, ImageTk
from datetime import datetime
from tensorflow.keras.models import load_model  # type: ignore

# Load the saved model
model_path = 'C:/Users/admin/Desktop/Python Prog/Age gender detector/sign_language_model.keras'
loaded_model = load_model(model_path)

# Define label map
label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
             10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
             19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

phrase_map = {
    "What is your Name": "Who are you",
    "Who are you": "What is your Name"
}

def predict_sign(image, model, label_map):
    image = cv2.resize(image, (64, 64))
    image = np.expand_dims(image, axis=0) / 255.0
    prediction = model.predict(image)
    print(f"Prediction: {prediction}") 
    label_idx = np.argmax(prediction)
    print(f"Label Index: {label_idx}")  
    if label_idx in label_map:
        label = label_map[label_idx]
    else:
        label = "Unknown"
    return label

def is_valid_time():
    current_time = datetime.now().time()
    start_time = datetime.strptime('18:00:00', '%H:%M:%S').time()
    end_time = datetime.strptime('22:00:00', '%H:%M:%S').time()
    return start_time <= current_time <= end_time

def detect_phrase(sign_sequence):
    sequence_str = " ".join(sign_sequence)
    for phrase, mapped_phrase in phrase_map.items():
        if phrase in sequence_str:
            return mapped_phrase
    return None

def upload_image():
    if not is_valid_time():
        messagebox.showwarning("Warning", "Predictions are only allowed between 6 PM and 10 PM.")
        return
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        if image is not None:
            sign = predict_sign(image, loaded_model, label_map)
            result_label.config(text=f'Predicted Sign: {sign}')
            display_image(file_path)

def upload_video():
    if not is_valid_time():
        messagebox.showwarning("Warning", "Predictions are only allowed between 6 PM and 10 PM.")
        return
    file_path = filedialog.askopenfilename()
    if file_path:
        cap = cv2.VideoCapture(file_path)
        sign_sequence = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                sign = predict_sign(frame, loaded_model, label_map)
                sign_sequence.append(sign)
                display_video_frame(frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()

        detected_phrase = detect_phrase(sign_sequence)
        if detected_phrase:
            result_label.config(text=f'Predicted Phrase: {detected_phrase}')
        else:
            result_label.config(text="Predicted Phrase: None")

def display_image(image_path):
    img = Image.open(image_path)
    img = img.resize((400, 400), Image.LANCZOS)
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img

def display_video_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img = img.resize((400, 400), Image.LANCZOS)
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img

root = tk.Tk()
root.title("Sign Language Predictor")
root.geometry("1000x700")

heading = Label(root, text="Sign Language Predictor", pady=20, font=("Arial", 24, "bold"))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()

frame = tk.Frame(root)
frame.pack(pady=20)

upload_image_button = tk.Button(frame, text="Upload Image", command=upload_image, font=('Arial', 16))
upload_image_button.grid(row=0, column=0, padx=20, pady=10)

upload_video_button = tk.Button(frame, text="Upload Video", command=upload_video, font=('Arial', 16))
upload_video_button.grid(row=0, column=1, padx=20, pady=10)

result_label = tk.Label(root, text="Predicted Sign: None", font=('Arial', 20))
result_label.pack(pady=20)

panel = tk.Label(root)
panel.pack(pady=20)

root.mainloop()
