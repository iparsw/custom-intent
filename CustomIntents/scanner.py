import cv2
import os
import os
import sys
import tkinter as tk
import customtkinter as ctk
import pkg_resources


def face_scanner(category: str,                  # category name
                 sub_category: str,              # sub category name
                 base_dir: str,                  # base directory path
                 number_of_photos: int,          # number of photos to take
                 number_of_frames_to_skip: int,  # number of frames to skip before taking images
                 file_name: str,                 # file name
                 image_size: int = 256,          # width and height of image (it will be square)
                 haar_file: str = None,          # directory containing haarcascade
                 camera: int = 0,                # camera
                 colored: bool = False):         # True if you want to save the colored image

    # preparing parameters
    image_size = int(image_size)
    camera = int(camera)
    number_of_frames_to_skip = int(number_of_frames_to_skip)
    number_of_photos = int(number_of_photos)
    number_of_frames_to_skip += 1
    if haar_file is None:
        haar_file = pkg_resources.resource_filename(__name__, "cascades/haarcascade_frontalcatface.xml")
    path = os.path.join(base_dir, category, sub_category)
    if not os.path.isdir(path):
        os.makedirs(path)
    # size e aks
    (width, height) = (image_size, image_size)
    # preparing cascade classifier
    face_cascade = cv2.CascadeClassifier(haar_file)
    webcam = cv2.VideoCapture(camera)

    count = 0
    frame_counter = 1

    if not colored:
        # grayscale
        while count < number_of_photos:
            (_, img) = webcam.read()

            if frame_counter % number_of_frames_to_skip == 0:

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 4)

                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), thickness=4)
                    face = gray[y:y + h, x:x + w]
                    face_resize = cv2.resize(face, (width, height))
                    cv2.imwrite('% s/% s.png' % (path, (category + sub_category + (str(count) + file_name))),
                                face_resize)
                    count += 1

            frame_counter += 1
            cv2.imshow('OpenCV', img)
            key = cv2.waitKey(10)
            if key == 27:
                break
    else:
        # colored
        while count < number_of_photos:
            (_, img) = webcam.read()

            if frame_counter % number_of_frames_to_skip == 0:

                gray = img.copy()
                faces = face_cascade.detectMultiScale(gray, 1.3, 4)

                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), thickness=4)
                    face = gray[y:y + h, x:x + w]
                    face_resize = cv2.resize(face, (width, height))
                    cv2.imwrite('% s/% s.png' % (path, (category + sub_category + (str(count) + file_name))),
                                face_resize)
                    count += 1

            frame_counter += 1
            cv2.imshow('OpenCV', img)
            key = cv2.waitKey(10)
            if key == 27:
                break


def facsScannerCliApp():
    haar_file = pkg_resources.resource_filename(__name__, "cascades/haarcascade_frontalcatface.xml")
    father_path = "data"
    datasets = input("category name : ")
    sub_data = input("sub category name : ")
    file_name = input("file name : ")
    tedad_aks = int(input("number of photos to take : "))
    har_chnd_frame = int(input("how many frames skip between every photo : "))
    camera_feed = int(input("camera feed : "))
    face_scanner(haar_file=haar_file, base_dir=father_path, number_of_frames_to_skip=har_chnd_frame, category=datasets,
                 sub_category=sub_data, number_of_photos=tedad_aks, file_name=file_name, camera=camera_feed)


def faceScannerGuiApp():
    def _faceScannerGuiAppScanner():
        face_scanner(category=entry_categoryName.get(),
                     number_of_photos=entry_numberOfPhotos.get(),
                     sub_category=entry_subCategory.get(),
                     file_name=entry_fileName.get(),
                     camera=entry_CameraNo.get(),
                     number_of_frames_to_skip=entry_SkipedFrames.get(),
                     image_size=entry_imagesize.get(),
                     base_dir="scanned_files",
                     haar_file=pkg_resources.resource_filename(__name__, "cascades/haarcascade_frontalcatface.xml"),
                     colored=colored.get())

    # Window initialization
    window = ctk.CTk()
    window.title("Face Scanner")
    window.geometry("300x430")
    # False : grayscale, True : RGB
    colored = ctk.BooleanVar(value=False)
    # Frame initialization
    frame = ctk.CTkFrame(master=window, corner_radius=10)
    frame.pack(pady=15, padx=10, anchor="center")
    # Category name input
    entry_categoryName = ctk.CTkEntry(master=frame, placeholder_text="Category", width=400, font=("Bold", 17))
    entry_categoryName.pack(pady=8, padx=10)
    # Sub Category name input
    entry_subCategory = ctk.CTkEntry(master=frame, placeholder_text="Sub Category", width=400, font=("Bold", 17))
    entry_subCategory.pack(pady=8, padx=10)
    # Saved file name input
    entry_fileName = ctk.CTkEntry(master=frame, placeholder_text="File Name", width=400, font=("Bold", 17))
    entry_fileName.pack(pady=8, padx=10)
    # Number of photos to take input
    entry_numberOfPhotos = ctk.CTkEntry(master=frame, placeholder_text="Number of Photos", width=400, font=("Bold", 17))
    entry_numberOfPhotos.pack(pady=8, padx=10)
    # Number of frames to skip input
    entry_SkipedFrames = ctk.CTkEntry(master=frame, placeholder_text="Number of skiped frames", width=400,
                                      font=("Bold", 17))
    entry_SkipedFrames.pack(pady=8, padx=10)
    # Camera number input
    entry_CameraNo = ctk.CTkEntry(master=frame, placeholder_text="Camera number", width=400, font=("Bold", 17))
    entry_CameraNo.pack(pady=8, padx=10)
    # Image size input
    entry_imagesize = ctk.CTkEntry(master=frame, placeholder_text="Image size", width=400, font=("Bold", 17))
    entry_imagesize.pack(pady=8, padx=10)
    # Colored or Grayscale
    entry_color = ctk.CTkCheckBox(master=frame, variable=colored, text="Colored")
    entry_color.pack(pady=8, padx=10)
    # Start Button
    button_start = ctk.CTkButton(master=frame, text="Start", font=("Roboto", 25), command=_faceScannerGuiAppScanner)
    button_start.pack(pady=10)
    # Starting mainloop
    window.mainloop()
