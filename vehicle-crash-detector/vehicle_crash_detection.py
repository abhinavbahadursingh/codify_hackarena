# Import the following module
import threading
import time
from tkinter.ttk import Style
import PIL
import tensorflow as tf
import cv2
import numpy as np
import tkinter as tk
from PIL import Image
from PIL import ImageTk
from object_detection.utils import label_map_util
from firebase import crash_to_firebase
from object_detection.utils import visualization_utils as vis_utils
from tensorflow.io import gfile  # ✅ FIXED: for label map loading
from tkinter import ttk
import functools
import datetime

class VehicleCrash:

    def __init__(self, detections_update_label, content, button1):
        self.detections_update_label = detections_update_label
        self.content = content
        self.source = None
        self.running = False
        self.button1 = button1
        self.count = 0
        self.i = 0

    def set_source(self, source):
        self.source = source

    PATH_TO_SAVED_MODEL = "inference_graph\\saved_model"

    category_index = label_map_util.create_category_index_from_labelmap("label_map.pbtxt", use_display_name=True)

    def visualise_on_image(self, frame, image, bboxes, labels, scores, thresh):
        (h, w, d) = image.shape
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        for bbox, label, score in zip(bboxes, labels, scores):
            if score > thresh:
                xmin, ymin = int(bbox[1] * w), int(bbox[0] * h)
                xmax, ymax = int(bbox[3] * w), int(bbox[2] * h)

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(image, f"{label}: {int(score * 100)} %", (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2)

                self.count += 1
                print(self.count)

                if self.count == 5:
                    label_box_image = frame[ymin:ymax, xmin:xmax]
                    cv2.imwrite("outputs/frame_img/vcd_frame" + str(current_datetime) + str(self.i) + ".jpg", image)
                    image_size = (1920, 1080)
                    resized_image = cv2.resize(label_box_image, image_size)
                    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                    sharpened_image = cv2.filter2D(resized_image, -1, kernel)
                    png_quality = 100
                    cv2.imwrite("outputs/inside_label_img/vcd_inlabel" + str(current_datetime) + str(self.i) + ".png",
                                sharpened_image,
                                [int(cv2.IMWRITE_JPEG_QUALITY), png_quality])

                if self.count == 20:
                    print("Vehicle_Crash_Detected")
                    perform_label_detected_func = threading.Thread(target=self.perform_label_detected)
                    perform_label_detected_func.start()
                    self.i += 1
                    self.count = 0
                    break

        return image

    def update_progress(self, progress, value):
        progress['value'] = value
        progress.update()

    def perform_label_detected(self):
        self.detections_update_label.configure(text="Vehicle Crash has been Detected")
        time.sleep(0.5)
        crash_to_firebase.crash_to_fire(23, 12, 19)
        time.sleep(0.5)
        self.detections_update_label.configure(text="")

    detect_fn = ""

    @functools.lru_cache(maxsize=None)
    def load_model(self):
        style = Style()
        style.theme_use('alt')
        style.configure("Horizontal.TProgressbar", troughcolor='white', background='black', thickness=30)

        progress = ttk.Progressbar(self.content, orient=tk.HORIZONTAL, style="Horizontal.TProgressbar", length=300,
                                   mode='determinate')
        progress.pack(pady=200, side="top", anchor="s")
        self.detections_update_label.configure(text="Loading 0%")
        self.update_progress(progress, 0)
        time.sleep(0.1)

        print("Loaded 10% saved model ...")
        self.update_progress(progress, 10)
        self.detections_update_label.configure(text="Loading .10%")
        time.sleep(0.1)

        global detect_fn
        print("Loading saved model ...")
        detect_fn = tf.saved_model.load(self.PATH_TO_SAVED_MODEL)

        print("Loaded 50% saved model ...")
        self.detections_update_label.configure(text="Loading ....50%")
        self.update_progress(progress, 50)
        time.sleep(1.5)

        print("Model Loaded!")
        self.detections_update_label.configure(text="Loading .........100%")
        self.update_progress(progress, 100)
        time.sleep(1.5)
        self.detections_update_label.configure(text="")
        time.sleep(0.1)
        progress.destroy()
        return detect_fn

    def close_canvas(self, canvas):
        canvas.destroy()
        self.content.update()

    def run_detection(self):
        self.running = True
        while self.running:
            print("Video Source : ", self.source)
            video_capture = cv2.VideoCapture(self.source)

            start_time = time.time()
            canvas = tk.Canvas(self.content, width=1000, height=600)
            canvas.pack(side="top", anchor="n", padx=10, pady=40)

            frame_width = int(video_capture.get(3))
            frame_height = int(video_capture.get(4))
            size = (frame_width, frame_height)

            result = cv2.VideoWriter('outputs/detection_video/det_vid.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                     15, size)

            while True:
                ret, frame = video_capture.read()
                if not ret:
                    self.close_canvas(canvas)
                    self.stop_detection()
                    self.button1.config(text="Detection \nOFF")
                    self.detections_update_label.configure(text="")
                    self.source = "Video Source"
                    print('Unable to read video / Video ended')
                    self.detections_update_label.configure(text="Unable to read video / Video ended")
                    break

                frame = cv2.flip(frame, 1)
                image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]

                detections = detect_fn(input_tensor)

                score_thresh = 0.92
                max_detections = 1
                scores = detections['detection_scores'][0, :max_detections].numpy()
                bboxes = detections['detection_boxes'][0, :max_detections].numpy()
                labels = detections['detection_classes'][0, :max_detections].numpy().astype(np.int64)
                labels = [self.category_index[n]['name'] for n in labels]

                self.visualise_on_image(frame, frame, bboxes, labels, scores, score_thresh)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = PIL.Image.fromarray(frame)
                image = image.resize((1000, 600))
                photo = PIL.ImageTk.PhotoImage(image)

                end_time = time.time()
                fps = int(1 / (end_time - start_time))
                start_time = end_time

                canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                canvas.create_text(50, video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) + 25, text=f"FPS: {fps}",
                                   font=("Arial", 14), fill="red", anchor=tk.NW)
                canvas.update()
                result.write(frame)

            video_capture.release()

    def stop_detection(self):
        self.running = False
        self.count = 0
        print("Detection Stopped")
