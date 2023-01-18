import os
from pathlib import Path
from time import perf_counter

import keras
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras_preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Dropout, MaxPooling2D, Flatten, \
    Conv2D, GlobalAveragePooling2D, Activation, AveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam, Adamax, Adagrad
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

import wandb
from wandb.keras import WandbCallback
import matplotlib.pyplot as plt

import imghdr
import cv2

from threading import Thread

from functools import wraps

from CustomIntents.Bcolor import bcolors

import pkg_resources

import gradio as gr


class VideoStream:

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the latest frame
        return self.frame

    def stop(self):
        self.stopped = True


class BinaryImageClassificator:
    def __init__(self, data_folder="data", model_name="imageclassification_model", first_class="1", second_class="2",
                 gpu=None):
        self.optimizer = None
        self.acc = None
        self.re = None
        self.pre = None
        self.first_class = first_class
        self.second_class = second_class
        self.tensorboard_callback = None
        self.logdir = None
        self.model = None
        self.batch = None
        self.test = None
        self.val = None
        self.train = None
        self.test_size = None
        self.val_size = None
        self.train_size = None
        self.data_iterator = None
        self.hist = None
        self.gpu_usage = gpu
        self._check_for_gpu_availability()
        self.data_folder = data_folder
        self.name = model_name
        self.data = None
        self._oom_avoider()

    def _check_for_gpu_availability(self):
        number_of_gpus = len(tf.config.list_physical_devices('GPU'))
        if self.gpu_usage and self.gpu_usage is not None:
            if number_of_gpus == 0:
                print(f"{bcolors.FAIL}NO GPU AVAILABLE !! we will use your cpu {bcolors.ENDC}")
                self.gpu_usage = False
        elif self.gpu_usage is None:
            if number_of_gpus == 0:
                self.gpu_usage = False

    def _remove_dogy_images(self):
        data_dir = self.data_folder
        image_exts = ['jpeg', 'jpg', 'bmp', 'png']
        for image_class in os.listdir(data_dir):
            for image in os.listdir(os.path.join(data_dir, image_class)):
                image_path = os.path.join(data_dir, image_class, image)
                try:
                    img = cv2.imread(image_path)
                    tip = imghdr.what(image_path)
                    if tip not in image_exts:
                        print('Image not in ext list {}'.format(image_path))
                        os.remove(image_path)
                except Exception:
                    print('Issue with image {}'.format(image_path))
                    os.remove(image_path)

    def _load_data(self):
        self.data = tf.keras.utils.image_dataset_from_directory(self.data_folder)
        self.data_iterator = self.data.as_numpy_iterator()
        self.batch = self.data_iterator.next()
        print(f"{bcolors.OKGREEN}loading data succsesfuly{bcolors.ENDC}")

    def _scale_data(self):
        self.data = self.data.map(lambda x, y: (x / 255, y))
        print(f"{bcolors.OKGREEN}scaling data succsesfuly{bcolors.ENDC}")

    def _augmanet_data(self, model_type="s1"):
        if "a" in model_type:
            data_augmentation = Sequential([
                tf.keras.layers.RandomFlip(mode="horizontal"),
                tf.keras.layers.RandomRotation((-0.3, 0.3)),
                tf.keras.layers.RandomZoom(height_factor=(-0.1, 0.1)),
                tf.keras.layers.RandomBrightness(factor=0.2)
            ])
            self.data = self.data.map(lambda x, y: (data_augmentation(x, training=True), y),
                                      num_parallel_calls=tf.data.AUTOTUNE)
            print(f"{bcolors.OKGREEN}augmenting data succsesfuly{bcolors.ENDC}")

    def _split_data(self, validation_split=0.2, test_split=0):
        self.train_size = int(len(self.data) * (1 - validation_split - test_split))
        self.val_size = int(len(self.data) * validation_split)
        self.test_size = int(len(self.data) * test_split)
        self.train = self.data.take(self.train_size)
        self.val = self.data.skip(self.train_size).take(self.val_size)
        self.test = self.data.skip(self.train_size + self.val_size).take(self.test_size)
        print(f"{bcolors.OKGREEN}spliting data succsesfuly{bcolors.ENDC}")

    def _prefetching_data(self):
        self.train = self.train.prefetch(tf.data.AUTOTUNE)
        self.val = self.val.prefetch(tf.data.AUTOTUNE)
        self.test = self.test.prefetch(tf.data.AUTOTUNE)
        print(f"{bcolors.OKGREEN}prefetching data succsesfuly{bcolors.ENDC}")

    @staticmethod
    def _make_small_Xception_model(input_shape, num_classes=2):
        inputs = keras.Input(shape=input_shape)

        x = tf.compat.v1.keras.layers.Rescaling(1.0 / 255)(inputs)
        x = Conv2D(128, 3, strides=2, padding="same")(x)
        x = tf.compat.v1.keras.layers.BatchNormalization()(x)
        x = Activation("relu")(x)

        previous_block_activision = x
        for size in [256, 512, 728]:
            x = Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = tf.compat.v1.keras.layers.BatchNormalization()(x)
            x = Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = tf.compat.v1.keras.layers.BatchNormalization()(x)
            x = MaxPooling2D(3, strides=2, padding="same")(x)
            # residual
            residual = Conv2D(size, 1, strides=2, padding="same")(previous_block_activision)
            x = layers.add([x, residual])
            previous_block_activision = x

        x = layers.SeparableConv2D(1024, 3, padding="same")(x)
        x = tf.compat.v1.keras.layers.BatchNormalization()(x)
        x = Activation("relu")(x)
        x = GlobalAveragePooling2D()(x)
        if num_classes == 2:
            activision = "sigmoid"
            units = 1
        else:
            activision = "softmax"
            units = num_classes
        x = Dropout(0.5)(x)
        outputs = Dense(units, activation=activision)(x)
        return keras.Model(inputs, outputs)

    def _build_model(self, optimizer, model_type="s1"):
        print(f"model type : {model_type}")
        succsesful = False
        if model_type == "s1" or model_type == "s1a":
            self.model = Sequential()
            self.model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
            self.model.add(MaxPooling2D())
            self.model.add(Conv2D(32, (3, 3), 1, activation='relu'))
            self.model.add(MaxPooling2D())
            self.model.add(Conv2D(16, (3, 3), 1, activation='relu'))
            self.model.add(MaxPooling2D())
            self.model.add(Flatten())
            self.model.add(Dense(256, activation='relu'))
            self.model.add(Dense(1, activation='sigmoid'))
            succsesful = True

        elif model_type == "s2":
            self.model = Sequential()
            self.model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
            self.model.add(MaxPooling2D())
            self.model.add(Conv2D(32, (3, 3), 1, activation='relu'))
            self.model.add(MaxPooling2D())
            self.model.add(Conv2D(32, (3, 3), 1, activation='relu'))
            self.model.add(MaxPooling2D())
            self.model.add(Flatten())
            self.model.add(Dense(256, activation='relu'))
            self.model.add(Dense(1, activation='sigmoid'))
            succsesful = True

        elif model_type == "s3":
            self.model = Sequential()
            self.model.add(Conv2D(32, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
            self.model.add(MaxPooling2D())
            self.model.add(Conv2D(32, (3, 3), 1, activation='relu'))
            self.model.add(MaxPooling2D())
            self.model.add(Conv2D(64, (3, 3), 1, activation='relu'))
            self.model.add(MaxPooling2D())
            self.model.add(Dropout(0.4))
            self.model.add(Flatten())
            self.model.add(Dense(256, activation='relu'))
            self.model.add(Dense(1, activation='sigmoid'))
            succsesful = True

        elif model_type == "m1":
            self.model = Sequential()
            self.model.add(Conv2D(32, (3, 3), 1, padding="same", activation='relu', input_shape=(256, 256, 3)))
            self.model.add(Conv2D(32, (3, 3), 1, activation='relu'))
            self.model.add(MaxPooling2D())
            self.model.add(Dropout(0.25))
            self.model.add(Conv2D(64, (3, 3), 1, padding="same", activation='relu'))
            self.model.add(Conv2D(64, (3, 3), 1, activation='relu'))
            self.model.add(MaxPooling2D())
            self.model.add(Dropout(0.25))
            self.model.add(Flatten())
            self.model.add(Dense(512, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(1, activation='sigmoid'))
            succsesful = True

        elif model_type == "l1" or model_type.lower() == "vgg-19":
            self.model = Sequential()
            self.model.add(Conv2D(64, (3, 3), 1, padding="same", activation='relu', input_shape=(256, 256, 3)))
            self.model.add(Conv2D(64, (3, 3), 1, padding="same", activation='relu'))
            self.model.add(MaxPooling2D())
            self.model.add(Conv2D(128, (3, 3), 1, padding="same", activation='relu'))
            self.model.add(Conv2D(128, (3, 3), 1, padding="same", activation='relu'))
            self.model.add(MaxPooling2D())
            self.model.add(Conv2D(256, (3, 3), 1, padding="same", activation='relu'))
            self.model.add(Conv2D(256, (3, 3), 1, padding="same", activation='relu'))
            self.model.add(Conv2D(256, (3, 3), 1, padding="same", activation='relu'))
            self.model.add(Conv2D(256, (3, 3), 1, padding="same", activation='relu'))
            self.model.add(MaxPooling2D())
            self.model.add(Conv2D(512, (3, 3), 1, padding="same", activation='relu'))
            self.model.add(Conv2D(512, (3, 3), 1, padding="same", activation='relu'))
            self.model.add(Conv2D(512, (3, 3), 1, padding="same", activation='relu'))
            self.model.add(Conv2D(512, (3, 3), 1, padding="same", activation='relu'))
            self.model.add(MaxPooling2D())
            self.model.add(Conv2D(512, (3, 3), 1, padding="same", activation='relu'))
            self.model.add(Conv2D(512, (3, 3), 1, padding="same", activation='relu'))
            self.model.add(Conv2D(512, (3, 3), 1, padding="same", activation='relu'))
            self.model.add(Conv2D(512, (3, 3), 1, padding="same", activation='relu'))
            self.model.add(MaxPooling2D())
            self.model.add(Dropout(0.1))
            self.model.add(Flatten())
            self.model.add(Dense(4096, activation='relu'))
            self.model.add(Dense(4096, activation='relu'))
            self.model.add(Dense(1, activation='sigmoid'))
            succsesful = True

        elif model_type == "l1.1":
            self.model = Sequential()
            self.model.add(Conv2D(64, (3, 3), 1, padding="same", activation='relu', input_shape=(256, 256, 3)))
            self.model.add(Conv2D(64, (3, 3), 1, padding="same", activation='relu'))
            self.model.add(MaxPooling2D())
            self.model.add(Conv2D(128, (3, 3), 1, padding="same", activation='relu'))
            self.model.add(Conv2D(128, (3, 3), 1, padding="same", activation='relu'))
            self.model.add(MaxPooling2D())
            self.model.add(Conv2D(256, (3, 3), 1, padding="same", activation='relu'))
            self.model.add(Conv2D(256, (3, 3), 1, padding="same", activation='relu'))
            self.model.add(Conv2D(256, (3, 3), 1, padding="same", activation='relu'))
            self.model.add(Conv2D(256, (3, 3), 1, padding="same", activation='relu'))
            self.model.add(MaxPooling2D())
            self.model.add(Conv2D(512, (3, 3), 1, padding="same", activation='relu'))
            self.model.add(Conv2D(512, (3, 3), 1, padding="same", activation='relu'))
            self.model.add(Conv2D(512, (3, 3), 1, padding="same", activation='relu'))
            self.model.add(Conv2D(512, (3, 3), 1, padding="same", activation='relu'))
            self.model.add(MaxPooling2D())
            self.model.add(Conv2D(512, (3, 3), 1, padding="same", activation='relu'))
            self.model.add(Conv2D(512, (3, 3), 1, padding="same", activation='relu'))
            self.model.add(Conv2D(512, (3, 3), 1, padding="same", activation='relu'))
            self.model.add(Conv2D(512, (3, 3), 1, padding="same", activation='relu'))
            self.model.add(BatchNormalization())
            self.model.add(GlobalAveragePooling2D())
            self.model.add(Dropout(0.1))
            self.model.add(Dense(2048, activation="relu"))
            self.model.add(Dropout(0.1))
            self.model.add(Dense(2048, activation="relu"))
            self.model.add(Dense(1, activation='sigmoid'))
            succsesful = True

        elif model_type == "l2":
            self.model = Sequential()
            self.model.add(Conv2D(128, (7, 7), 1, padding="same", activation='relu', input_shape=(256, 256, 3)))
            self.model.add(MaxPooling2D())
            for _ in range(6):
                self.model.add(Conv2D(128, (3, 3), 1, padding="same", activation='relu'))  # 6
            for _ in range(8):
                self.model.add(Conv2D(256, (3, 3), 1, padding="same", activation='relu'))  # 8
            for _ in range(12):
                self.model.add(Conv2D(512, (3, 3), 1, padding="same", activation='relu'))  # 12
            for _ in range(6):
                self.model.add(Conv2D(512, (3, 3), 1, padding="same", activation='relu'))  # 6
            self.model.add(BatchNormalization())
            self.model.add(GlobalAveragePooling2D())
            self.model.add(Dropout(0.1))
            self.model.add(Dense(2048, activation="relu"))
            self.model.add(Dropout(0.1))
            self.model.add(Dense(2048, activation="relu"))
            self.model.add(Dense(1, activation='sigmoid'))
            succsesful = True

        elif model_type == "x1":
            self.model = self._make_small_Xception_model(input_shape=(256, 256, 3), num_classes=2)
            succsesful = True

        else:
            print(f"{bcolors.FAIL}model {model_type} is undifinde\n"
                  f"it will defuat to s1 {bcolors.ENDC}")
            self._build_model(model_type="s1", optimizer=optimizer)
            succsesful = False

        if succsesful:
            print(self.model.summary())

        self.model.compile(optimizer=optimizer, loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])

    def _build_optimizer(self, learning_rate=0.00001, optimizer_type="adam"):
        if optimizer_type.lower() == "adam":
            self.optimizer = Adam(learning_rate=learning_rate)

    def _seting_logdir(self):
        current_dir = os.getcwd()
        parent_dir = os.path.dirname(current_dir)
        if self.logdir is None:
            if Path('new_folder').is_dir():
                self.logdir = "logs"
            else:
                path = os.path.join(parent_dir, "logs")
                os.mkdir(path)
                self.logdir = "logs"
        else:
            if Path(self.logdir).is_dir():
                pass
            else:
                path = os.path.join(parent_dir, self.logdir)
                os.mkdir(path)

    def train_model(self, epochs=20, model_type="s1", logdir=None, optimizer_type="adam", learning_rate=0.00001,
                    class_weight=None, prefetching=False, plot_model=True, validation_split=0.2, test_split=0):
        if type(epochs) is not int:
            print(f"{bcolors.FAIL}epochs should be an int\n"
                  f"it will defualt to 20{bcolors.ENDC}")
            epochs = 20
        self._oom_avoider()
        self._remove_dogy_images()
        self._load_data()
        self._scale_data()
        self._augmanet_data(model_type=model_type)
        self._split_data(validation_split=validation_split, test_split=test_split)
        if prefetching:
            self._prefetching_data()
        self.logdir = logdir
        self._seting_logdir()
        self._build_optimizer(optimizer_type=optimizer_type, learning_rate=learning_rate)
        self._build_model(model_type=model_type, optimizer=self.optimizer)
        if plot_model:
            tf.keras.utils.plot_model(self.model, show_shapes=True, show_layer_activations=True)
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=self.logdir)
        self.hist = self.model.fit(self.train, epochs=epochs, validation_data=self.val,
                                   callbacks=[self.tensorboard_callback], class_weight=class_weight)
        self._plot_acc()
        self._plot_loss()

    def _plot_loss(self):
        fig = plt.figure()
        plt.plot(self.hist.history['loss'], color='teal', label='loss')
        plt.plot(self.hist.history['val_loss'], color='orange', label='val_loss')
        fig.suptitle('Loss', fontsize=20)
        plt.legend(loc="upper left")
        plt.grid()
        plt.show()

    def _plot_acc(self):
        fig = plt.figure()
        plt.plot(self.hist.history['accuracy'], color='teal', label='accuracy')
        plt.plot(self.hist.history['val_accuracy'], color='orange', label='val_accuracy')
        fig.suptitle('Accuracy', fontsize=20)
        plt.legend(loc="upper left")
        plt.grid()
        plt.show()

    def save_model(self, model_file_name=None):
        if model_file_name is None:
            model_file_name = self.name
        if model_file_name.endswith(".h5"):
            self.model.save(model_file_name)
        else:
            self.model.save(f"{model_file_name}.h5")

    def load_model(self, name="imageclassification_model"):
        if name.endswith(".h5"):
            self.model = load_model(name)
        else:
            self.model = load_model(f"{name}.h5")

    def _oom_avoider(self):
        if self.gpu_usage:
            gpus = tf.config.experimental.list_physical_devices("GPU")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

    def predict_from_files_path(self, image_file_path):
        img = cv2.imread(image_file_path)
        resize = tf.image.resize(img, (256, 256))
        yhat = self.model.predict(np.expand_dims(resize / 255, 0))
        if yhat > 0.5:
            return self.first_class, ((yhat[0][0] - 0.5) * 2) * 100
        else:
            return self.second_class, (1 - (yhat[0][0] * 2)) * 100

    def predict_from_imshow(self, img):
        resize = tf.image.resize(img, (256, 256))
        yhat = self.model.predict(np.expand_dims(resize / 255, 0))
        if yhat > 0.5:
            return self.first_class, ((yhat[0][0] - 0.5) * 2) * 100
        else:
            return self.second_class, (1 - (yhat[0][0] * 2)) * 100

    def predict_from_numpy(self, img):
        resize = tf.image.resize(img, (256, 256))
        yhat = self.model.predict(np.expand_dims(resize / 255, 0))
        if yhat > 0.5:
            return self.first_class, ((yhat[0][0] - 0.5) * 2) * 100
        else:
            return self.second_class, (1 - (yhat[0][0] * 2)) * 100

    def gradio_preview(self, share=False, inbrowser=True):
        demo = gr.Interface(self.predict_from_numpy,
                            inputs=gr.Image(label="your image"),
                            outputs=[gr.Label(label="class"),
                                     gr.Label(label="Accuracy")],
                            allow_flagging="never")
        print(f"open http://localhost:7860 for viewing your model preview")
        demo.launch(share=share, inbrowser=inbrowser)

    def evaluate_model(self):
        self.pre = Precision()
        self.re = Recall()
        self.acc = BinaryAccuracy()
        for batch in self.test.as_numpy_iterator():
            X, y = batch
            yhat = self.model.predict(X)
            self.pre.update_state(y, yhat)
            self.re.update_state(y, yhat)
            self.acc.update_state(y, yhat)
        return [self.pre.result(), self.re.result(), self.acc.result()]

    def realtime_prediction(self, src=0):
        # Variables declarations
        frame_count = 0
        last = 0
        font = cv2.FONT_HERSHEY_TRIPLEX
        font_color = (255, 255, 255)
        vs = VideoStream(src=src).start()
        while True:
            frame = vs.read()
            frame_count += 1

            # Only run every 10 frames
            if frame_count % 10 == 0:
                prediction = self.predict_from_imshow(frame)
                # Change the text position depending on your camera resolution
                cv2.putText(frame, prediction, (20, 400), font, 1, font_color)

                if frame_count > 20:
                    fps = vs.stream.get(cv2.CAP_PROP_FPS)
                    fps_text = "fps: " + str(np.round(fps, 2))
                    cv2.putText(frame, fps_text, (460, 460), font, 1, font_color)

                cv2.imshow("Frame", frame)
                last += 1

                # if the 'q' key is pressed, stop the loop
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        # cleanup everything
        vs.stop()
        cv2.destroyAllWindows()
        print("Done")

    def realtime_face_prediction(self, src=0):
        haar_cascade_file = pkg_resources.resource_filename(__name__, "cascades/haarcascade_frontalcatface.xml")
        detector = cv2.CascadeClassifier(haar_cascade_file)
        camera = cv2.VideoCapture(src)
        # keep looping
        while True:
            # grab the current frame
            (grabbed, frame) = camera.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frameClone = frame.copy()
            rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                              minNeighbors=5, minSize=(10, 10),
                                              flags=cv2.CASCADE_SCALE_IMAGE)
            # loop over the face bounding boxes
            for (fX, fY, fW, fH) in rects:
                # extract the ROI of the face from the grayscale image,
                # resize it to a fixed 28x28 pixels, and then prepare the
                # ROI for classification via the CNN
                roi = frame[fY:fY + fH, fX:fX + fW]
                roi = cv2.resize(roi, (256, 256))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = str(self.model.predict(roi))
                label = prediction
                cv2.putText(frameClone, label, (fX, fY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                              (0, 0, 255), 2)
            # show our detected faces along with smiling/not smiling labels
            cv2.imshow("Face", frameClone)
            # if the 'q' key is pressed, stop the loop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        # cleanup the camera and close any open windows
        camera.release()
        cv2.destroyAllWindows()
