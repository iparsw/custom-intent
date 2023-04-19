import tensorflow as tf

import os
import math
import random
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image_dataset_from_directory

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import PIL

from CustomIntents.utils import SSIM_loss, ESPCNCallback, Involution, PSNR_loss, IPCUSL_loss, PSNR_metric, charbonnier_loss, TV_loss, \
    TVD_loss, WS_loss


class SuperRes:
    def __init__(self, input_size: tuple = (300, 300),
                 upscale_factor: int = 3,
                 cpu_only: bool = False):
        self.dataset_path = None
        self.test_img_paths = None
        self.hist = None
        self.valid_ds = None
        self.train_ds = None
        self.model = None
        self.model_type = None
        self.loss_fn_weight = None
        self.input_size = input_size
        self.upscale_factor = upscale_factor
        self.output_size = (self.input_size[0] * self.upscale_factor, self.input_size[1] * self.upscale_factor)
        self.cpu_only = cpu_only
        self._cpu_only()

    def _cpu_only(self):
        if self.cpu_only:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    @staticmethod
    def _scaling(input_image):
        input_image = input_image / 255.0
        return input_image

    def _process_input(self, input):
        input = tf.image.rgb_to_yuv(input)
        last_dimension_axis = len(input.shape) - 1
        y, u, v = tf.split(input, 3, axis=last_dimension_axis)
        return tf.image.resize(y, [self.input_size[0], self.input_size[1]], method="area")

    @staticmethod
    def _process_target(input):
        input = tf.image.rgb_to_yuv(input)
        last_dimension_axis = len(input.shape) - 1
        y, u, v = tf.split(input, 3, axis=last_dimension_axis)
        return y

    def _build_model(self, model_type):
        self.model_type = model_type
        if self.model_type == "xs1":
            channels = 1
            conv_args = {
                "activation": "relu",
                "kernel_initializer": "Orthogonal",
                "padding": "same",
            }
            input = keras.Input(shape=(None, None, channels))
            x = layers.Conv2D(64, 5, **conv_args)(input)
            x = layers.Conv2D(64, 3, **conv_args)(x)
            x = layers.Conv2D(32, 3, **conv_args)(x)
            x = layers.Conv2D(channels * (self.upscale_factor ** 2), 3, **conv_args)(x)
            outputs = tf.nn.depth_to_space(x, self.upscale_factor)
            self.model = keras.Model(input, outputs)

        if self.model_type == "xs1.1":
            channels = 1
            conv_args = {
                "activation": "relu",
                "kernel_initializer": "Orthogonal",
                "padding": "same",
            }
            input = keras.Input(shape=(None, None, channels))
            x = layers.Conv2D(128, 5, **conv_args)(input)
            x = layers.Conv2D(128, 3, **conv_args)(x)
            x = layers.Conv2D(64, 3, **conv_args)(x)
            x = layers.Conv2D(channels * (self.upscale_factor ** 2), 3, **conv_args)(x)
            outputs = tf.nn.depth_to_space(x, self.upscale_factor)
            self.model = keras.Model(input, outputs)

        elif self.model_type == "xs2":
            channels = 1
            conv_args = {
                "activation": "relu",
                "kernel_initializer": "Orthogonal",
                "padding": "same",
            }
            input = keras.Input(shape=(None, None, channels))
            x = layers.Conv2D(64, 5, **conv_args)(input)
            x = layers.Conv2D(64, 3, **conv_args)(x)
            x = layers.Conv2D(64, 3, **conv_args)(x)
            x = layers.Conv2D(32, 3, **conv_args)(x)
            x = layers.Conv2D(32, 3, **conv_args)(x)
            x = layers.Conv2D(channels * (self.upscale_factor ** 2), 3, **conv_args)(x)
            outputs = tf.nn.depth_to_space(x, self.upscale_factor)
            self.model = keras.Model(input, outputs)

        elif self.model_type == "xs3":
            channels = 1
            conv_args = {
                "activation": "relu",
                "kernel_initializer": "Orthogonal",
                "padding": "same",
            }
            input = keras.Input(shape=(None, None, channels))
            x = layers.Conv2D(64, 5, **conv_args)(input)
            x = layers.Conv2D(64, 3, **conv_args)(x)
            x = layers.Conv2D(32, 3, **conv_args)(x)
            x = layers.Conv2D(channels * (self.upscale_factor ** 2), 3, **conv_args)(x)
            x = tf.nn.depth_to_space(x, self.upscale_factor)
            x = layers.Conv2D(32, 3, **conv_args)(x)
            outputs = layers.Conv2D(1, 3, **conv_args)(x)
            self.model = keras.Model(input, outputs)

        elif self.model_type == "ts1":
            channels = 1
            conv_args = {
                "activation": "relu",
                "kernel_initializer": "Orthogonal",
                "padding": "same",
            }
            input = keras.Input(shape=(None, None, channels))
            x = layers.Conv2D(64, 5, **conv_args)(input)
            x = layers.Conv2D(64, 3, **conv_args)(x)
            x = layers.Conv2DTranspose(64, 1, strides=self.upscale_factor, **conv_args)(x)
            x = layers.Conv2D(64, 3, **conv_args)(x)
            x = layers.Conv2D(32, 3, **conv_args)(x)
            outputs = layers.Conv2D(channels, 3, **conv_args)(x)
            self.model = keras.Model(input, outputs)

    @staticmethod
    def plot_results(img, prefix, title, save_images: bool = False):
        """Plot the result with zoom-in area."""
        img_array = img_to_array(img)
        img_array = img_array.astype("float32") / 255.0

        # Create a new figure with a default 111 subplot.
        fig, ax = plt.subplots()
        im = ax.imshow(img_array[::-1], origin="lower")

        plt.title(title)
        """# zoom-factor: 2.0, location: upper-left
        axins = zoomed_inset_axes(ax, 2, loc=2)
        axins.imshow(img_array[::-1], origin="lower")

        # Specify the limits.
        x1, x2, y1, y2 = 200, 300, 100, 200
        # Apply the x-limits.
        axins.set_xlim(x1, x2)
        # Apply the y-limits.
        axins.set_ylim(y1, y2)"""

        plt.yticks(visible=False)
        plt.xticks(visible=False)

        # Make the line.
        """mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="blue")"""
        if save_images:
            plt.savefig(str(prefix) + "-" + title + ".png")
        plt.show()

    def _get_lowres_image(self, img):
        """Return low-resolution image to use as model input."""
        return img.resize(
            (img.size[0] // self.upscale_factor, img.size[1] // self.upscale_factor),
            PIL.Image.BICUBIC,
        )

    def upscale_image_from_path(self, img_path,
                                save_name: str = None,
                                save_image: bool = True):
        img = load_img(img_path)
        return self.upscale_image(img, save_name=save_name, save_image=save_image)

    def upscale_image(self, img,
                      save_name: str = None,
                      save_image: bool = True):
        """Predict the result based on input image and restore the image as RGB."""
        ycbcr = img.convert("YCbCr")
        y, cb, cr = ycbcr.split()
        y = img_to_array(y)
        y = y.astype("float32") / 255.0

        input = np.expand_dims(y, axis=0)
        out = self.model.predict(input)

        out_img_y = out[0]
        out_img_y *= 255.0

        # Restore the image in RGB color space.
        out_img_y = out_img_y.clip(0, 255)
        out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
        out_img_y = PIL.Image.fromarray(np.uint8(out_img_y), mode="L")
        out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
        out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
        out_img = PIL.Image.merge("YCbCr", (out_img_y, out_img_cb, out_img_cr)).convert(
            "RGB"
        )
        if save_name is None and save_image:
            out_img.save(f"test{random.randint(0, 9999)}.jpg")
        elif save_image:
            out_img.save(f"{save_name}.jpg")
        return out_img

    def load_training_data(self, dataset_path: str,
                           validation_split: float = 0.2,
                           batch_size: int = 8):
        self.dataset_path = dataset_path
        self.train_ds = image_dataset_from_directory(
            dataset_path,
            batch_size=batch_size,
            image_size=self.output_size,
            validation_split=validation_split,
            subset="training",
            seed=1337,
            label_mode=None
        )
        self.valid_ds = image_dataset_from_directory(
            dataset_path,
            batch_size=batch_size,
            image_size=self.output_size,
            validation_split=validation_split,
            subset="validation",
            seed=1337,
            label_mode=None
        )
        # scale rgb value
        self.train_ds = self.train_ds.map(self._scaling)
        self.valid_ds = self.valid_ds.map(self._scaling)
        # prepare inputs and outputs dataset
        self.train_ds = self.train_ds.map(lambda x: (self._process_input(x), self._process_target(x)))
        self.train_ds = self.train_ds.prefetch(buffer_size=32)

        self.valid_ds = self.valid_ds.map(lambda x: (self._process_input(x), self._process_target(x)))
        self.valid_ds = self.valid_ds.prefetch(buffer_size=32)

    def _set_loss_fns(self, loss_fns: list = ["MAE"]):
        self.loss_function = []
        mse_loss = keras.losses.mean_squared_error
        mae_loss = keras.losses.mean_absolute_error
        mape_loss = keras.losses.mean_absolute_percentage_error

        if "mse" in loss_fns or "MSE" in loss_fns:
            self.loss_function.append(mse_loss)
            print("MSE loss appended")
        if "mae" in loss_fns or "MAE" in loss_fns:
            self.loss_function.append(mae_loss)
            print("MAE loss appended")
        if "mape" in loss_fns or "MAPE" in loss_fns:
            self.loss_function.append(mape_loss)
            print("MAPE loss appended")
        if "ssim" in loss_fns or "SSIM" in loss_fns:
            self.loss_function.append(SSIM_loss)
            print("SSIM loss appended")
        if "psnr" in loss_fns or "PSNR" in loss_fns:
            self.loss_function.append(PSNR_loss)
            print("PSNR loss appended")
        if "ipcusl" in loss_fns or "IPCUSL" in loss_fns:
            self.loss_function.append(IPCUSL_loss)
            print("IPCUSL loss appended")
        if "charbonnier" in loss_fns or "CHARBONNIER" in loss_fns:
            self.loss_function.append(charbonnier_loss)
            print("charbonnier loss appended")
        if "tv" in loss_fns or "TV" in loss_fns:
            self.loss_function.append(TV_loss)
            print("total variation loss appended")
        if "tvd" in loss_fns or "TVD" in loss_fns:
            self.loss_function.append(TVD_loss)
            print("total variation diffrence loss appended")

        if len(self.loss_function) > 1:
            if self.loss_fn_weight is None:
                self.loss_fn_weight = [1 for _ in range(len(self.loss_function))]
            else:
                self.loss_fn_weight = self.loss_fn_weight
            self.loss_functions_list = self.loss_function
            self.loss_function = WS_loss(loss_fns=self.loss_functions_list,
                                         loss_fns_weight=self.loss_fn_weight)

    def _set_optimizer(self, optimizer: str = "Adam", lr: float = 0.005):
        if optimizer == "Adam":
            self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        elif optimizer == "Adagrad":
            self.optimizer = keras.optimizers.Adagrad(learning_rate=lr)
        elif optimizer == "Adamax":
            self.optimizer = keras.optimizers.Adamax(learning_rate=lr)
        elif optimizer == "Adadelta":
            self.optimizer = keras.optimizers.Adadelta(learning_rate=lr)
        elif optimizer == "SGD":
            self.optimizer = keras.optimizers.SGD(learning_rate=lr)
        elif optimizer == "RMSprop":
            self.optimizer = keras.optimizers.RMSprop(learning_rate=lr)
        elif optimizer == "Nadam":
            self.optimizer = keras.optimizers.Nadam(learning_rate=lr)
        else:
            print("Invalid optimizer specified it will be defaulted to Adam")
            self.optimizer = keras.optimizers.Adam(learning_rate=lr)

    def _set_metrics(self):
        self.metrics = ["mse", "mae", SSIM_loss, PSNR_metric, TV_loss, TVD_loss]

    def _compile_model(self):
        self._set_metrics()
        self.model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=self.metrics)

    def train(self, lr: float = 0.001,
              optimizer: str = "Adam",
              epochs: int = 50,
              ESPCNCallback_usage: bool = True,
              ESPCNCallback_test_path: str = "test",
              epoch_per_psnr: int = 20,
              psnr_plot: bool = True,
              model_type: str = "xs1",
              loss_fns: list = None,
              loss_fns_weight: list = None):

        """initilizing callbacks"""
        early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10)
        checkpoint_filepath = "./tmp/checkpoint"
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor="loss",
            mode="min",
            save_best_only=True,
        )

        callbacks = [early_stopping_callback, model_checkpoint_callback]
        if ESPCNCallback_usage:
            test_path = ESPCNCallback_test_path

            self.test_img_paths = sorted(
                [
                    os.path.join(test_path, fname)
                    for fname in os.listdir(test_path)
                    if fname.endswith(".jpg") or fname.endswith(".JPG")
                ]
            )
            callbacks.append(ESPCNCallback(upscale_factor=self.upscale_factor,
                                           test_img_paths=self.test_img_paths,
                                           psnr_plot=psnr_plot,
                                           epoch_per_psnr=epoch_per_psnr))

        """Building model"""
        self._build_model(model_type=model_type)

        """Printing model information"""
        self.model.summary()

        """Defining loss function and metrics"""
        self.loss_fn_weight = loss_fns_weight
        self._set_loss_fns(loss_fns=loss_fns)

        """Defining optimizer"""
        self._set_optimizer(optimizer, lr=lr)

        """Compiling model"""
        self._compile_model()

        callbacks.append(keras.callbacks.TensorBoard())
        """Fit the model"""
        print("Training")
        self.hist = self.model.fit(self.train_ds, epochs=epochs, callbacks=callbacks, validation_data=self.valid_ds,
                                   verbose=1)

        """The model weights (that are considered the best) are loaded into the model."""
        self.model.load_weights(checkpoint_filepath)

        """Auto save the model"""
        self.save_model()

        """Plot the results"""
        self._plot_statics()

    def fine_tuning(self, lr: float = 0.001,
                    epochs: int = 50,
                    model_name: str = "super_res",
                    ESPCNCallback_usage: bool = True,
                    ESPCNCallback_test_path: str = "test",
                    epoch_per_psnr: int = 20,
                    psnr_plot: bool = True,
                    recompile: bool = False,
                    optimizer: str = None,
                    loss_fns: list = None,
                    loss_fns_weight: list = None):

        """initilizing callbacks"""
        early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10)
        checkpoint_filepath = "./tmp/checkpoint"
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor="loss",
            mode="min",
            save_best_only=True,
        )

        callbacks = [early_stopping_callback, model_checkpoint_callback]
        if ESPCNCallback_usage:
            test_path = ESPCNCallback_test_path
            self.test_img_paths = sorted(
                [
                    os.path.join(test_path, fname)
                    for fname in os.listdir(test_path)
                    if fname.endswith(".jpg") or fname.endswith(".JPG")
                ]
            )
            callbacks.append(ESPCNCallback(upscale_factor=self.upscale_factor,
                                           test_img_paths=self.test_img_paths,
                                           psnr_plot=psnr_plot,
                                           epoch_per_psnr=epoch_per_psnr))

        """loading model"""
        self.load_model(model_name=model_name)

        """printing model summery"""
        self.model.summary()

        if recompile or optimizer is not None or loss_fns is not None:
            """Defining loss function and metrics"""
            self.loss_fn_weight = loss_fns_weight
            self._set_loss_fns(loss_fns=loss_fns)

            """Defining optimizer"""
            self._set_optimizer(optimizer, lr=lr)

            """Compiling model"""
            self._compile_model()

        callbacks.append(keras.callbacks.TensorBoard())
        """Fit the model"""
        print("Training")
        self.hist = self.model.fit(self.train_ds, epochs=epochs, callbacks=callbacks, validation_data=self.valid_ds,
                                   verbose=1)

        """The model weights (that are considered the best) are loaded into the model."""
        self.model.load_weights(checkpoint_filepath)

        """Auto save the model"""
        self.save_model(model_name=f"{model_name}_fine_tuned")

        """Plot the results"""
        self._plot_statics()

    def load_model(self, model_name: str = "super_res"):
        self.model = tf.keras.models.load_model(model_name, custom_objects={"SSIM_loss": SSIM_loss,
                                                                            "ssim_loss": SSIM_loss,
                                                                            "IPCUSL_loss": IPCUSL_loss,
                                                                            "PSNR_loss": PSNR_loss,
                                                                            "custom metric": SSIM_loss,
                                                                            "PSNR_metric": PSNR_metric,
                                                                            "charbonnier_loss": charbonnier_loss,
                                                                            "TV_loss": TV_loss,
                                                                            "TVD_loss": TVD_loss,
                                                                            "WS_loss": WS_loss})
        print(f"{model_name} loaded successfully")

    def save_model(self, model_name: str = "super_res"):
        self.model.save(model_name)
        print(f"{model_name} saved successfully")

    def _plot_statics(self):
        try:
            self._plot_loss()
        except:
            print("ploting loss failed")
        try:
            self._plot_acc()
        except:
            print("ploting accuracy failed")
        try:
            self._plot_mse()
        except:
            print("plotting mse failed")
        try:
            self._plot_mae()
        except:
            print("plotting mae failed")
        try:
            self._plot_ssim()
        except:
            print("plotting ssim failed")
        try:
            self._plot_all_metrics()
        except:
            print("plotting all metrics failed")

    def _plot_acc(self):
        fig = plt.figure()
        plt.plot(self.hist.history['accuracy'], color='teal', label='accuracy')
        plt.plot(self.hist.history['val_accuracy'], color='orange', label='val_accuracy')
        fig.suptitle('Accuracy', fontsize=20)
        plt.legend(loc="upper left")
        plt.grid()
        plt.show()

    def _plot_loss(self):
        fig = plt.figure()
        plt.plot(self.hist.history['loss'], color='teal', label='loss')
        plt.plot(self.hist.history['val_loss'], color='orange', label='val_loss')
        fig.suptitle('Loss', fontsize=20)
        plt.legend(loc="upper left")
        plt.grid()
        plt.show()

    def _plot_mse(self):
        fig = plt.figure()
        plt.plot(self.hist.history['mse'], color='teal', label='mse')
        plt.plot(self.hist.history['val_mse'], color='orange', label='val_mse')
        fig.suptitle('MSE', fontsize=20)
        plt.legend(loc="upper left")
        plt.grid()
        plt.show()

    def _plot_mae(self):
        fig = plt.figure()
        plt.plot(self.hist.history['mae'], color='teal', label='mae')
        plt.plot(self.hist.history['val_mae'], color='orange', label='val_mae')
        fig.suptitle('MAE', fontsize=20)
        plt.legend(loc="upper left")
        plt.grid()
        plt.show()

    def _plot_ssim(self):
        fig = plt.figure()
        plt.plot(self.hist.history['SSIMloss'], color='teal', label='SSIMloss')
        plt.plot(self.hist.history['val_SSIMloss'], color='orange', label='val_SSIMloss')
        fig.suptitle('SSIMloss', fontsize=20)
        plt.legend(loc="upper left")
        plt.grid()
        plt.show()

    def _plot_metrics_legacy(self):
        fig = plt.figure()
        plt.plot(self.hist.history['loss'], color='blue', label='loss')
        plt.plot(self.hist.history['val_loss'], color='orange', label='val_loss')

        plt.plot(self.hist.history['mse'], color='green', label='mse')
        plt.plot(self.hist.history['val_mse'], color='red', label='val_mse')

        plt.plot(self.hist.history['mae'], color='purple', label='mae')
        plt.plot(self.hist.history['val_mae'], color='brown', label='val_mae')

        plt.plot(self.hist.history['SSIMloss'], color='pink', label='SSIMloss')
        plt.plot(self.hist.history['val_SSIMloss'], color='gray', label='val_SSIMloss')

        plt.plot(self.hist.history['SSIMloss'], color='olive', label='SSIMloss')
        plt.plot(self.hist.history['val_SSIMloss'], color='cyan', label='val_SSIMloss')

        fig.suptitle('METRICS', fontsize=20)
        plt.legend(loc="upper left")
        plt.grid()
        plt.show()

    def _plot_all_metrics(self):
        fig = plt.figure()
        counter = 0
        for metric in self.hist.history:
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            plt.plot(self.hist.history[metric], color=color, label=f'{metric}')
            counter += 1
        fig.suptitle('METRICS', fontsize=20)
        plt.legend(loc="upper left")
        plt.grid()
        plt.show()
        plt.savefig()

    def benchmark(self,
                  image_path: str = "testimage.jpg",
                  input_size: tuple = (300, 300)):

        """Run benchmark on a single photo"""
        """setting the output size"""
        output_size = (input_size[0] * self.upscale_factor, input_size[1] * self.upscale_factor)
        """loading the test image"""
        test_image = load_img(image_path)
        """creating the ground truth"""
        ground_truth = test_image.resize(output_size)
        """down scaling the image"""
        down_scaled = test_image.resize(input_size)
        """upscaling with model"""
        up_scaled = self.upscale_image(down_scaled, save_image=False)
        """turnonhg them to arrays"""
        ground_truth_array = np.array(ground_truth)
        down_scaled_array = np.array(down_scaled)
        up_scaled_array = np.array(up_scaled)
        """calculating the diffrence between upscaled and ground truth"""
        diffrence_array = np.absolute(ground_truth_array - up_scaled_array)
        diffrence_photo = array_to_img(diffrence_array)
        mean_diffrence = np.mean(diffrence_array)
        print(f"Mean difference = {mean_diffrence}")
        """Ploting results"""
        self.plot_results(ground_truth, prefix="Ground_truth", title="Ground_truth")
        self.plot_results(down_scaled, prefix="Down scaled", title="Down scaled")
        self.plot_results(up_scaled, prefix="Up scaled", title="Up scaled")
        self.plot_results(diffrence_array, prefix="Diffrence", title="Diffrence")

    def benchmark_from_directory(self,
                                 image_directory_path: str = "test",
                                 input_size: tuple = None):
        if input_size is None:
            input_size = self.input_size

        test_path = image_directory_path
        test_img_paths = sorted(
            [
                os.path.join(test_path, fname)
                for fname in os.listdir(test_path)
                if fname.endswith(".jpg") or fname.endswith(".JPG")
            ]
        )
        mean_difreences = []
        ssims = []
        counter = 1
        results = ""
        ssim_results = ""
        for image_path in test_img_paths:
            """setting the output size"""
            output_size = (input_size[0] * self.upscale_factor, input_size[1] * self.upscale_factor)
            """loading the test image"""
            test_image = load_img(image_path)
            """creating the ground truth"""
            ground_truth = test_image.resize(output_size)
            """down scaling the image"""
            down_scaled = test_image.resize(input_size)
            """upscaling with model"""
            up_scaled = self.upscale_image(down_scaled, save_image=False)
            """turnonhg them to arrays"""
            ground_truth_array = np.array(ground_truth)
            down_scaled_array = np.array(down_scaled)
            up_scaled_array = np.array(up_scaled)
            """calculating the diffrence between upscaled and ground truth"""
            diffrence_array = np.absolute(ground_truth_array - up_scaled_array)
            diffrence_photo = array_to_img(diffrence_array)
            mean_diffrence = np.mean(diffrence_array)
            results = results + f"Image No{counter} Mean difference = {mean_diffrence} \n"
            mean_difreences.append(mean_diffrence)
            ssim = SSIM_loss(ground_truth_array, up_scaled_array)
            ssims.append(ssim)
            ssim_results = ssim_results + f"Image No{counter} SSIM = {ssim} \n"
            counter += 1

        mean_diffrences = np.array(mean_difreences)
        mean_mean_diffrences = np.mean(mean_diffrences)

        mean_ssim = np.mean(ssims)

        print(results)
        print(ssim_results)
        print(f"Mean of Mean Diffrences = {mean_mean_diffrences}")
        print(f"Mean of ssims = {mean_ssim}")

    @staticmethod
    def compare_images(image1=None,
                       image2=None,
                       image1_path: str = None,
                       image2_path: str = None,
                       save_result: bool = True):

        """check for input types"""
        if image1 is None:
            image1 = load_img(image1_path)

        if image2 is None:
            image2 = load_img(image2_path)

        if image1 is None and image1_path is None:
            print("You should pass a img as image1 or a path as image1_path")
            return

        if image2 is None and image2_path is None:
            print("You should pass a img as image2 or a path as image2_path")
            return

        """Converting images to numpy arrays"""
        image1_numpy = np.array(image1)
        image2_numpy = np.array(image2)

        deference_array = np.absolute(image1_numpy - image2_numpy)
        mean_deference = np.mean(deference_array)

        print(f"Mean deferance = {mean_deference}")

        if save_result:
            deference_image = array_to_img(deference_array)
            deference_image.save("deference_image.jpg")
