import math

import PIL
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset





def SSIM_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


def PSNR_loss_legacy(y_true, y_pred):
    return 100 - (10 * (K.log(K.mean(K.square(y_pred - y_true)) + 1) / K.log(10.)))


def PSNR_metric_legacy(y_true, y_pred):
    return 10 * (K.log(K.mean(K.square(y_pred - y_true)) + 1) / K.log(10.))


def PSNR_loss(y_true, y_pred):
    return 50 - tf.image.psnr(y_pred, y_true, max_val=1.)


def PSNR_metric(y_true, y_pred):
    return tf.image.psnr(y_pred, y_true, max_val=1.)


def TV_loss(y_true, y_pred):
    """
    total variation loss
    """
    return tf.reduce_sum(tf.image.total_variation(y_pred))


def TVD_loss(y_true, y_pred):
    """
    total variation diffrence loss
    it calculates the absoloute diffrence between y_pred and y_true
    """
    return K.abs(tf.reduce_sum(tf.image.total_variation(y_pred)) - tf.reduce_sum(tf.image.total_variation(y_true)))


def charbonnier_loss(y_true, y_pred):
    return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + tf.square(1e-3)))


def color_constancy_loss(y_true, y_pred):
    mean_rgb = tf.reduce_mean(y_pred, axis=(1, 2), keepdims=True)
    mr, mg, mb = mean_rgb[:, :, :, 0], mean_rgb[:, :, :, 1], mean_rgb[:, :, :, 2]
    d_rg = tf.square(mr - mg)
    d_rb = tf.square(mr - mb)
    d_gb = tf.square(mb - mg)
    return tf.sqrt(tf.square(d_rg) + tf.square(d_rb) + tf.square(d_gb))


def exposure_loss(y_true, y_pred, mean_val=0.6):
    y_pred = tf.reduce_mean(y_pred, axis=3, keepdims=True)
    mean = tf.nn.avg_pool2d(y_pred, ksize=16, strides=16, padding="VALID")
    return tf.reduce_mean(tf.square(mean - mean_val))


def IPCUSL_loss(y_true, y_pred):
    """
    iparsw custom up sampling loss function
    """
    return (charbonnier_loss(y_true, y_pred)) * (SSIM_loss(y_true, y_pred)) * (
        PSNR_loss(y_true, y_pred)) * (TV_loss(y_true, y_pred))


class WS_loss(keras.losses.Loss):
    def __init__(self, loss_fns: list = None,
                 loss_fns_weight: list = None,
                 name='weighted_sum'):
        super().__init__(name=name)
        self.loss_fns = loss_fns
        self.loss_fns_weight = loss_fns_weight

    def call(self, y_true, y_pred):
        loss_value = 0
        counter = 0
        for loss_function in self.loss_fns:
            loss_value += loss_function(y_true, y_pred) * self.loss_fns_weight[counter]
            counter += 1
        return loss_value

    def get_config(self):
        return {"loss_fns": self.loss_fns,
                "loss_fns_weight": self.loss_fns_weight}


class SpatialConsistencyLoss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(reduction="none")

        self.left_kernel = tf.constant(
            [[[[0, 0, 0]], [[-1, 1, 0]], [[0, 0, 0]]]], dtype=tf.float32
        )
        self.right_kernel = tf.constant(
            [[[[0, 0, 0]], [[0, 1, -1]], [[0, 0, 0]]]], dtype=tf.float32
        )
        self.up_kernel = tf.constant(
            [[[[0, -1, 0]], [[0, 1, 0]], [[0, 0, 0]]]], dtype=tf.float32
        )
        self.down_kernel = tf.constant(
            [[[[0, 0, 0]], [[0, 1, 0]], [[0, -1, 0]]]], dtype=tf.float32
        )

    def call(self, y_true, y_pred):
        original_mean = tf.reduce_mean(y_true, 3, keepdims=True)
        enhanced_mean = tf.reduce_mean(y_pred, 3, keepdims=True)
        original_pool = tf.nn.avg_pool2d(
            original_mean, ksize=4, strides=4, padding="VALID"
        )
        enhanced_pool = tf.nn.avg_pool2d(
            enhanced_mean, ksize=4, strides=4, padding="VALID"
        )

        d_original_left = tf.nn.conv2d(
            original_pool, self.left_kernel, strides=[1, 1, 1, 1], padding="SAME"
        )
        d_original_right = tf.nn.conv2d(
            original_pool, self.right_kernel, strides=[1, 1, 1, 1], padding="SAME"
        )
        d_original_up = tf.nn.conv2d(
            original_pool, self.up_kernel, strides=[1, 1, 1, 1], padding="SAME"
        )
        d_original_down = tf.nn.conv2d(
            original_pool, self.down_kernel, strides=[1, 1, 1, 1], padding="SAME"
        )

        d_enhanced_left = tf.nn.conv2d(
            enhanced_pool, self.left_kernel, strides=[1, 1, 1, 1], padding="SAME"
        )
        d_enhanced_right = tf.nn.conv2d(
            enhanced_pool, self.right_kernel, strides=[1, 1, 1, 1], padding="SAME"
        )
        d_enhanced_up = tf.nn.conv2d(
            enhanced_pool, self.up_kernel, strides=[1, 1, 1, 1], padding="SAME"
        )
        d_enhanced_down = tf.nn.conv2d(
            enhanced_pool, self.down_kernel, strides=[1, 1, 1, 1], padding="SAME"
        )

        d_left = tf.square(d_original_left - d_enhanced_left)
        d_right = tf.square(d_original_right - d_enhanced_right)
        d_up = tf.square(d_original_up - d_enhanced_up)
        d_down = tf.square(d_original_down - d_enhanced_down)
        return d_left + d_right + d_up + d_down


class ESPCNCallback(keras.callbacks.Callback):
    def __init__(self, upscale_factor, test_img_paths, psnr_plot, epoch_per_psnr: int = 20):
        super().__init__()
        self.psnr = None
        self.epoch_per_psnr = epoch_per_psnr
        self.psnr_plot = psnr_plot
        self.upscale_factor = upscale_factor
        self.test_img = self._get_lowres_image(load_img(test_img_paths[0]))

    # Store PSNR value in each epoch.
    def on_epoch_begin(self, epoch, logs=None):
        self.psnr = []

    def on_epoch_end(self, epoch, logs=None):
        """print(" Mean PSNR for epoch: %.2f" % (np.mean(self.psnr)))"""
        # was a bad idea -.-
        if epoch % self.epoch_per_psnr == 0 and self.psnr_plot:
            prediction = self.upscale_image(self.model, self.test_img)
            self.plot_results(prediction, "epoch-" + str(epoch), "prediction")

    def on_test_batch_end(self, batch, logs=None):
        try:
            self.psnr.append(10 * math.log10(1 / logs["loss"]))
        except:
            print("Loss is negetive")

    def _get_lowres_image(self, img):
        """Return low-resolution image to use as model input."""
        return img.resize(
            (img.size[0] // self.upscale_factor, img.size[1] // self.upscale_factor),
            PIL.Image.BICUBIC,
        )

    @staticmethod
    def upscale_image(model, img):
        """Predict the result based on input image and restore the image as RGB."""
        ycbcr = img.convert("YCbCr")
        y, cb, cr = ycbcr.split()
        y = img_to_array(y)
        y = y.astype("float32") / 255.0

        input = np.expand_dims(y, axis=0)
        out = model.predict(input)

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
        return out_img

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


class Involution(keras.layers.Layer):
    def __init__(
            self, channel, group_number, kernel_size, stride, reduction_ratio, name
    ):
        super().__init__(name=name)

        # Initialize the parameters.
        self.output_reshape = None
        self.input_patches_reshape = None
        self.kernel_reshape = None
        self.stride_layer = None
        self.kernel_gen = None
        self.channel = channel
        self.group_number = group_number
        self.kernel_size = kernel_size
        self.stride = stride
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        # Get the shape of the input.
        (_, height, width, num_channels) = input_shape

        # Scale the height and width with respect to the strides.
        height = height // self.stride
        width = width // self.stride

        # Define a layer that average pools the input tensor
        # if stride is more than 1.
        self.stride_layer = (
            keras.layers.AveragePooling2D(
                pool_size=self.stride, strides=self.stride, padding="same"
            )
            if self.stride > 1
            else tf.identity
        )
        # Define the kernel generation layer.
        self.kernel_gen = keras.Sequential(
            [
                keras.layers.Conv2D(
                    filters=self.channel // self.reduction_ratio, kernel_size=1
                ),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
                keras.layers.Conv2D(
                    filters=self.kernel_size * self.kernel_size * self.group_number,
                    kernel_size=1,
                ),
            ]
        )
        # Define reshape layers
        self.kernel_reshape = keras.layers.Reshape(
            target_shape=(
                height,
                width,
                self.kernel_size * self.kernel_size,
                1,
                self.group_number,
            )
        )
        self.input_patches_reshape = keras.layers.Reshape(
            target_shape=(
                height,
                width,
                self.kernel_size * self.kernel_size,
                num_channels // self.group_number,
                self.group_number,
            )
        )
        self.output_reshape = keras.layers.Reshape(
            target_shape=(height, width, num_channels)
        )

    def call(self, x):
        # Generate the kernel with respect to the input tensor.
        # B, H, W, K*K*G
        kernel_input = self.stride_layer(x)
        kernel = self.kernel_gen(kernel_input)

        # reshape the kerenl
        # B, H, W, K*K, 1, G
        kernel = self.kernel_reshape(kernel)

        # Extract input patches.
        # B, H, W, K*K*C
        input_patches = tf.image.extract_patches(
            images=x,
            sizes=[1, self.kernel_size, self.kernel_size, 1],
            strides=[1, self.stride, self.stride, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )

        # Reshape the input patches to align with later operations.
        # B, H, W, K*K, C//G, G
        input_patches = self.input_patches_reshape(input_patches)

        # Compute the multiply-add operation of kernels and patches.
        # B, H, W, K*K, C//G, G
        output = tf.multiply(kernel, input_patches)
        # B, H, W, C//G, G
        output = tf.reduce_sum(output, axis=3)

        # Reshape the output kernel.
        # B, H, W, C
        output = self.output_reshape(output)

        # Return the output tensor and the kernel.
        return output, kernel

    def get_config(self):
        return {"channel": self.channel,
                "group_number": self.group_number,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "reduction_ratio": self.reduction_ratio}
