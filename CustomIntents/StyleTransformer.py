import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19
import gradio as gr
from CustomIntents.Pfunction.Pfunctions import string_value_check, float_value_check, boolean_value_check, int_value_check


class StyleTransformer:

    def __init__(self, image_path=None,
                 style_reference_image_path=None,
                 result_prefix="test_generated"):

        ###############################
        # check that inputs are valid #
        ###############################
        if not string_value_check(image_path):
            raise ValueError("image_path must be a string")
        if not string_value_check(style_reference_image_path):
            raise ValueError("style_reference_image_path must be a string")
        if not string_value_check(result_prefix):
            raise ValueError("result_prefix must be a string")
        ################################

        self.img_ncols = None
        self.img_nrows = None
        self.height = None
        self.width = None
        self.image_path = image_path
        self.style_reference_image_path = style_reference_image_path
        self.result_prefix = result_prefix
        # Weights of the different loss components
        self.total_variation_weight = 1e-6
        self.style_weight = 1e-6
        self.content_weight = 2.5e-8

    def _preprocess_image(self, image_path, image_type="path"):
        img = None
        if image_type == "path":
            # Util function to open, resize and format pictures into appropriate tensors
            img = keras.preprocessing.image.load_img(
                image_path, target_size=(self.img_nrows, self.img_ncols)
            )
        elif image_type == "array":
            img = keras.preprocessing.image.array_to_img(image_path)
            img = img.resize(size=(self.img_nrows, self.img_ncols))
        img = keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = vgg19.preprocess_input(img)
        return tf.convert_to_tensor(img)

    def _deprocess_image(self, x):
        # Util function to convert a tensor into a valid image
        x = x.reshape((self.img_nrows, self.img_ncols, 3))
        # Remove zero-center by mean pixel
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        # 'BGR'->'RGB'
        x = x[:, :, ::-1]
        x = np.clip(x, 0, 255).astype("uint8")
        return x

    @staticmethod
    def _gram_matrix(x):
        # The gram matrix of an image tensor (feature-wise outer product)
        x = tf.transpose(x, (2, 0, 1))
        features = tf.reshape(x, (tf.shape(x)[0], -1))
        gram = tf.matmul(features, tf.transpose(features))
        return gram

    def _style_loss(self, style, combination):
        # The "style loss" is designed to maintain
        # the style of the reference image in the generated image.
        # It is based on the gram matrices (which capture style) of
        # feature maps from the style reference image
        # and from the generated image
        S = self._gram_matrix(style)
        C = self._gram_matrix(combination)
        channels = 3
        size = self.img_nrows * self.img_ncols
        return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

    @staticmethod
    def _content_loss(base, combination):
        # An auxiliary loss function
        # designed to maintain the "content" of the
        # base image in the generated image
        return tf.reduce_sum(tf.square(combination - base))

    def _total_variation_loss(self, x):
        # The 3rd loss function, total variation loss,
        # designed to keep the generated image locally coherent
        a = tf.square(
            x[:, : self.img_nrows - 1, : self.img_ncols - 1, :] - x[:, 1:, : self.img_ncols - 1, :]
        )
        b = tf.square(
            x[:, : self.img_nrows - 1, : self.img_ncols - 1, :] - x[:, : self.img_nrows - 1, 1:, :]
        )
        return tf.reduce_sum(tf.pow(a + b, 1.25))

    def _buid_vgg19(self):
        # Build a VGG19 model loaded with pre-trained ImageNet weights
        self.model = vgg19.VGG19(weights="imagenet", include_top=False)
        # Get the symbolic outputs of each "key" layer (we gave them unique names).
        self.outputs_dict = dict([(layer.name, layer.output) for layer in self.model.layers])
        # Set up a model that returns the activation values for every layer in
        # VGG19 (as a dict).
        self.feature_extractor = keras.Model(inputs=self.model.inputs, outputs=self.outputs_dict)
        # List of layers to use for the style loss.
        self.style_layer_names = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ]
        # The layer to use for the content loss.
        self.content_layer_name = "block5_conv2"

    def _compute_loss(self, combination_image, base_image, style_reference_image):
        input_tensor = tf.concat(
            [base_image, style_reference_image, combination_image], axis=0
        )
        features = self.feature_extractor(input_tensor)
        # Initialize the loss
        loss = tf.zeros(shape=())
        # Add content loss
        layer_features = features[self.content_layer_name]
        base_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        loss = loss + self.content_weight * self._content_loss(
            base_image_features, combination_features
        )
        # Add style loss
        for layer_name in self.style_layer_names:
            layer_features = features[layer_name]
            style_reference_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = self._style_loss(style_reference_features, combination_features)
            loss += (self.style_weight / len(self.style_layer_names)) * sl
        # Add total variation loss
        loss += self.total_variation_weight * self._total_variation_loss(self.combination_image)
        return loss

    @tf.function
    def _compute_loss_and_grads(self, combination_image, base_image, style_reference_image):
        with tf.GradientTape() as tape:
            loss = self._compute_loss(combination_image, base_image, style_reference_image)
        grads = tape.gradient(loss, combination_image)
        return loss, grads

    def _build_opimizer(self):
        self.optimizer = keras.optimizers.SGD(
            keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
            )
        )

    def _prepare_pictures(self):
        self.base_image = self._preprocess_image(self.image_path)
        self.style_reference_image = self._preprocess_image(self.style_reference_image_path)
        self.combination_image = tf.Variable(self._preprocess_image(self.image_path))

    def transfer(self, iterations=4000):

        ###############################
        # check that inputs are valid #
        ###############################
        if not int_value_check(iterations, start=50):
            raise ValueError('Invalid number of iterations the number of iterations must be a integer and bigger than 50')
        ###############################

        # Dimensions of the generated picture.
        self.width, self.height = keras.preprocessing.image.load_img(self.image_path).size
        self.img_nrows = 400
        self.img_ncols = int(self.width * self.img_nrows / self.height)
        self._build_opimizer()
        self._buid_vgg19()
        self._prepare_pictures()
        for i in range(1, iterations + 1):
            loss, grads = self._compute_loss_and_grads(
                self.combination_image, self.base_image, self.style_reference_image
            )
            self.optimizer.apply_gradients([(grads, self.combination_image)])
            if i % 50 == 0:
                print("Iteration %d: loss=%.2f" % (i, loss))
                img = self._deprocess_image(self.combination_image.numpy())
                fname = self.result_prefix + "_at_iteration_%d.png" % i
                keras.preprocessing.image.save_img(fname, img)

    def _gradio_transfer(self, base_image, style_reference_image, iterations):
        keras.preprocessing.image.save_img("temp_base_image_temprory.png", base_image)
        keras.preprocessing.image.save_img("temp_style_reference_image_temprory.png", style_reference_image)
        self.image_path = "temp_base_image_temprory.png"
        self.style_reference_image_path = "temp_style_reference_image_temprory.png"
        # actuall transform
        self.width, self.height = keras.preprocessing.image.load_img(self.image_path).size
        self.img_nrows = 400
        self.img_ncols = int(self.width * self.img_nrows / self.height)
        self._build_opimizer()
        self._buid_vgg19()
        self._prepare_pictures()
        img = None
        iterations = int(iterations)
        for i in range(1, iterations + 1):
            loss, grads = self._compute_loss_and_grads(
                self.combination_image, self.base_image, self.style_reference_image
            )
            self.optimizer.apply_gradients([(grads, self.combination_image)])
            if i % 10 == 0:
                print("Iteration %d: loss=%.2f" % (i, loss))
                img = self._deprocess_image(self.combination_image.numpy())
                fname = self.result_prefix + "_at_iteration_%d.png" % i
        return img

    def gradio_preview(self, share=False, inbrowser=True):

        ###############################
        # check that inputs are valid #
        ###############################
        if not boolean_value_check(share) and not bool_value_check(inbrowser):
            raise ValueError("share and inbrowser must both be boolean")
        ###############################

        demo = gr.Interface(self._gradio_transfer,
                            inputs=[
                                gr.Image(label="base image"),
                                gr.Image(label="style reference image"),
                                gr.Number(label="number of iterations", value=200)
                            ],
                            outputs=[
                                gr.Image(label="output")
                            ], allow_flagging="never")
        print(f"open http://localhost:7860 for viewing your model preview")
        demo.launch(share=share, inbrowser=inbrowser)
