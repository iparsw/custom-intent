from keras_cv.models import StableDiffusion
from PIL import Image
import tensorflow.keras as keras
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import gradio as gr


class ImageGenerator:

    def __init__(self, *,
                 model: str = "StableDiffusion",
                 img_height: int = 256,
                 img_width: int = 256,
                 jit_compile: bool = False,
                 cpu_only: bool = False):

        self.prompt = None
        self.filename = None
        self.images = None
        self.batch_size = None
        self.StableDiffusion_use = False
        self.Dall_e_use = False
        self.model_type = model
        self._set_model_use()
        self.img_height = img_height
        self.img_width = img_width
        self.jit_compile = jit_compile
        self.cpu_only = cpu_only
        self._cpu_only()
        self._model_init()

    def _cpu_only(self):
        if self.cpu_only:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    def _set_model_use(self):
        if self.model_type == "StableDiffusion":
            self.StableDiffusion_use = True
        elif self.model_type == "Dall-E":
            self.Dall_e_use = True
        else:
            raise ValueError("model must be either 'StableDiffusion' or 'Dall-E'")

    def _model_init(self):
        if self.StableDiffusion_use:
            self._StableDiffusion_init()
        if self.Dall_e_use:
            self._Dall_init()

    def _StableDiffusion_init(self):
        self.model = StableDiffusion(img_height=self.img_height, img_width=self.img_width, jit_compile=self.jit_compile)

    def _Dall_E_init(self):
        pass

    def _plot_images(self):
        plt.figure(figsize=(20, 20))
        for i in range(len(self.images)):
            ax = plt.subplot(1, len(self.images), i + 1)
            plt.imshow(self.images[i])
            plt.axis("off")
        plt.show()

    def _save_images(self, model):
        for i in range(len(self.images)):
            Image.fromarray(self.images[i]).save(f"{model}-{self.filename}-{i}.png")

    def _StableDiffusion_generate(self, prompt, num_steps, batch_size):
        self.images = self.model.text_to_image(prompt, batch_size=batch_size, num_steps=num_steps)
        self._save_images("SD")
        self._plot_images()

    def generate(self, *,
                 prompt: str = "Iron man making breakfast",
                 batch_size: int = 1,
                 filename: str = "sample",
                 num_steps: int = 50):

        self.filename = filename
        if self.StableDiffusion_use:
            self._StableDiffusion_generate(prompt=prompt, num_steps=num_steps, batch_size=batch_size)

    def _gradio_generate(self, prompt, num_steps):
        num_steps = int(num_steps)
        self.images = self.model.text_to_image(prompt, batch_size=1, num_steps=num_steps)
        return self.images[0]

    def gradio_preview(self):
        demo = gr.Interface(self._gradio_generate,
                            inputs=[gr.Text(label="prompt"),
                                    gr.Number(label="number of steps")],
                            outputs=[gr.Image(label="output")],
                            allow_flagging="never")
        demo.launch(inbrowser=True)
