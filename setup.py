import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CustomIntents",
    version="0.8.0",
    author="Parsa Roshanak (iparsw)",
    author_email="parsaroshanak@gmail.com",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/iparsw/custom-intent.git",
    packages=["CustomIntents", "CustomIntents.Pfunction"],
    python_requires='>=3.9, <=3.11',
    include_package_data=True,
    package_data={'': ["cascades/haarcascade_frontalcatface.xml"]},
    install_requires=[
        "gmpy2==2.1.5",
        "gradio==3.16.2",
        "keras==2.10.0",
        "Keras_Preprocessing==1.1.2",
        "matplotlib==3.6.2",
        "nltk==3.8",
        "numpy==1.23.5",
        "opencv_python==4.7.0.68",
        "setuptools==65.7.0",
        "tensorflow==2.10.0",
        "wandb==0.13.7",
        "opencv-python==4.7.0.68",
        "music21==8.1.0",
        "future==0.18.2",
        "customtkinter==5.0.3"

    ]
)
