import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CustomIntents",
    version="0.7.0",
    author="Parsa Roshanak (iparsw)",
    author_email="parsaroshanak@gmail.com",
    long_description=long_description,
    url="https://github.com/iparsw/custom-intent.git",
    packages=["CustomIntents"],
    python_requires='>=3.9, <3.11',
    include_package_data=True,
    package_data={'': ["cascades/haarcascade_frontalcatface.xml"]},
    install_requires=[
        "gmpy2==2.1.5",
        "keras==2.10.0",
        "Keras_Preprocessing==1.1.2",
        "matplotlib==3.6.2",
        "nltk==3.8",
        "numba==0.56.4",
        "numpy==1.23.5",
        "opencv_python==4.7.0.68",
        "tensorflow==2.10.0",
        "wandb==0.13.7"
    ]
)
