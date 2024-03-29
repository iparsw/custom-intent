
# Custom Intents

V1.0.0

(it's still a bit buggy)

## a simple way to create chatbots Ai, image classification Ai, image generation Ai, image supper resolution Ai and more!!

# installation

you can install the package from pypi (pip)

```commandline
pip install CustomIntents
```

## examples

![Demo1](images/img2.png)

![Demo2](images/img.png)



A package build on top of keras for creating and training deep learning chatbots (text classification), image classification, image generation Ai, image super resolution, style transforming and linear regression models in just three lines of code 

the package is inspired by NeuralNine, neuralintents package a packege that let you build chatbot models with 3 lines of code

# list of features:
* [SuperRes class](#superres-class)
* [ImageGenerator class](#imagegenerator-class)
* [ChatBot class](#chatbot-class)
* [JsonIntents](#jsonintents-class)
* [ImageClassificator class](#imageclassificator-class)
* [StyleTransformer class](#styletransformer-class)
* [BinaryImageClassificator](#binaryimageclassificator-class)
* [PLinearRegression class](#plinearregression-class)
* [scanner moudule](#scanner-moudule)

# SuperRes class

## Init arguments
```python
def __init__(self, input_size: tuple = (300, 300),
             upscale_factor: int = 3,
             cpu_only: bool = False):
```

input_size : this is only required when training or finetuning

upscale_factor : the upscale factor

cpu_only : whether to use only CPU or not

## upscale_image method
```python
def upscale_image(self, img,
                  save_name: str = None,
                  save_image: bool = True):
```
img : the image you want to upscale

save_image : whether to save the image

save_name : the name of the saved file

## upscale_image_from_path method
```python
def upscale_image_from_path(self, img_path,
                            save_name: str = None,
                            save_image: bool = True)
```
its like upscale_image method except that it requires a path to the image

## load_training_data method
```python
def load_training_data(self, dataset_path: str,
                       validation_split: float = 0.2,
                       batch_size: int = 8)
```
dataset_path : path to the training and validation data directory

validation_split : a float between 0 and 1 where the validation split is specified

batch_size : batch size

## train method
```python
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
```
optimizer : name of the optimizer (corently supported optimizers are : Adam, Adagrad, Adamax, Adadelta, SGD, RMSprop, Nadam)

epochs : number of epochs to train

ESPCNCallback_usage : wether to use ESPCNCallback or not

ESPCNCallback_test_path : a path to the test data directory for ESPCNCallback

epoch_per_psnr : if you use ESPCNCallback it will show a sample every few epochs you can choose that number here

psnr_plot : if you use ESPCNCall you can turn plutting a sample off

model_type : the type of the model (there are some different types in this package but for now i recommend using using the diffault xs1)

loss_fns : a list containing loss functions you want to use if you want only one loss function to use you can pass a list with only on loss functions 

(corently supported loss functions are : mse, mae, mape, ssim, psnr, ipcusl (this is a custom loss function for more information read IPCUSL.md), charbonnier, tv (total variation), tvd (total variation difference))

loss_fns_weight : if you use multiple loss functions the model will calculate their weighted sum and this is a list that contains their sum in this order (mse,mae,mape,ssim,psnr,ipcusl,charbonnier,tv,tvd)

## fine_tuning method
```python
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
                loss_fns_weight: list = None)
```
its almost equivalent to the train method except you should load a model first

## load_model method

```python
def load_model(self, model_name: str = "super_res")
```

model_name : the name of the model

## save_model method

```python
def save_model(self, model_name: str = "super_res")
```

model_name : the name of the model

## benchmark method

```python
def benchmark(self, image_path: str = "testimage.jpg",
              input_size: tuple = (300, 300))
```

this method will benchmark the model based on a single image

input_size : the image will be resized to the given size image and then upscaled by model

## benchmark_from_directory method

```python
def benchmark_from_directory(self, image_directory_path: str = "test",
                             input_size: tuple = None)
```

this method will benchmark the model based on a directory of images

input_size : the image will be resized to the given size image and then upscaled by model

## example of training a model
```python
from CustomIntents import SuperRes

model = SuperRes(input_size=(300, 300), upscale_factor=3)
model.load_training_data(dataset_path="dataset", batch_size=8)
model.train(epochs=5, model_type="xs1", psnr_plot=True, loss_fns=["IPCUSL"], epoch_per_psnr=4)
```
## example of fine tuning a model
```python
from CustomIntents import SuperRes

model = SuperRes(input_size=(100, 100), upscale_factor=3)
model.load_training_data(dataset_path="dataset", batch_size=32)
model.fine_tuning(epochs=2, lr=0.00008, model_name="CSR3X-1.1.3", psnr_plot=False, loss_fns=["mse", "mae"])
```
## example of using the model to generate upscaled images
```python
from CustomIntents import SuperRes

model = SuperRes(input_size=(300, 300), upscale_factor=3)
model.load_model("CSR3X-1.1.2")
model.upscale_image_from_path(img_path="test_image_2_300x300.jpg", save_name="test_result_300x300_to_900x900_7.jpg")
```

## image example
![DemoSuperRes](images/superres2.jpg)


# ImageGenerator class
you can easily use state of the art StableDiffiusion model with this class

## Init arguments
```python
def __init__(self, *,
             model: str = "StableDiffusion",
             img_height: int = 256,
             img_width: int = 256,
             jit_compile: bool = False,
             cpu_only: bool = False):
```

model : for now only StableDiffusion is available

img_height : it's the height of the genarated image it should be a multiple of 128

img_width : it's the width of the genarated image it should be a multiple of 128

jit_comple : it's a boolean indicating using just in time compliling

cpu_only : it's a boolean indicating whether to use CPU only or GPU

note every argument should be passed as an keyword argument

## generate method

```python
def generate(self, *,
             prompt: str = "Iron man making breakfast",
             batch_size: int = 1,
             filename: str = "sample",
             num_steps: int = 50):
```

prompt : it's the prompt to create the image from

batch_size : how many of images to create

filename : the name of file to save

num_steps : the number of steps to run the image through the model bigger the number it will generate better images but also it will take longer to generate

## gradio_preview method

this method will create a gradio preview

 this method doesn't get any arguments

## examples of using this class 

### creating a gradio preview

```python
from CustomIntents import ImageGenerator

model = ImageGenerator(model="StableDiffusion",
                       img_width=512,
                       img_height=512,
                       cpu_only=True,
                       jit_compile=True)

model.gradio_preview()
```

### generating an image

this code will generate two images and save them as "a sample image.jpg"  

```python
from ImageGenerator import ImageGenerator

model = ImageGenerator(model="StableDiffusion",
                       img_width=512,
                       img_height=512,
                       cpu_only=True,
                       jit_compile=True)

model.generate(prompt="a cat lying on a bed",
               batch_size=2,
               filename="a sample image",
               num_steps=50)
```

### Setting Up A Basic Chatbot

```python
from CustomIntents import ChatBot
chatbot = ChatBot(model_name="test_model", intents="intents.json")
assistant.train_model()
assistant.save_model()
done = False
while not done:
    message = input("Enter a message: ")
    if message == "STOP":
        done = True
    else:
        assistant.request(message)
```

### Binding Functions To Requests

this is inspired by neuralintents

```python
from CustomIntents import ChatBot
def function_for_greetings():
    print("You triggered the greetings intent!")
    # Some action you want to take
def function_for_stocks():
    print("You triggered the stocks intent!")
    # Some action you want to take
mappings = {'greeting' : function_for_greetings, 'stocks' : function_for_stocks}
assistant = ChatBot('intents.json', intent_methods=mappings ,model_name="test_model")
assistant.train_model()
assistant.save_model()
done = False
while not done:
    message = input("Enter a message: ")
    if message == "STOP":
        done = True
    else:
        assistant.request(message)
```
### Sample intents.json File
```json
{"intents": [
  {"tag": "greeting",
    "patterns": ["Hi", "Salam", "Nice to meet you", "Hello", "Good day", "Hey", "greetings"],
    "responses": ["Hello!", "Good to see you again!", "Hi there, how can I help?"]
  },
  {"tag": "goodbye",
    "patterns": ["bye", "good bye", "see you later"],
    "responses": ["bye", "good bye"],
    "context_set": ""
  },
  {"tag": "something",
    "patterns": ["something", "something else", "etc"],
    "responses": ["the response to something"],
  }
]
}
```

# ChatBot Class

the first class in CustomIntent package is ChatBot
its exacly what you thing a chatbot

## Init arguaments

```python
def __init__(self, intents, intent_methods, model_name="assistant_model", threshold=0.25, w_and_b=False,
             tensorboard=False):
```
intents : its the path of your intents file

intents_method : its a dictionary of mapped functions

model_name : its just the name of your model

threshold : its the accuracy threshold of your model its set to 0.25 by default

w_and_b : it will connect to wandb if set to True (you will need to login first)

tensorboard : Not available at the time

## Training

you can start training your model with one function call train_model

training model arguments :
```python
def train_model(self, epoch=None, batch_size=5, learning_rate=None,
                ignore_letters=None, timeIt=True, model_type='s1',
                validation_split=0, optimizer=None, accuracy_and_loss_plot=True):
```

epoch: An epoch refers to one cycle of training the neural network with all the training data. This argument specifies the number of cycles that the network will undergo.

batch_size: An integer or None. This determines the number of samples per gradient update. You can ignore this argument if you like.

learning_rate: The learning rate is a hyper-parameter that controls the weights of the neural network with respect to the loss gradient. It defines how quickly the network updates the concepts it has learned. In simple terms, a larger learning rate makes the model learn faster but it can also deviate from the correct path more easily.

ignore_letters: A list of letters that you want to ignore. By default, it ignores the characters (? . , !). You can pass an empty list if you don't want to ignore any characters.

timeIt: This argument times the training process.

model_type: You can select one of the predefined models (which will be described later).

validation_split: You can split a portion of your data for validation only, meaning the model will not be trained on these samples. This argument should be a float between 0 and 1. I recommend not creating a validation split unless you have a very large dataset with many similar patterns.

optimizer: You can choose between SGD, Adam, Adamx, and Adagard.

## save_model

it will save your model as two .pkl files and a .h5 file (don't add .h5 or .pkl)

```python
def save_model(self, model_name=None):
```

model_name : if its None (defualt), it will save the files like (model_name.h5)(model_name_words.pkl)(model_name_classes.pkl) where the model_name is the name you specified in the first place

## load_model

it will load a model from those three files

```python
def load_model(self, model_name=None):
```

model_name : if its None (defualt), it will look for files like (model_name.h5)(model_name_words.pkl)(model_name_classes.pkl) where the model_name is the name you specified in the first place

## request_tag

you will pass it a massege and it will return the predicted tag for you

```python
def request_tag(self, message, debug_mode=False, threshold=None):
```

message : the actual message

debug_mode : it will print every step of the procces for debuging perpes

threshold : you can set a accuracy threshold if not specified it will use the threshold you set when initilizing the bot and if you didn't specified there either it is set to 0.25 by default

## request_response

the same as request_tag but it will return a random response from intents

```python
def request_response(self, message, debug_mode=False, threshold=None):
```

## gradio_preview

it will open up a nice gui for testing your model in your browser

```python
def gradio_preview(self, ask_for_threshold=False, share=False, inbrowser=True):
```

ask_for_threshold : if set to True it will create a slider that you can set the threshold of the model with it

share : if set to True it will make the demo public

inbrowser : it will aoutomaticlly open a new browser page if set to True

![Demo](images/img.png)

## cli_preview

```python
def cli_preview(self):
```

a simple cli interface for testing your model

## gui_preview

a custom gui for triyng or even deploying your model

```python
def gui_preview(self, user_name=""):
```

user_name : it will only say hello to you if you pass your name for now

![Demo](images/img2.png)

## model types

you can choose one of the defined models according to the size of diffrente patterns and tags you have (you can just try and see wich one is right for your use case)

xs1 : a very fast and small model

xs2 : still small but better for more tags


s1 : the default model (hidden layers : 128-64)

s2 : its better than s1 when you have small number of similar tags that s1 cant predict

s3 : most of the time you dont need this (hidden layers : 128-64-64)

s4 : its like a s2 on streoid its suited when you have a lot of patterns for tags that have similar patterns

s5 : most of the time you dont need this either (hidden layers : 128-64-64-32)


m1 : great balance of perfomance and accuracy for medium size intent files

m2 : great accuracy for medium size intent files

m3 : m3 to m1 is like s2 to s1 its more suited when you have smaller number of tags but hard to difrentiat

l1 - l2 - l3 - l4 - l5 - xl1 - xl2 - xl3 - xl5 : are bigger models for more information read MODELS.md

# JsonIntents Class

this class is used to add and edit Json files containing intents
```python
def __init__(self, json_file_adrees):
```

you just need to pass the path of the json file the function you want

## add_pattern_app

its a function that ask you to input new patterns for tags (you can pass an especific tag to ask for that or it will cycle through them all and will go to the next tag by inputing D or d)

```python
def add_pattern_app(self, tag=None):
```

## add_tag_app

it will add new tags to your json file 

```python
def add_tag_app(self, tag=None, responses=None):
```
tag : the name of the new tag you want to add

responses : a list of responses (you can add later on as well)

## delete_duplicate_app

it will check for duplicate in patterns and deletes them for you

## an example of using this class

```python
file = JsonIntents("internet intents.json")
file.delete_duplicate_app()
file.add_tag_app(tag="about")
file.add_pattern_app("about")
```

# ImageClassificator class

the seccond class in CustomIntent package is ImageClassificator
it let you create and train deep learning image classification models with just three line of code !!

## init arguments
```python
def __init__(self, data_folder="data",
                 model_name="imageclassification_model",
                 number_of_classes=2,
                 classes_name=None,
                 gpu=None, 
                 checkpoint_filepath='/tmp/checkpoint_epoch:{epoch}'):
```

data_folder : the path to where you located your data

model_name : name your model

number_of_classes : number of difrent classes you have in your data (you should put the pictures of every class in a sub folder in your data folder)

classes_name : a list of names correspunding to your classes (they should be the same as the folder name of correspunding data folder for example if you have 3 sub folder in your data folder as banana apple pineapple you should pass ["banana", "apple", "pineapple"])

gpu : you can pass True or False if you dont pass anything it will try to use your gpu if you have a cuda enaibled graphic card and you have cudatoolkit and cuDNN installed and if you dont it will use your cpu

checkpoint_filepath : path to where you want your checkpoints

## Training

you can start training your model with one function call train_model

training model arguments :
```python
def train_model(self, epochs=20,
                    model_type="s1",
                    logdir=None,
                    optimizer_type="adam",
                    learning_rate=0.00001,
                    class_weight=None,
                    prefetching=False,
                    plot_model=True,
                    validation_split=0.2,
                    test_split=0,
                    augment_data=False,
                    tensorboard_usage=False,
                    stop_early=False,
                    checkpoint=False):
```

epoch : an epoch basicly means training the neural network with all the training data for one cycle and this arguament says how many of this circles it will go

model_type : you can select one of the defined models (we will look at the available models later on)

logdir : a directory to hold your tensorboard log files you can leave is empty if you dont care

optimizer_type : you can only choose adam right now

learning_rate : Learning rate is a hyper-parameter that controls the weights of our neural network with respect to the loss gradient. It defines how quickly the neural network updates the concepts it has learned. (in simple terms if its bigger our model learn faster but it can go of track faster)

class_weight : if you have an unbalanced dataset you can path a dictionary with the weight that you what to assosiate with every class () 

prefetching : prefetching data

plot_model : it will plot the model architecture for you 

validation_split : you can split a portion of your data for validation only (model will not get trained on them) it should be float between 0 and 1 (i will recommend to not create a validation split unless you have a really huge data set with lots of similar patterns)

augment_data : if set to true the model will also be trained on augmented data 

tensorboard_usage : it will use tensorboard

stop_early : if set to true it will stop training if validation loss is the same or increasing for more than 5 epochs

checkpoint : if set to true it will save checkpoints if the validation loss is the lovest ever seen

## save_model

it will save your model a .h5 file

```python
def save_model(self, model_file_name=None):
```

model_name : if its None (defualt), it will save the files like (model_name.h5) where the model_name is the name you specified in the first place

## load_model

it will load a model from those three files

```python
def load_model(self, name="imageclassification_model"):
```

model_name : if its None (defualt), it will look for files like (imageclassification_model.h5) 

## predict

now you can predict 

```python
def predict(self, image, image_type=None, full_mode=False, accuracy=False):
```

image : a path to an image file or a numpy array of the image or a cv2 image

image_type : if its None (defualt), it will try to detect if the image is a cv2 image or a numpy array of the image or a path to the image

full_mode : if you set it to true it will return every class and its probability

accuracy : if you set it to true it will return a tuple of the class name and the probability

(if both full_mode and accuracy set to false (defualt behavier) it will just return the most likly class name)

## predict_face

```python
    def predict_face(self, img, image_type=None, full_mode=False,
                     accuracy=False, return_picture=False,
                     save_returned_picture=False, saved_returned_picture_name="test.jpg",
                     show_returned_picture=False):
```

img : a path to an image file or a numpy array of the image or a cv2 image

image_type : if its None (defualt), it will try to detect if the image is a cv2 image or a numpy array of the image or a path to the image

full_mode : if you set it to true it will return every class and its probability

accuracy : if you set it to true it will return a tuple of the class name and the probability

return_picture : if set to true it will return a picture with faces in rectangles and their predicted class writen on top of them

save_returned_picture : if set to True it will save the returned picture

saved_returned_picture_name : if you set save_returned_picture to true you can use this to especifie the name of the saved picture

show_returned_picture : if set to true it will open the returned picture in a cv2 preview

## realtime_prediction

```python
def realtime_prediction(self, src=0):
```

src : if you have multiple webcams or virtual webcams it will let you choose from them if you only have one live it empty

## realtime face prediction

its exacly like the realtime_prediction() method but it will detect facec with a haarcascadde and will feed the model with the facec to predict

```python
def realtime_face_prediction(self, src=0, frame_rate=5):
```

src : if you have multiple webcams or virtual webcams it will let you choose from them if you only have one live it empty

frame_rate : its the number of frames to skip before predicting again


## gradio_preview

it will open up a nice gui for testing your model in your browser

```python
def gradio_preview(self, share=False, inbrowser=True, predict_face=False):
```

share : if set to True it will make the demo public

inbrowser : it will aoutomaticlly open a new browser page if set to True

predict_face : if set to True it will look for faces and feed them to the model 

## example of using ImageClassificator

in this example i have a folder in data/full that contains 4 sub folders (beni, khodm, matin, parsa) and in every one of them i have a lot of pictures of my friends (the folder name corredpunds to their names for example in beni folder there are beni's pictures, btw khodm means myself in my languge) and i want to train a model to detect which one of us we are in the picture

```python
from CustomIntents import ImageClassificator

model = ImageClassificator(model_name="test_m1", data_folder="data/full", number_of_classes=4, classes_name=["beni", "khodm", "matin", "parsa"])
model.train_model(epochs=10, model_type="m1", logdir="logs", learning_rate=0.0001, prefetching=False)
model.save_model(model_file_name="test_m1")
```
```python
from CustomIntents import BinaryImageClassificator

model = ImageClassificator(model_name="test_m1", data_folder="data/full", number_of_classes=4, classes_name=["beni", "khodm", "matin", "parsa"])
model.load_model(name="test_m1")
result = model.realtime_face_prediction()
```
![Demo3](images/realtime_prediction_example.png)
and as you see in the picture above you can see it under the that it is me in the picture with a really high accuracy

# StyleTransformer class

the fourth class in CustomIntent package is StyleTransformer
it let you transform an image to the style of another image

![Demo3](images/style_transformer.png)

## init arguments

```python
def __init__(self, image_path=None,
             style_reference_image_path=None,
             result_prefix="test_generated"):
```

image path : the path to the original image

style_reference_image_path : the path to the style reference image

result_prefix : the prefix for the result file

## transform method

the main method of this class

```python
def transfer(self, iterations=4000, iteration_per_save=100):
```

iterations : the number of iterations

iteration_per_save : save every _ iteration (where _ is the number you pass)

## gradio_preview method

a browser based app to use this class

```python
def gradio_preview(self, share=False, inbrowser=True):
```

share : if set to True it will make the demo public

inbrowser : it will aoutomaticlly open a new browser page if set to True

## example of using StyleTransformer

### passing the path of base and reference image

```python
from CustomIntents import StyleTransformer

model = StyleTransformer(image_path="base_image.jpg", style_reference_image_path="style_reference_image.jpg")
model.transfer(iterations=500, iteration_per_save=50)
```

this code will perform the teransformation for 500 times and save them every 50 steps

### Using gradio preview

```python
from CustomIntents import StyleTransformer

model = StyleTransformer()
model.gradio_preview()
```
![Demo3](images/dtyleTransformer_gradio.png)

#### *this model is slow so use reasonable iteration counts
don't use ridiculous numbers like 4000 like me, it took about 15 minutes on a 1660ti

# BinaryImageClassificator class

the fourth class in CustomIntent package is BinaryImageClassificator
it let you create and train deep learning image classification models with just three line of code !!

## Init arguaments

```python
def __init__(self, data_folder="data", model_name="imageclassification_model",
             first_class="1", second_class="2"):
```

data_folder : it's the path of your data folder (you should put your training images in two subfolder representing their label (class))

model_name : your model's name

first_class : you can name your classes so when you whant to predict it returns their name insted of 1s and 2s

seccond_class : //

## Training

you can start training your model with one function call train_model

training model arguments :
```python
def train_model(self, epochs=20, model_type="s1", logdir=None,
                optimizer_type="adam", learning_rate=0.00001,
                class_weight=None, prefetching=False, plot_model=True,
                validation_split=0.2):
```

epoch : an epoch basicly means training the neural network with all the training data for one cycle and this arguament says how many of this circles it will go

model_type : you can select one of the defined models (read MODELS.md for more information)

logdir : a directory to hold your tensorboard log files you can leave is empty if you don't care

optimizer_type : you can only choose adam right now

learning_rate : Learning rate is a hyper-parameter that controls the weights of our neural network with respect to the loss gradient. It defines how quickly the neural network updates the concepts it has learned. (in simple terms if its bigger our model learn faster but it can go of track faster)

class_weight : if you have an unbalanced dataset you can path a dictionary with the weight that you what to assosiate with every class () 

prefetching : prefetching data

plot_model : it will plot the model architecture for you 

validation_split : you can split a portion of your data for validation only (model will not get trained on them) it should be float between 0 and 1 (i will recommend to not create a validation split unless you have a really huge data set with lots of similar patterns)

## save_model

it will save your model a .h5 file (don't add .h5)

```python
def save_model(self, model_file_name=None):
```

model_name : if its None (defualt), it will save the files like (model_name.h5) where the model_name is the name you specified in the first place

## load_model

it will load a model from those three files

```python
def load_model(self, name="imageclassification_model"):
```

model_name : if its None (defualt), it will look for files like (imageclassification_model.h5) 

## predict

now you can predict 

```python
def predict(self, image, image_type=None, accuracy=False):
```

image : a path to an image file or a numpy array of the image or a cv2 image

image_type : if its None (defualt), it will it will try to detect if the image is a cv2 image or a numpy array of the image or a path to the image

accuracy : if you set it to true it will return a tuple of the class name and the probability


## predict from file path (legacy)

it will predict what class the image blongs to from a path

```python
def predict_from_files_path(self, image_file_path):
```

image_file_path : the path of the image you want to predict

it will return the name of the class and the percentage that its correct

## predict from imshow (legacy)

it will predict what class the image blongs to from a cv2 object

```python
def predict_from_imshow(self, img):
```

img : a cv2 image object 

it will return the name of the class and the percentage that its correct

## realtime prediction

it will predict from a live video feed (it will open a live cv2 video feed)

```python
def realtime_prediction(self, src=0):
```

src : if you have multiple webcams or virtual webcams it will let you choose from them if you only have one live it empty

## realtime face prediction

its exacly like the realtime_prediction() method but it will detect facec with a haarcascadde and will feed the model with the facec to predict

```python
def realtime_face_prediction(self, src=0):
```

src : if you have multiple webcams or virtual webcams it will let you choose from them if you only have one live it empty

## gradio_preview

it will open up a nice gui for testing your model in your browser

```python
def gradio_preview(self, share=False, inbrowser=True):
```

share : if set to True it will make the demo public

inbrowser : it will aoutomaticlly open a new browser page if set to True

## example of using BinaryImageClassificator

```python
from CustomIntents import BinaryImageClassificator

model = BinaryImageClassificator(model_name="test1", data_folder="data/parsa", first_class="sad", second_class="happy")
model.train_model(epochs=5, model_type="s1", logdir="logs", learning_rate=0.0001, prefetching=True) #, class_weight={0: 1.0, 1: 2.567750677506775})
model.save_model(model_file_name="test1")
```
```python
from CustomIntents import BinaryImageClassificator

model.load_model("models/test1")
model.realtime_face_prediction()
```

# PLinearRegression class

it's a simple linear regression class with one input and one output

## Init arguaments

```python
def __init__(self, data=None, x_axes1=None, y_axes1=None, model_name="test model"):
```

data : if you have your data as a aray like [[x_axes], [y_axes]] you can pass it here

x_axes1 : if you have your x values (inputs) and y values (output) seperetly you can pass the array that contains x valeus here

y_axes1 : if you have your x values (inputs) and y values (output) seperetly you can pass the array that contains y valeus here

model_name : name your model

## train model

you will train your model on your data with this method

```python
def train_model(self, algorythm="1", training_steps=10000,
                start_step=None, verbose=1, plot_input_data=True,
                learning_rate=0.01, plot_result=True):
```

algorythm : you can choose bitween 1, 1.1 and 2  (1 is really simple and fast but 2 is the propper linear reggresion one)

training_steps : it's how many steps you want to train your model

start_step : it's the starting stepfor algorythm 1 and 1.1

verbose : if it's set to 1 it will show you the details of trainong in every step

plot_input_data : it will plot the training data

learning_rate : it's the learning rate used for algorythm 2

plot_result : it will plot the line of best fit that it found along the side of the training data

## save the model to csv

it will save your model as a csv file containing the information of the line of  best fit

```python
def save_model_to_csv(self, file_dir="test_model.csv"):
```

file_dir : the name and dir you want to save your model to (include .csv)

## load model from csv

it will load your model from a csv file containing the information of the line of  best fit

```python
def load_model_from_csv(self, file_dir="test_model.csv")
```

file_dir : the name and dir you want to load your model from (include .csv)

## make prediction

it will make predictions for you

```python
def make_prediction(self, x):
```

x : your input data either in a numerical form or a numpy array containing multiple numerical values

it will return either a float (if you input is just a numerical value) or a numpy array containing multiple floats

# scanner moudule

this moudule is created for helping you scan faces this is helpful for person recognition emotion recognition etc.

## face_scanner function
```python
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
```

## facsScannerCliApp
this function will command app for facsScanner

```python
from CustomIntents import scanner

scanner.facsScannerCliApp()
```

## faceScannerGuiApp
it will command start a GUI app for faceScanner

```python
from CustomIntents import scanner

scanner.faceScannerGuiApp()
```

<h3>Visitors :</h3>
<br>
<img src="https://profile-counter.glitch.me/iparsw/count.svg" alt="Visitors">
