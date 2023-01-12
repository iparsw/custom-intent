
# Custom Intents

V0.7.2

A package build on top of keras for creating and training deep learning chatbots (text classification), binary image classification and linear regression models in just three lines of code 

the package is inspired by NeuralNine, neuralintents package a packege that let you build chatbot models with 3 lines of code

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

epoch : an epoch basicly means training the neural network with all the training data for one cycle and this arguament says how many of this circles it will go

batch_size : Integer or None. Number of samples per gradient update (you can just ignore this)

learning_rate : Learning rate is a hyper-parameter that controls the weights of our neural network with respect to the loss gradient. It defines how quickly the neural network updates the concepts it has learned. (in simple terms if its bigger our model learn faster but it can go of track faster)

ignore_letters : a list of letters you want to ignore (by defualt it will ignore (? . , !) (you can pas a empty list if you dont want to ignore (?.,!)))

timeIt : it will just time the training

model_type : you can select one of the defined models (we will look at the available models later on)

validation_split : you can split a portion of your data for validation only (model will not get trained on them) it should be float between 0 and 1 (i will recommend to not create a validation split unless you have a really huge data set with lots of similar patterns)

optimizer : you can choose beetwin SGD, Adam, Adamx and Adagard

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

# BinaryImageClassificator class

the first class in CustomIntent package is BinaryImageClassificate
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

model_type : you can select one of the defined models (we will look at the available models later on)

logdir : a directory to hold your tensorboard log files you can leave is empty if you dont care

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

## predict from file path

it will predict what class the image blongs to from a path

```python
def predict_from_files_path(self, image_file_path):
```

image_file_path : the path of the image you want to predict

it will return the name of the class and the percentage that its correct

## predict from imshow

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

## models

BinaryImageClassificator class have a verity of predefined models including

s1 : a small but powrful model

s1a : s1 but with augmentong data in random ways preventing the risk of overfiting to an extand

s2 :

s3 :

m1 :

for more nformation read MODELS.md file

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

