
# Custom Intents

V0.1.2

Really high level API to create simple chat bots in few lines of code

the package is inspired by NeuralNine, neuralintents
package 

## Setting Up A Basic Chatbot

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

## Binding Functions To Requests

this is inspired by neuralintents

```python
from neuralintents import GenericAssistant
def function_for_greetings():
    print("You triggered the greetings intent!")
    # Some action you want to take
def function_for_stocks():
    print("You triggered the stocks intent!")
    # Some action you want to take
mappings = {'greeting' : function_for_greetings, 'stocks' : function_for_stocks}
assistant = GenericAssistant('intents.json', intent_methods=mappings ,model_name="test_model")
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
## Sample intents.json File
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

the first class in CustomIntent moudle is ChatBot
its exacly what you thing a chat bot

## Init arguaments

```python
def __init__(self, intents, intent_methods={}, model_name="assistant_model", threshold=0.25, w_and_b=False,
                 tensorboard=False):
```
intents : its the path of your intents file

intents_method :

model_name : its just the name of your model

threshold : its the accuracy threshold of your model its set to 0.25 by default

w_and_b : it will connect to wandb if set to True (you will need to login first)

tensorboard : Not available at the time

## Training

you can start training your model with one function call train_model

training model arguments :
```python
def train_model(self, epoch=500, batch_size=5, learning_rate=None, ignore_letters=None, timeIt=True, model_type='s1', validation_split=0):
```

epoch : an epoch basicly means training the neural network with all the training data for one cycle and this arguament says how many of this circles it will go

batch_size : Integer or None. Number of samples per gradient update (you can just ignore this)

learning_rate : Learning rate is a hyper-parameter that controls the weights of our neural network with respect to the loss gradient. It defines how quickly the neural network updates the concepts it has learned. (in simple terms if its bigger our model learn faster but it can go of track faster)

ignore_letters : a list of letters you want to ignore (by defualt it will ignore (? . , !) (you can pas a empty list if you dont want to ignore (?.,!)))

timeIt : it will just time the training

model_type : you can select one of the defined models (we will look at the available models later on)

validation_split : you can split a portion of your data for validation only (model will not get trained on them) it should be float between 0 and 1 (i will recommend to not create a validation split unless you have a really huge data set with lots of similar patterns)

## save_model

it will save your model as two .pkl files and a .h5 file

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

debug_mode : it will print every step of the procces for debuging

threshold : you can set a accuracy threshold if not specified it will use the  threshold you set when initilizing the bot and if you didnt specified there either its set to 0.25 by default

## request_response

the same as request_tag but it will return a random response from intents

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


# JsonIntents Class

this class is used to add and edit Json files containing intents
```python
def __init__(self, json_file_adrees):
```

you just need to pass the path of the json file the function you want

## add_pattern_app

its a function that ask you to input new patterns for tags (you can pass an especific tag to ask for that or it will cycle through them all and will go to the next tag by inputing D or d)

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