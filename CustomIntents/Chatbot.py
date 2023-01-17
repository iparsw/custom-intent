import json
import os
import pickle
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.metrics import Precision

import random

import wandb
from wandb.keras import WandbCallback
import matplotlib.pyplot as plt

from CustomIntents.Pfunction.Pfunctions import is_float
from CustomIntents.Bcolor import bcolors

import gradio as gr

from datetime import datetime
import tkinter as tk
import customtkinter as ctk
import textwrap


class ChatBot:

    def __init__(self, intents, intent_methods={}, model_name="assistant_model", threshold=0.25, w_and_b=False,
                 tensorboard=False):
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        self.model = None
        self.words = None
        self.classes = None
        self.hist = None
        self.intents = intents
        self.intent_methods = intent_methods
        self.model_name = model_name
        self.model_threshold = threshold
        self.w_and_b = w_and_b
        if intents.endswith(".json"):
            self.load_json_intents(intents)

        self.lemmatizer = WordNetLemmatizer()
        if w_and_b:
            wandb.init(project=model_name)
        if tensorboard:
            pass

    def load_json_intents(self, intents: str):
        self.intents = json.loads(open(intents).read())

    def train_model(self, epoch=None, batch_size=5, learning_rate=None, ignore_letters=None, timeIt=True,
                    model_type='s1', validation_split=0, optimizer=None, accuracy_and_loss_plot=True):
        start_time = perf_counter()

        # ckeing for right types of input
        # validation split
        if not is_float(validation_split):
            print(f"{bcolors.FAIL}validation split should be a float ! \n"
                  f"it will defualt to 0{bcolors.ENDC}")
            validation_split = 0
        else:
            if validation_split < 0 or validation_split >= 1:
                print(f"{bcolors.FAIL}validation split should be beetwen 0 and 1\n"
                      f"it will defualt to 0 {bcolors.ENDC}")
                validation_split = 0
        # ignore letters
        if type(ignore_letters) is not list and ignore_letters is not None:
            print(f"{bcolors.FAIL}ignore letters should be a list of letters you want to ignore\n"
                  f"it will set to defualt (['!', '?', ',', '.']){bcolors.ENDC}")
        # batch size
        if type(batch_size) is not int:
            print(f"{bcolors.FAIL}batch size should be an int\n"
                  f"it will set to defualt (5){bcolors.ENDC}")
        # timeIt
        if type(timeIt) is not bool:
            print(f"{bcolors.FAIL}timeIt should be a bool\n"
                  f"it will set to defualt (True)")
        # accuracy and loss plot
        if type(accuracy_and_loss_plot) is not bool:
            print(f"{bcolors.FAIL}accuracy_and_loss_plot should be a bool\n"
                  f"it will set to defualt (True)")

        # defualt optimizer
        if optimizer is None:
            optimizer = "Adam"
        # defualt learning_rate
        learning_rate_is_defualt = False
        if type(learning_rate) is int and learning_rate is not None:
            print(f"{bcolors.FAIL}learning rate should be an int\n"
                  f"it will defualt to defualt learning rate of the selected mdel{bcolors.ENDC}")
            learning_rate = None
        if learning_rate is None:
            learning_rate_is_defualt = True
            if model_type == "m2" or model_type == "s2" or model_type == "l1":
                learning_rate = 0.005
            elif model_type == "m3" or model_type == "s5" or model_type == "s4" or model_type == "s3":
                learning_rate = 0.001
            elif model_type == "l2":
                learning_rate = 0.0005
            elif model_type == "l3":
                learning_rate = 0.00025
            elif model_type == "l4":
                learning_rate = 0.0002
            elif model_type == "l5" or model_type == "l5f" or model_type == "xl1" or model_type == "xl2":
                learning_rate = 0.0001
            else:
                learning_rate = 0.01
        if learning_rate_is_defualt and optimizer == "Adamgrad":
            learning_rate = learning_rate * 50

        # defualt epoch
        if type(epoch) is not int and epoch is not None:
            print(f"{bcolors.FAIL}epochs should be an int\n"
                  f"it will defualt to defualt epoch of the selected mdel{bcolors.ENDC}")
            epoch = None
        if epoch is None:
            if model_type == "l1" or model_type == "xs2" or model_type == "s1" or model_type == "s2" or model_type == "s3" or model_type == "m1" or model_type == "m2":
                epoch = 200
            elif model_type == "xl2":
                epoch = 700
            elif model_type == "l3" or model_type == "l5f":
                epoch = 1000
            elif model_type == "l4" or model_type == "l5":
                epoch = 2000
            else:
                epoch = 500

        if ignore_letters is None:
            ignore_letters = ['!', '?', ',', '.']
        self.words = []
        self.classes = []
        documents = []

        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                word = nltk.word_tokenize(pattern)
                self.words.extend(word)
                documents.append((word, intent['tag']))
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in ignore_letters]
        self.words = sorted(list(set(self.words)))

        self.classes = sorted(list(set(self.classes)))

        training = []
        output_empty = [0] * len(self.classes)

        for doc in documents:
            bag = []
            word_patterns = doc[0]
            word_patterns = [self.lemmatizer.lemmatize(word.lower()) for word in word_patterns]
            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)

            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            training.append([bag, output_row])

        random.shuffle(training)
        training = np.array(training, dtype=object)

        train_x = list(training[:, 0])
        train_y = list(training[:, 1])

        print(f"model type = {model_type}")
        # defining layers start

        # xs1 model
        if model_type == "xs1":
            self.model = Sequential()
            self.model.add(Dense(32, input_shape=(len(train_x[0]),), activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(16, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(len(train_y[0]), activation='softmax'))
        # xs2 model
        elif model_type == "xs2":
            self.model = Sequential()
            self.model.add(Dense(64, input_shape=(len(train_x[0]),), activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(32, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(len(train_y[0]), activation='softmax'))
        # s1 model
        elif model_type == "s1":
            self.model = Sequential()
            self.model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(len(train_y[0]), activation='softmax'))
        # s2 model
        elif model_type == "s2":
            self.model = Sequential()
            self.model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(32, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(len(train_y[0]), activation='softmax'))
        # s3 model
        elif model_type == "s3":
            self.model = Sequential()
            self.model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(len(train_y[0]), activation='softmax'))
        # s4 model
        elif model_type == "s4":
            self.model = Sequential()
            self.model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(32, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(16, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(len(train_y[0]), activation='softmax'))
        # s5 model
        elif model_type == "s5":
            self.model = Sequential()
            self.model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(32, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(len(train_y[0]), activation='softmax'))
        # m1 model
        elif model_type == "m1":
            self.model = Sequential()
            self.model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(len(train_y[0]), activation='softmax'))
        # m2 model
        elif model_type == "m2":
            self.model = Sequential()
            self.model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(len(train_y[0]), activation='softmax'))
        # m3 model
        elif model_type == "m3":
            self.model = Sequential()
            self.model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(32, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(len(train_y[0]), activation='softmax'))
        # l1 model
        elif model_type == "l1":
            self.model = Sequential()
            self.model.add(Dense(512, input_shape=(len(train_x[0]),), activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(256, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(len(train_y[0]), activation='softmax'))
        # l2 model
        elif model_type == "l2":
            self.model = Sequential()
            self.model.add(Dense(512, input_shape=(len(train_x[0]),), activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(256, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(len(train_y[0]), activation='softmax'))
        # l3 model
        elif model_type == "l3":
            self.model = Sequential()
            self.model.add(Dense(512, input_shape=(len(train_x[0]),), activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(256, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(32, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(len(train_y[0]), activation='softmax'))
        # l4 model
        elif model_type == "l4":
            self.model = Sequential()
            self.model.add(Dense(512, input_shape=(len(train_x[0]),), activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(256, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(32, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(16, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(len(train_y[0]), activation='softmax'))
        # l5 model
        elif model_type == "l5" or model_type == "l5f":
            self.model = Sequential()
            self.model.add(Dense(512, input_shape=(len(train_x[0]),), activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(256, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dense(len(train_y[0]), activation='softmax'))
        # xl1 model
        elif model_type == "xl1":
            self.model = Sequential()
            self.model.add(Dense(1024, input_shape=(len(train_x[0]),), activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(512, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(256, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(len(train_y[0]), activation='softmax'))
        # xl2 model
        elif model_type == "xl2":
            self.model = Sequential()
            self.model.add(Dense(1024, input_shape=(len(train_x[0]),), activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(512, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(256, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(len(train_y[0]), activation='softmax'))
        # xl3 model
        elif model_type == "xl3":
            self.model = Sequential()
            self.model.add(Dense(1024, input_shape=(len(train_x[0]),), activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(512, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(256, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(len(train_y[0]), activation='softmax'))
        # xl4 model
        elif model_type == "xl4":
            self.model = Sequential()
            self.model.add(Dense(1024, input_shape=(len(train_x[0]),), activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(512, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(256, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(32, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(len(train_y[0]), activation='softmax'))
        # undifined model
        else:
            print(f"{bcolors.FAIL}model {model_type} is undifinde\n"
                  f"it will defuat to s1 {bcolors.ENDC}")
            self.model = Sequential()
            self.model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(len(train_y[0]), activation='softmax'))
        # defining layers end

        # callbacks define
        call_back_list = []
        # wheight and biases config
        if self.w_and_b:
            wandb.config = {
                "learning_rate": learning_rate,
                "epochs": epoch,
                "batch_size": batch_size
            }
            call_back_list.append(WandbCallback())

        # training start
        # SGD optimizer
        if optimizer == "SGD":
            opt = SGD(learning_rate=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        # Adama optimizer
        elif optimizer == "Adam":
            opt = Adam(learning_rate=learning_rate)
        # Adamax optimizer
        elif optimizer == "Adamx":
            opt = Adamax(learning_rate=learning_rate)
        # Adagrad optimizer
        elif optimizer == "Adagrad":
            opt = Adagrad(learning_rate=learning_rate)
        else:
            print(f"{bcolors.FAIL}the optimizer {optimizer} is unknown \n"
                  f"it will defualt to Adam optimizer{bcolors.ENDC}")
            opt = Adam(learning_rate=learning_rate)

        # printing summery
        print(self.model.summary())
        print(f"learning rate : {learning_rate}")
        print(f"epoch : {epoch}")
        print(f"validation split : {validation_split}")
        print(f"batch size : {batch_size}")
        print(f"optimizer : {optimizer}")

        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.hist = self.model.fit(np.array(train_x), np.array(train_y), epochs=epoch, batch_size=batch_size,
                                   verbose=1, validation_split=validation_split, callbacks=call_back_list)
        # training ends
        # training info plot
        if accuracy_and_loss_plot:
            history_dict = self.hist.history
            f_acc = history_dict['accuracy']
            f_loss = history_dict['loss']
            f_epochs = range(1, len(f_acc) + 1)
            plt.plot(f_epochs, f_loss, "b", label="Training los")
            f_val_acc = None
            if validation_split != 0:
                f_val_acc = history_dict['val_accuracy']
                f_val_loss = history_dict['val_loss']
                plt.plot(f_epochs, f_val_loss, 'r', label="Validation loss")
                plt.title('Training and validation loss')
            else:
                plt.title('Training loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid()
            plt.legend()
            plt.show()
            plt.plot(f_epochs, f_acc, 'b', label="Training acc")
            if validation_split != 0:
                plt.plot(f_epochs, f_val_acc, 'r', label="Validation acc")
                plt.title("Training and validation accuracy")
            else:
                plt.title("Training accuracy")
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.grid()
            plt.legend(loc="lower right")
            plt.show()
        # time
        if timeIt:
            print(f"training time in sec : {perf_counter() - start_time}")
            print(f"training time in min : {(perf_counter() - start_time) / 60}")
            print(f"training time in hour : {(perf_counter() - start_time) / 3600}")

    def save_model(self, model_name=None):
        if model_name is None:
            self.model.save(f"{self.model_name}.h5", self.hist)
            pickle.dump(self.words, open(f'{self.model_name}_words.pkl', 'wb'))
            pickle.dump(self.classes, open(f'{self.model_name}_classes.pkl', 'wb'))
        else:
            self.model.save(f"{model_name}.h5", self.hist)
            pickle.dump(self.words, open(f'{model_name}_words.pkl', 'wb'))
            pickle.dump(self.classes, open(f'{model_name}_classes.pkl', 'wb'))

    def load_model(self, model_name=None):
        if model_name is None:
            self.words = pickle.load(open(f'{self.model_name}_words.pkl', 'rb'))
            self.classes = pickle.load(open(f'{self.model_name}_classes.pkl', 'rb'))
            self.model = load_model(f'{self.model_name}.h5')
        else:
            self.words = pickle.load(open(f'{model_name}_words.pkl', 'rb'))
            self.classes = pickle.load(open(f'{model_name}_classes.pkl', 'rb'))
            self.model = load_model(f'{model_name}.h5')

    def _clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def _bag_of_words(self, sentence, words):
        sentence_words = self._clean_up_sentence(sentence)
        bag = [0] * len(words)
        for s in sentence_words:
            for i, word in enumerate(words):
                if word == s:
                    bag[i] = 1
        return np.array(bag)

    def _predict_class(self, sentence, threshold=None):
        if threshold is None:
            threshold = self.model_threshold
        p = self._bag_of_words(sentence, self.words)
        res = self.model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = threshold
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]], 'probability': str(r[1])})
        return return_list

    def _get_response(self, ints, intents_json):
        result = None
        try:
            tag = ints[0]['intent']
            list_of_intents = intents_json['intents']
            for i in list_of_intents:
                if i['tag'] == tag:
                    result = random.choice(i['responses'])
                    break
        except IndexError:
            result = "I don't understand!"
        return result

    def _get_tag(self, ints, intents_json):
        result = None
        try:
            tag = ints[0]['intent']
            list_of_intents = intents_json['intents']
            for i in list_of_intents:
                if i['tag'] == tag:
                    result = (i['tag'])
                    break
        except IndexError:
            result = "I don't understand!"
        return result

    def summery(self):
        return self.model.summary()

    def request_tag(self, message, debug_mode=False, threshold=None):
        if debug_mode:
            print(f"message = {message}")
            ints = self._predict_class(message, threshold=threshold)
            print(f"ints = {ints}")
            res = self._get_tag(ints, self.intents)
            print(f"res = {res}")
        else:
            ints = self._predict_class(message, threshold=threshold)
            res = self._get_tag(ints, self.intents)
        return res

    def request_response(self, message, threshold=None, debug_mode=False):
        if debug_mode:
            print(f"message = {message}")
            ints = self._predict_class(message, threshold=threshold)
            print(f"ints = {ints}")
            res = self._get_response(ints, self.intents)
            print(f"res = {res}")
        else:
            ints = self._predict_class(message, threshold=threshold)
            res = self._get_response(ints, self.intents)
        return res

    def get_tag_by_id(self, id1):
        pass

    def request_method(self, message):
        pass

    def request(self, message, threshold=None):
        ints = self._predict_class(message, threshold=threshold)

        if ints[0]['intent'] in self.intent_methods.keys():
            self.intent_methods[ints[0]['intent']]()
        else:
            return self._get_response(ints, self.intents)

    def _gradio_chatbot(self, message, history, threshhold=None):
        history = history or []
        message = message.lower()
        response = self.request_response(message, threshold=threshhold)
        history.append((message, response))
        return history, history

    def gradio_preview(self, ask_for_threshold=False, share=False, inbrowser=True):
        inputs = [gr.Textbox(lines=1, label="input"), "state"]
        if ask_for_threshold:
            inputs.append(gr.Slider(0, 1, label="threshhold"))
        chatbot = gr.Chatbot(label="chat bot").style(color_map=("green", "pink"))
        demo = gr.Interface(fn=self._gradio_chatbot,
                            inputs=inputs,
                            outputs=[chatbot, "state"],
                            allow_flagging="never")
        print(f"open http://localhost:7860 for viewing your model preview")
        demo.launch(share=share, inbrowser=inbrowser)

    def cli_preview(self):
        print("Print exit to leave")
        while True:
            text = input("YOU : ")
            if text == 'exit':
                break
            print("BOT : " + self.request_response(text))

        print("Ended successfully")

    def gui_preview(self, user_name=""):
        def new_massege(msg=None, bot_massage=None):
            # user message
            if msg is not None:
                ChatLog.config(state="normal")
                ChatLog.insert(tk.END, ' ' + current_time + ' ', ("small", "right", "greycolour"))
                ChatLog.window_create(tk.END, window=ctk.CTkLabel(ChatLog, text=msg,
                                                                  wraplength=300, font=("Arial", 12), justify="left",
                                                                  fg_color="#1f6aa5", text_color="#ffffff", padx=1,
                                                                  pady=5,
                                                                  corner_radius=10))
                ChatLog.insert(tk.END, '\n ', "left")
                ChatLog.config(foreground="#0000CC", font=("Helvetica", 10))
                ChatLog.yview(tk.END)
                ChatLog.config(state="disabled")
            # bot respons
            if bot_massage is None and msg is not None:
                res = self.request_response(msg)
            else:
                res = bot_massage
            ChatLog.config(state="normal")
            ChatLog.insert(tk.END, current_time + ' ', ("small", "greycolour", "left"))
            ChatLog.window_create(tk.END, window=ctk.CTkLabel(ChatLog, text=res,
                                                              wraplength=300, font=("Arial", 12), justify="left",
                                                              fg_color="#DDDDDD", text_color="#000000", padx=1, pady=5,
                                                              corner_radius=10))
            ChatLog.insert(tk.END, '\n ', "right")
            ChatLog.config(state="disabled")
            ChatLog.yview(tk.END)

        def send_by_enter(event):
            msg = EntryBox.get("1.0", 'end-1c').strip()
            EntryBox.delete("0.0", tk.END)

            if msg != '':
                new_massege(msg)

        def send_by_button():
            getmsg = EntryBox.get("1.0", 'end-1c').strip()
            msg = textwrap.fill(getmsg, 30)
            EntryBox.delete("0.0", tk.END)

            if msg != '':
                new_massege(msg)

        base = ctk.CTk()
        base.title("Chat Bot")
        base.geometry("400x500")
        base.resizable(width=False, height=False)

        now = datetime.now()
        current_time = now.strftime("%H:%M \n")

        # Create Chat window
        ChatLog = tk.Text(base, bd=0, height="8", width=50, font="Helvetica", wrap="word", background="#242424")
        ChatLog.config(state="normal")
        ChatLog.tag_config("right", justify="right")
        ChatLog.tag_config("small", font=("Helvetica", 8), foreground="#c7c7c7")
        ChatLog.tag_config("colour", foreground="#000000")
        ChatLog.config(state="disabled")
        # first message
        new_massege(bot_massage=f"Hello {user_name}, How can I assist you?")

        # Bind scrollbar to Chat window
        scrollbar = ctk.CTkScrollbar(base, command=ChatLog.yview, cursor="double_arrow")
        ChatLog['yscrollcommand'] = scrollbar.set

        # Create Button to send message
        SendButton = ctk.CTkButton(base, font=("Comic Sans MS", 15, 'bold'), text="Send", width=100, height=70,
                                   command=send_by_button)

        # Create the box to enter message
        EntryBox = ctk.CTkTextbox(base, width=29, height=5, font=("Arial", 17), wrap="word")

        # Place all components on the screen
        scrollbar.place(x=380, y=6, height=480)
        ChatLog.place(x=20, y=20, height=480, width=450)
        EntryBox.place(x=20, y=421, height=70, width=300)
        SendButton.place(x=280, y=421, height=70)

        base.bind('<Return>', send_by_enter)

        base.mainloop()


