import random
import json
import pickle
import numpy as np
import os
from time import perf_counter
from collections import OrderedDict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Embedding, GlobalAveragePooling1D
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.python.keras.models import load_model

import wandb
from wandb.keras import WandbCallback

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)


class ChatBot:

    def __init__(self, intents, intent_methods={}, model_name="assistant_model", threshold=0.25, w_and_b=False,
                 tensorboard=False):
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

    def load_json_intents(self, intents):
        self.intents = json.loads(open(intents).read())

    def train_model(self, epoch=None, batch_size=5, learning_rate=None, ignore_letters=None, timeIt=True,
                    model_type='s1', validation_split=0):
        start_time = perf_counter()

        # defualt learning_rate

        if learning_rate is None:
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

        # defualt epoch
        if epoch is None:
            if model_type == "l1" or model_type == "xs2" or model_type == "s1" or model_type == "s2" or model_type == "s3" or model_type == "m1":
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
            self.model.add(Dense(32, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(len(train_y[0]), activation='softmax'))
        # defining layers end
        # printing summery
        print(self.model.summary())
        print(f"learning rate : {learning_rate}")
        print(f"epoch : {epoch}")
        print(f"validation split : {validation_split}")
        print(f"batch size : {batch_size}")
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
        sgd = SGD(learning_rate=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        self.hist = self.model.fit(np.array(train_x), np.array(train_y), epochs=epoch, batch_size=batch_size,
                                   verbose=1, validation_split=validation_split, callbacks=call_back_list)
        # training ends

        if timeIt:
            print(f"training time in sec : {perf_counter() - start_time}")
            print(f"training time in min : {(perf_counter() - start_time) / 60}")

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

    def request_response(self, message, debug_mode=False, threshold=None):
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

    def get_tag_by_id(self, id):
        pass

    def request_method(self, message):
        pass

    def request(self, message, threshold=None):
        ints = self._predict_class(message, threshold=threshold)

        if ints[0]['intent'] in self.intent_methods.keys():
            self.intent_methods[ints[0]['intent']]()
        else:
            return self._get_response(ints, self.intents)


class JsonIntents:
    def __init__(self, json_file_adrees):
        self.json_file_adress = json_file_adrees
        self.json_file = None
        if json_file_adrees.endswith(".json"):
            self.load_json_intents(json_file_adrees)

    def load_json_intents(self, json_file_adress):
        self.json_file = json.loads(open(json_file_adress).read())

    def add_pattern_app(self, tag=None):
        if tag is None:
            intents = self.json_file
            counter = 0

            for tag in (intents["intents"]):
                while True:
                    new_term = input(intents["intents"][counter]["tag"] + " : ")
                    if new_term.upper() == "D":
                        break
                    elif any(str(obj).lower() == new_term.lower() for obj in intents["intents"][counter]["patterns"]):
                        print("it exist ! ")
                    elif new_term.isspace() or new_term == "":
                        print("type a valid intent ! ")
                    else:
                        intents["intents"][counter]["patterns"] = list(intents["intents"][counter]["patterns"]).__add__(
                            [new_term])
                        print("added")
                counter += 1

            out_file = open(self.json_file_adress, "w")
            json.dump(intents, out_file)
            out_file.close()
            print("intents updated ! ")
            self.load_json_intents(self.json_file_adress)
        else:
            intents = self.json_file
            tag_counter = 0
            for i in (intents["intents"]):
                if intents["intents"][tag_counter]["tag"] == tag:
                    break
                else:
                    tag_counter += 1

            while True:
                new_term = input(intents["intents"][tag_counter]["tag"] + " : ")
                if new_term.upper() == "D":
                    break
                elif any(str(obj).lower() == new_term.lower() for obj in intents["intents"][tag_counter]["patterns"]):
                    print("it exist ! ")
                elif new_term.isspace() or new_term == "":
                    print("type a valid intent ! ")
                else:
                    intents["intents"][tag_counter]["patterns"] = list(
                        intents["intents"][tag_counter]["patterns"]).__add__([new_term])
                    print("added")

            out_file = open(self.json_file_adress, "w")
            json.dump(intents, out_file)
            out_file.close()
            print("intents updated ! ")
            self.load_json_intents(self.json_file_adress)

    def delete_duplicate_app(self):
        intents = self.json_file
        counter = 0

        for tag in (intents["intents"]):
            intents["intents"][counter]["patterns"] = list(
                OrderedDict.fromkeys(intents["intents"][counter]["patterns"]))
            counter += 1

        out_file = open(self.json_file_adress, "w")
        json.dump(intents, out_file)
        out_file.close()
        self.load_json_intents(self.json_file_adress)

    def add_tag_app(self, tag=None, responses=None):
        if tag is None and responses is None:
            json_file = self.json_file
            new_tag = input("what should the tag say ? ")
            responses = []
            while True:
                new_response = input("add a response for it : (d for done) ")
                if new_response.lower() == "d":
                    break
                else:
                    responses.append(new_response)
            json_file["intents"] = list(json_file["intents"]).__add__(
                [{"tag": [new_tag], "patterns": [], "responses": responses}])
            out_file = open(self.json_file_adress, "w")
            json.dump(json_file, out_file)
            out_file.close()
            print("new tag added !")
            self.load_json_intents(self.json_file_adress)
        elif tag is None:
            json_file = self.json_file
            new_tag = input("what should the tag say ? ")
            json_file["intents"] = list(json_file["intents"]).__add__(
                [{"tag": [new_tag], "patterns": [], "responses": responses}])
            out_file = open(self.json_file_adress, "w")
            json.dump(json_file, out_file)
            out_file.close()
            print("new tag added !")
            self.load_json_intents(self.json_file_adress)
        elif responses is None:
            json_file = self.json_file
            new_tag = tag
            responses = []
            while True:
                new_response = input("add a response for it : (d for done) ")
                if new_response.lower() == "d":
                    break
                else:
                    responses.append(new_response)
            json_file["intents"] = list(json_file["intents"]).__add__(
                [{"tag": [new_tag], "patterns": [], "responses": responses}])
            out_file = open(self.json_file_adress, "w")
            json.dump(json_file, out_file)
            out_file.close()
            print("new tag added !")
            self.load_json_intents(self.json_file_adress)
        else:
            json_file = self.json_file
            new_tag = tag
            json_file["intents"] = list(json_file["intents"]).__add__(
                [{"tag": [new_tag], "patterns": [], "responses": responses}])
            out_file = open(self.json_file_adress, "w")
            json.dump(json_file, out_file)
            out_file.close()
            print("new tag added !")
            self.load_json_intents(self.json_file_adress)
