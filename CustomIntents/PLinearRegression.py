import json
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from random import random
import matplotlib.pyplot as plt
import csv
from CustomIntents.Pfunctions import ecualidean_distance, Ptimeit


class PLinearRegression:

    def __init__(self, data=None, x_axes1=None, y_axes1=None, model_name="test model"):
        self.y_avarage = None
        self.x_avarage = None
        self.data = data
        self.x_axes1 = x_axes1
        self.y_axes1 = y_axes1
        self.name = model_name
        self.result_a = None
        self.result_b = None

    def prepare_data(self):
        if self.data is None:
            self.data = np.array([self.x_axes1, self.y_axes1])

    def getting_avarages(self):
        self.x_avarage = np.sum(a=self.data[0]) / (len(self.data[0]))
        self.y_avarage = np.sum(a=self.data[1]) / (len(self.data[1]))

    def counting_up_down(self, a):
        uper = 0
        lower = 0
        for i in range(len(self.data[0])):
            x = self.data[0, i]
            if x < self.x_avarage:
                predicted_y = (a * x) + self.y_avarage - (a * self.x_avarage)
                actual_y = self.data[1, i]
                if predicted_y > actual_y:
                    uper += 1
                elif predicted_y < actual_y:
                    lower += 1
        result = None
        # 0 mean lower is more
        # 1 mean equal
        # 2 mean uper is more
        if lower > uper:
            result = 0
        elif lower == uper:
            result = 1
        else:
            result = 2
        return result, uper, lower

    def plot_input_data(self):
        plt.scatter(self.data[0], self.data[1])
        plt.grid()
        plt.show()

    def plot_prediction(self):
        plt.scatter(self.data[0], self.data[1])
        plt.grid()
        x_min = np.amin(self.data[0])
        x_max = np.amax(self.data[0])
        x = np.linspace(x_min, x_max, 100)
        y = (x * self.result_a) + self.result_b
        plt.plot(x, y, color="red")
        plt.show()

    def save_model_to_csv(self, file_dir="test_model.csv"):
        model_dict = [{"a": self.result_a, "b": self.result_b}]
        with open(file_dir, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["a", "b"])
            writer.writeheader()
            writer.writerows(model_dict)

    def load_model_from_csv(self, file_dir="test_model.csv"):
        with open(file_dir, "r") as csvfile:
            csvfile = csv.reader(csvfile)
            counter = 0
            for lines in csvfile:
                counter += 1
                if counter == 3:
                    self.result_a = float(lines[0])
                    self.result_b = float(lines[1])
                    print("model loaded succsesfully")
                    print(f"Line info : {self.result_a} X + {self.result_b}")

    def make_prediction(self, x):
        result = x * self.result_a + self.result_b
        return result

    @Ptimeit
    def algorythm_1(self, start_step=None, verbose=1, training_steps=10000, version1_1=False, plot_result=True):
        if start_step is None:
            start_step = 0.1
        step = start_step
        a = 1
        for _ in range(training_steps):
            self.counting_up_down(a=a)
            result, uper, lower = self.counting_up_down(a=a)
            if lower > uper:
                a -= step
            elif lower == uper:
                break
            elif lower < uper:
                a += step
            if verbose == 1:
                print(f"uper : {uper}")
                print(f"lower : {lower}")
                print(f"a : {a}")
                print(f"result : {result}")
                print("-----------------")
            if version1_1:
                step *= 0.999
        self.result_a = a
        self.result_b = self.y_avarage - (a * self.x_avarage)
        print(f"a = {self.result_a}\n"
              f"b = {self.result_b}")
        if plot_result:
            self.plot_prediction()

    @Ptimeit
    def algorythm_2(self, training_step=10000, learning_rate=0.01, verbose=1, plot_result=True):
        x = np.array(self.data[0])
        y = np.array(self.data[1])
        n_samples = len(x)
        weight = 0
        bias = 0

        for _ in range(training_step):
            y_pred = np.dot(x, weight) + bias

            dw = np.dot(x.T, (y_pred - y)) / n_samples * 2
            db = np.sum(y_pred - y) / n_samples * 2

            weight = weight - learning_rate * dw
            bias = bias - learning_rate * db
            if verbose == 1:
                print(f"a : {weight}")
                print(f"b : {bias}")
                print("-----------------")
        self.result_a = float(weight)
        self.result_b = float(bias)
        print(f"a = {self.result_a}\n"
              f"b = {self.result_b}")
        if plot_result:
            self.plot_prediction()

    def train_model(self, algorythm="1", training_steps=10000, start_step=None, verbose=1, plot_input_data=True,
                    learning_rate=0.01, plot_result=True):
        self.prepare_data()
        if plot_input_data:
            self.plot_input_data()
        self.getting_avarages()
        if algorythm == "1" or algorythm == "1.1":
            version1_1 = False
            if algorythm == "1.1":
                version1_1 = True
            self.algorythm_1(start_step=start_step, verbose=verbose,
                             training_steps=training_steps,
                             version1_1=version1_1, plot_result=plot_result)
            return self.result_a, self.result_b
        elif algorythm == "2":
            self.algorythm_2(training_step=training_steps, learning_rate=learning_rate, plot_result=plot_result)
        else:
            print(f"{bcolors.FAIL}this algorythm is not defiined !{bcolors.ENDC}")
            return

    @staticmethod
    def data_creator_scatter(a, b, noise_range, x_range, data_number):
        x_axes1 = []
        for _ in range(data_number):
            i = random() * x_range
            x_axes1.append(i)

        x_axes1 = np.array(x_axes1)
        y_axes1 = (x_axes1 * a) + b

        for i in range(len(y_axes1)):
            y_axes1[i] = y_axes1[i] + ((random() - 0.5) * noise_range)
        return np.array([x_axes, y_axes])


class PKNN:

    def __init__(self, k=3):
        self.y_train = None
        self.x_train = None
        self.k = k

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    def predict(self, x):
        predictions = [self._predict(curent_x) for curent_x in x]
        return predictions

    def _predict(self, x):
        # fasele
        distances = [ecualidean_distance(x, x_train1) for x_train1 in self.x_train]
        # nazdik tarin k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # majority vote
        most_common = Counter(k_nearest_labels).most_common()
        return most_common
