import os
import glob
import pickle
import numpy
import numpy as np
from music21 import converter, instrument, note, chord, stream
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop


class NoisyBoy:

    def __init__(self, model_name="NoisyBoy_test", model_type="s1", data_path="midi_songs", save_every_step=True):
        self.pitchnames = None
        self.batch_size = None
        self.epochs = None
        self.notes = None
        self.model = None
        self.n_vocab = None
        self.network_output = None
        self.network_input = None
        self.hist = None
        self.has_weight = False
        self.save_every_step = save_every_step
        self.data_path = data_path
        self.model_name = model_name
        self.model_type = model_type
        self.sequence_length = 100

    def train_model(self, epochs=200, batch_size=128):
        """ training """
        gpus = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(gpus, 'GPU')
        self.epochs = epochs
        self.batch_size = batch_size
        self._get_notes()
        # amount of pitches
        self.n_vocab = len(set(self.notes))
        self._prepare_sequences()
        self._create_model()
        print(self.model.summary())
        self._train()

    def _get_notes(self):
        """ Get all the notes and chords from the midi files in the ./midi_songs directory """
        self.notes = []

        for file in glob.glob(f"{self.data_path}/*.mid"):
            midi = converter.parse(file)

            print("Parsing %s" % file)

            notes_to_parse = None

            try:  # file has instrument parts
                s2 = instrument.partitionByInstrument(midi)
                notes_to_parse = s2.parts[0].recurse()
            except:  # file has notes in a flat structure
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    self.notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    self.notes.append('.'.join(str(n) for n in element.normalOrder))
        if os.path.isdir(f"{self.model_name}/data"):
            with open(f'{self.model_name}/data/notes', 'wb+') as filepath:
                pickle.dump(self.notes, filepath)
        else:
            os.makedirs(f"{self.model_name}/data")
            with open(f'{self.model_name}/data/notes', 'wb+') as filepath:
                pickle.dump(self.notes, filepath)

    def _prepare_sequences(self):
        """ Prepare the sequences used by the Neural Network """
        # get all pitch names
        self.pitchnames = sorted(set(item for item in self.notes))
        # create a dictionary to map pitches to integers
        note_to_int = dict((note, number) for number, note in enumerate(self.pitchnames))
        self.network_input = []
        self.network_output = []
        # create input sequences and the corresponding outputs
        for i in range(0, len(self.notes) - self.sequence_length, 1):
            sequence_in = self.notes[i:i + self.sequence_length]
            sequence_out = self.notes[i + self.sequence_length]
            self.network_input.append([note_to_int[char] for char in sequence_in])
            self.network_output.append(note_to_int[sequence_out])
        n_patterns = len(self.network_input)
        # reshape the input into a format compatible with LSTM layers
        self.network_input = numpy.reshape(self.network_input, (n_patterns, self.sequence_length, 1))
        # normalize input
        self.network_input = self.network_input / float(self.n_vocab)
        self.network_output = to_categorical(self.network_output)

    def _create_model(self):
        """ create the structure of the neural network """
        if self.model_type == "s1":
            self.model = Sequential()
            self.model.add(LSTM(512, input_shape=(self.network_input.shape[1], self.network_input.shape[2]),
                                recurrent_dropout=0, return_sequences=True, activation="tanh", unroll=False,
                                use_bias=True, recurrent_activation="sigmoid"))

            self.model.add(LSTM(512, return_sequences=True, activation="tanh",
                                recurrent_activation="sigmoid", recurrent_dropout=0,
                                unroll=False, use_bias=True))
            self.model.add(LSTM(512, activation="tanh", recurrent_activation="sigmoid",
                                recurrent_dropout=0, unroll=False, use_bias=True))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.3))
            self.model.add(Dense(256))
            self.model.add(Activation('relu'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.3))
            self.model.add(Dense(self.n_vocab))
            self.model.add(Activation('softmax'))
            self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        if self.model_type == "s2":
            self.model = Sequential()
            self.model.add(LSTM(1024, input_shape=(self.network_input.shape[1], self.network_input.shape[2]),
                                recurrent_dropout=0, return_sequences=True, activation="tanh", unroll=False,
                                use_bias=True, recurrent_activation="sigmoid"))

            self.model.add(LSTM(1024, return_sequences=True, activation="tanh",
                                recurrent_activation="sigmoid", recurrent_dropout=0,
                                unroll=False, use_bias=True))
            self.model.add(LSTM(1024, activation="tanh", recurrent_activation="sigmoid",
                                recurrent_dropout=0, unroll=False, use_bias=True))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.3))
            self.model.add(Dense(1024))
            self.model.add(Dense(1024))
            self.model.add(Activation('relu'))
            self.model.add(BatchNormalization())
            self.model.add(Dense(512))
            self.model.add(Dense(512))
            self.model.add(Activation('relu'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.3))
            self.model.add(Dense(self.n_vocab))
            self.model.add(Activation('softmax'))
            self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        if self.model_type == "m1":
            self.model = Sequential()
            self.model.add(LSTM(2048, input_shape=(self.network_input.shape[1], self.network_input.shape[2]),
                                recurrent_dropout=0, return_sequences=True, activation="tanh", unroll=False,
                                use_bias=True, recurrent_activation="sigmoid"))
            self.model.add(LSTM(2048, return_sequences=True, activation="tanh",
                                recurrent_activation="sigmoid", recurrent_dropout=0,
                                unroll=False, use_bias=True))
            self.model.add(LSTM(2048, return_sequences=True, activation="tanh", recurrent_activation="sigmoid",
                                recurrent_dropout=0, unroll=False, use_bias=True))
            self.model.add(LSTM(2048, activation="tanh", recurrent_activation="sigmoid",
                                recurrent_dropout=0, unroll=False, use_bias=True))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.2))
            self.model.add(Dense(2048, activation="relu"))
            self.model.add(Dense(2048, activation="relu"))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.4))
            self.model.add(Dense(1024, activation="relu"))
            self.model.add(Dense(1024, activation="relu"))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.4))
            self.model.add(Dense(512, activation="relu"))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.4))
            self.model.add(Dense(256, activation="relu"))
            self.model.add(Dense(self.n_vocab))
            self.model.add(Activation('softmax'))
            self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    def _train(self):
        """ train the neural network """
        filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
        callbacks_list = []
        if self.save_every_step:
            checkpoint = ModelCheckpoint(
                filepath,
                monitor='loss',
                verbose=0,
                save_best_only=True,
                mode='min'
            )
            callbacks_list.append(checkpoint)

        self.hist = self.model.fit(self.network_input, self.network_output, epochs=self.epochs,
                                   batch_size=self.batch_size,
                                   callbacks=callbacks_list)

    def generate(self, model_weight_path="test_model.hdf5", n_of_notes=500, verbose=0,
                 output_midi_file_name="test_output"):
        if self.notes is None:
            with open(f'{self.model_name}/data/notes', 'rb') as filepath:
                self.notes = pickle.load(filepath)
        if self.pitchnames is None:
            self.pitchnames = sorted(set(item for item in self.notes))
        if self.n_vocab is None:
            self.n_vocab = len(set(self.notes))
        if self.network_input is None:
            self._prepare_sequences()
        if self.model is None:
            self._create_model()
        if not self.has_weight:
            self.model.load_weights(model_weight_path)
        self._generate_note(n_of_notes=n_of_notes, verbose=verbose)
        self._create_midi(output_midi_file_name=output_midi_file_name)

    def _generate_note(self, n_of_notes=500, verbose=0):
        """ Generate notes from the neural network based on a sequence of notes """
        # pick a random sequence from the input as a starting point for the prediction
        start = numpy.random.randint(0, len(self.network_input) - 1)

        int_to_note = dict((number, note) for number, note in enumerate(self.pitchnames))

        pattern = self.network_input[start]
        self.prediction_output = []
        # generate notes
        for note_index in range(n_of_notes):
            if verbose == 1:
                print(f"note {note_index}")
            prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
            prediction_input = prediction_input / float(self.n_vocab)

            prediction = self.model.predict(prediction_input, verbose=verbose)

            index = int(numpy.argmax(prediction))
            result = int_to_note[index]
            self.prediction_output.append(result)
            pattern = np.append(pattern, index)
            pattern = pattern[1:len(pattern)]

    def _create_midi(self, output_midi_file_name="test_output"):
        """ convert the output from the prediction to notes and create a midi file
            from the notes """
        offset = 0
        output_notes = []

        # create note and chord objects based on the values generated by the model
        for pattern in self.prediction_output:
            # pattern is a chord
            if ('.' in pattern) or pattern.isdigit():
                notes_in_chord = pattern.split('.')
                notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(int(current_note))
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                new_chord = chord.Chord(notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
            # pattern is a note
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)

            # increase offset each iteration so that notes do not stack
            offset += 1

        midi_stream = stream.Stream(output_notes)

        midi_stream.write('midi', fp=f'{output_midi_file_name}.mid')

    def load_model(self, weight_path="NoisyBoy.hdf5"):
        if self.model is None:
            self._create_model()
        self.model.load_weights(weight_path)
        self.has_weight = True

    def save_model(self, model_name="test_model.hdf5"):
        self.model.save_weights(model_name)
