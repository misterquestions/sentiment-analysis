import re

import numpy as np
import pandas as pd
import tensorflow as tf
from emoji import demojize
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout, GlobalMaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from prettytable import PrettyTable


class EmotionDetector:
    DATA_SET_FILE_PATH = 'tweet_emotions.csv'
    MODEL_FILE_PATH = 'emotion_detection.h5'

    def __init__(self, threshold=0.25):
        self.threshold = threshold
        self.max_len = 200
        self.embedding_dim = 100
        self.batch_size = 128
        self.epochs = 100
        self.tokenizer = Tokenizer()
        self.emotions = None
        self.model = None

    def load_data(self):
        # Cargue los datos del archivo CSV en un DataFrame pandas
        data = pd.read_csv('tweet_emotions.csv')

        # Barajea los datos de forma aleatoria para evitar cualquier orden no deseado
        data = data.sample(frac=1).reset_index(drop=True)

        # Extraiga las columnas "content" y "sentiment" como matrices NumPy
        X = data['content'].values
        y = data['sentiment'].values

        # Limpie los datos de texto para eliminar caracteres especiales, emojis y mayúsculas/minúsculas inconsistentes
        X_cleaned = [re.sub(r'[^\w\s#@/:%.,_-]', '', text) for text in X]
        X_cleaned = [re.sub(r'http\S+', '', text) for text in X_cleaned]
        X_cleaned = [re.sub(r'\d+', '', text) for text in X_cleaned]
        X_cleaned = [text.lower() for text in X_cleaned]
        X_cleaned = [demojize(text) for text in X_cleaned]

        # Ajuste un Tokenizer de Keras en los datos de texto para crear una correspondencia entre los tokens y los enteros
        self.tokenizer.fit_on_texts(X_cleaned)

        # Convierta las secuencias de texto a secuencias de enteros utilizando el Tokenizer ajustado
        X_cleaned = self.tokenizer.texts_to_sequences(X_cleaned)

        # Ajuste la longitud máxima de secuencia y rellene o corte las secuencias según sea necesario
        X_cleaned = pad_sequences(X_cleaned, maxlen=self.max_len, truncating='post', padding='post')

        # Extraiga las emociones únicas de la matriz de etiquetas y ordénelas alfabéticamente
        self.emotions = sorted(list(set(y)))

        # Codifique las emociones como vectores de clasificación binarios
        y = np.array([np.array([1 if emotion in s.split(',') else 0 for emotion in self.emotions]) for s in y])

        return X_cleaned, y

    def build_model(self, X):
        # Determine el tamaño del vocabulario para la capa de incrustación
        vocab_size = len(self.tokenizer.word_index) + 1

        # Cree un modelo secuencial
        self.model = Sequential()

        # Agregue una capa de incrustación para convertir los tokens enteros en vectores densos
        self.model.add(Embedding(vocab_size, self.embedding_dim, input_length=X.shape[1]))

        # Agregue una capa LSTM bidireccional para aprender patrones en los datos de texto
        self.model.add(Bidirectional(LSTM(128, return_sequences=True)))

        # Agregue una capa de Dropout para reducir el sobreajuste durante el entrenamiento
        self.model.add(Dropout(0.2))

        # Agregue otra capa LSTM bidireccional para aprender patrones en los datos de texto
        self.model.add(Bidirectional(LSTM(64, return_sequences=True)))

        # Agregue una capa de GlobalMaxPooling para reducir la dimensionalidad de las características de salida
        self.model.add(GlobalMaxPooling1D())

        # Agregue otra capa de Dropout para reducir el sobreajuste durante el entrenamiento
        self.model.add(Dropout(0.2))

        # Agregue una capa densa para clasificar los vectores de características reducidos
        self.model.add(Dense(64, activation='relu'))

        # Agregue otra capa de Dropout para reducir el sobreajuste durante el entrenamiento
        self.model.add(Dropout(0.2))

        # Agregue una capa densa final para clasificar las emociones
        self.model.add(Dense(len(self.emotions), activation='sigmoid'))

        # Compile el modelo con la función de pérdida y optimizador seleccionados
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_model(self, x_train, y_train, X_val, y_val):
        # Definir un EarlyStopping para detener el entrenamiento si no hay mejora después de 5 épocas
        # Utilizamos la métrica de pérdida y definimos una ventana de espera (patience) de 5
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Entrenar el modelo con los datos de entrenamiento y validación proporcionados
        # Utilizamos un tamaño de lote de self.batch_size y un número de épocas de self.epochs
        # Y también definimos el EarlyStopping que acabamos de crear
        history = self.model.fit(x_train, y_train, validation_data=(X_val, y_val),
                                 batch_size=self.batch_size, epochs=self.epochs, callbacks=[early_stop])

        # Imprimir la precisión (accuracy) durante el entrenamiento y la validación en cada época
        print('Exactitud (accuracy) durante el entrenamiento:')

        for acc in history.history['accuracy']:
            print('  {:.2f}%'.format(acc * 100))

        print('Exactitud (accuracy) durante la validación:')

        for acc in history.history['val_accuracy']:
            print('  {:.2f}%'.format(acc * 100))

        # Guardar el modelo entrenado en un archivo
        self.model.save(self.MODEL_FILE_PATH)

    def load_model(self):
        self.model = tf.keras.models.load_model(self.MODEL_FILE_PATH)

    def predict_emotions(self, text):
        text = [text]
        text = self.tokenizer.texts_to_sequences(text)
        text = pad_sequences(text, maxlen=self.max_len, truncating='post', padding='post')
        predictions = self.model.predict(text)
        results = {}

        for i in range(len(predictions[0])):
            if predictions[0][i] > self.threshold:
                results[self.emotions[i]] = predictions[0][i]

        return results


if __name__ == '__main__':
    # Crea un objeto EmotionDetector
    detector = EmotionDetector()
    # Carga los datos del archivo CSV y ajusta el Tokenizer
    X, y = detector.load_data()
    # Construye el modelo de la red neuronal
    detector.build_model(X)

    try:
        # Intenta cargar un modelo previamente guardado
        detector.load_model()
    except:
        # Si falla al cargar un modelo, divide los datos en conjuntos de entrenamiento y validación y entrena el modelo
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
        detector.train_model(X_train, y_train, X_val, y_val)

    # Bucle para que el usuario ingrese texto para detectar la emoción
    while True:
        text = input('Ingrese un texto para detectar la emoción: ')

        if text == 'q':
            break

        # Predice la emoción del texto ingresado
        emotions = detector.predict_emotions(text)

        # Imprime la tabla de emociones detectadas, ordenada de mayor a menor por probabilidad
        if len(emotions) == 0:
            print('No se detectaron emociones por encima del threshold.')
        else:
            table = PrettyTable()
            table.field_names = ['Emoción', 'Probabilidad']
            table.align = 'l'

            for k, v in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
                table.add_row([k, f'{v:.0%}'])

            print(f"Emociones detectadas en '{text}':")
            print(table)
