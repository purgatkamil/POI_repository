import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix
import numpy as np

# Funkcja do otwierania okna dialogowego dla wyboru pliku
def choose_file():
    root = tk.Tk()
    root.withdraw()  # Ukrycie głównego okna Tkinter
    path = filedialog.askopenfilename()  # Otwarcie okna dialogowego i zwrócenie wybranej ścieżki
    return path

# Wybieranie pliku przez użytkownika
file_path = choose_file()

# 2b. Wczytywanie danych z pliku csv
data = pd.read_csv(file_path)
print(data.columns)
# ii. Wyodrębnienie wektorów cech do macierzy X
X = data.drop('label', axis=1)  # Zastąp 'column_label' nazwą kolumny etykiet
# iii. Wyodrębnienie etykiet kategorii do wektora y
y = data['label']  # Zastąp 'column_label' nazwą kolumny etykiet

# 2c. Wstępne przetwarzanie danych
# i. Kodowanie całkowitoliczbowe dla wektora y
label_encoder = LabelEncoder()
y_int = label_encoder.fit_transform(y)
# ii. Kodowanie 1 z n dla wektora y_int (używając OneHotEncoder)
onehot_encoder = OneHotEncoder(sparse=False)
y_onehot = onehot_encoder.fit_transform(y_int.reshape(-1, 1))
# iii. Podzielenie zbioru X oraz wektora etykiet y_onehot
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.3, random_state=42)

# 2d. Tworzenie modelu sieci neuronowej
model = Sequential()
model.add(Dense(10, input_dim=X.shape[1], activation='sigmoid'))
model.add(Dense(3, activation='softmax'))
# Skompilowanie modelu
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 2e. Uczenie sieci
model.fit(X_train, y_train, epochs=50, batch_size=10)

# 2f. Testowanie sieci
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)
# Macierz pomyłek
conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
print(conf_matrix)
