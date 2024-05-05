import tkinter as tk
from tkinter import filedialog
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
# Wyodrębnienie wektorów cech do macierzy X
X = data.drop('label', axis=1)  # Zastąp 'column_label' nazwą kolumny etykiet
# Wyodrębnienie etykiet kategorii do wektora y
y = data['label']  # Zastąp 'column_label' nazwą kolumny etykiet

# 2c. Wstępne przetwarzanie danych
# Kodowanie całkowitoliczbowe dla wektora y
label_encoder = LabelEncoder()
y_int = label_encoder.fit_transform(y)
# Kodowanie 1 z n dla wektora y_int (używając OneHotEncoder)
onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoded = onehot_encoder.fit_transform(y_int.reshape(-1, 1))
# Podzielenie zbioru X oraz wektora etykiet onehot_encoded
X_train, X_test, y_train, y_test = train_test_split(X, onehot_encoded, test_size=0.3)

# 2d. Tworzenie modelu sieci neuronowej
model = Sequential()
model.add(Dense(10, input_dim=X.shape[1], activation='sigmoid'))
model.add(Dense(3, activation='softmax'))
# Skompilowanie modelu
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 2e. Uczenie sieci
model.fit(X_train, y_train, epochs=100, batch_size=10, shuffle=True)

# 2f. Testowanie sieci
y_pred = model.predict(X_test)
y_pred_int = np.argmax(y_pred, axis=1)
y_test_int = np.argmax(y_test, axis=1)
# Macierz pomyłek
cm = confusion_matrix(y_test_int, y_pred_int)
print(cm)
