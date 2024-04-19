import os
import numpy as np
import pandas as pd
from skimage import io, color, img_as_ubyte
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import filedialog

# Funkcja do wyboru katalogu przez użytkownika
def select_folder(prompt):
    root = tk.Tk()
    root.withdraw()  # Ukrycie głównego okna
    folder_selected = filedialog.askdirectory(title=prompt)
    root.destroy()
    return folder_selected

# Funkcja do wczytywania obrazów i wycinania próbek tekstury
def load_and_cut_images(directory, size):
    images = []  # Inicjalizacja pustej listy, która będzie przechowywać wycięte próbki tekstury.

    for filename in os.listdir(directory):  # Iteracja przez wszystkie pliki znajdujące się w podanym katalogu.
        img = io.imread(os.path.join(directory, filename))  # Wczytanie obrazu z dysku. Funkcja os.path.join łączy ścieżkę katalogu z nazwą pliku.

        for i in range(0, img.shape[0], size):  # Pętla przechodząca przez obraz w pionie z krokiem równym 'size'. img.shape[0] to wysokość obrazu.
            for j in range(0, img.shape[1], size):  # Pętla przechodząca przez obraz w poziomie z krokiem równym 'size'. img.shape[1] to szerokość obrazu.

                if i + size <= img.shape[0] and j + size <= img.shape[1]:  # Warunek sprawdzający, czy wycięty fragment obrazu nie wykracza poza jego granice.
                    images.append(img[i:i+size, j:j+size])  # Dodanie do listy 'images' fragmentu obrazu o rozmiarach 'size x size', wyciętego od pixela (i, j).

    return images  # Zwrócenie listy zawierającej wszystkie wycięte próbki tekstury.


# Funkcja do obliczania cech tekstury
def texture_features(images, distances, angles, properties):
    features = []  # Inicjalizacja pustej listy, która będzie przechowywać wektory cech dla każdego obrazu.

    for img in images:  # Iteracja po liście obrazów (próbek tekstury).
        gray = color.rgb2gray(img)  # Konwersja obrazu do skali szarości. Wynikowy obraz ma wartości od 0 (czarny) do 1 (biały).

        gray = img_as_ubyte(gray)  # Konwersja obrazu w skali szarości do 8-bitowej głębi kolorów, wartości od 0 do 255.

        gray //= 4  # Redukcja liczby poziomów szarości z 256 do 64, co jest wymagane przez funkcję greycomatrix.

        glcm = graycomatrix(gray, distances=distances, angles=angles, levels=64, symmetric=True, normed=True)  
        # Wyliczenie macierzy współwystępowania poziomów szarości (GLCM) dla obrazu.
        # 'distances' i 'angles' definiują dystanse i kąty dla których macierz jest tworzona,
        # 'levels' określa liczbę poziomów szarości,
        # 'symmetric' oznacza, że GLCM będzie symetryczna,
        # 'normed' mówi o normalizacji GLCM.

        feature_vector = []  # Inicjalizacja listy, która będzie przechowywać wartości cech dla obrazu.

        for prop in properties:  # Iteracja po liście nazw cech, które mają być wyliczone (np. 'contrast', 'dissimilarity').
            feature_vector.extend([graycoprops(glcm, prop).ravel()])  
            # Dodanie do wektora cech wyników funkcji greycoprops, która oblicza wybrane cechy z GLCM.
            # Metoda ravel() jest używana do spłaszczenia tablicy do jednego wymiaru.

        features.append(np.concatenate(feature_vector))  
        # Dodanie skonkatenowanego wektora cech do listy wszystkich cech, gdzie każdy element listy odpowiada jednemu obrazowi.

    return np.array(features)  # Konwersja listy cech do tablicy NumPy i zwrócenie jako wynik funkcji.


def classify_new_image(model, scaler):
    root = tk.Tk()  # Tworzenie głównego okna aplikacji Tkinter.
    root.withdraw()  # Tymczasowe ukrycie głównego okna, aby nie było widoczne podczas wyświetlania okna dialogowego.

    file_path = filedialog.askopenfilename(title="Wybierz obraz do klasyfikacji")  # Otwarcie okna dialogowego do wyboru pliku obrazu.
    root.destroy()  # Natychmiastowe zniszczenie głównego okna aplikacji po wybraniu pliku.

    if file_path:  # Warunek sprawdzający, czy użytkownik wybrał plik (file_path nie jest pusty).
        img = io.imread(file_path)  # Wczytanie obrazu z wybranej ścieżki.

        img_cut = img[:128, :128]  # Wycinanie fragmentu obrazu o rozmiarach 128x128 pikseli z lewego górnego rogu.
        # Zakładamy, że obraz ma przynajmniej 128x128 pikseli, co jest potrzebne do analizy.

        img_features = texture_features([img_cut], [1, 3, 5], [0, np.pi/4, np.pi/2, 3*np.pi/4], ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM'])
        # Wywołanie funkcji texture_features, aby obliczyć cechy tekstury dla wyciętego fragmentu obrazu.
        # Parametry funkcji definiują dystanse, kąty oraz typy cech tekstury do obliczenia.

        img_features_scaled = scaler.transform(img_features)  # Skalowanie obliczonych cech tekstury przy użyciu wcześniej dopasowanego skalera.

        prediction = model.predict(img_features_scaled)  # Użycie wytrenowanego modelu do przewidzenia kategorii tekstury na podstawie skalowanych cech.

        print("Przewidziana kategoria tekstury:", prediction[0])  # Wyświetlenie przewidzianej kategorii tekstury.


# Wczytywanie i przetwarzanie obrazów
# Użytkownik wybiera 3 katalogi zawierające obrazy do analizy tekstur.
texture_dirs = [select_folder(f'Wybierz katalog dla tekstury {i+1}') for i in range(3)] 

# Inicjalizacja pustej listy, która będzie przechowywać cechy wszystkich obrazów.
all_features = []

# Inicjalizacja pustej listy, która będzie przechowywać etykiety kategorii dla wszystkich obrazów.
labels = []

# Iteracja po każdym katalogu wybranym przez użytkownika.
for texture_dir in texture_dirs:
    # Wczytanie i wycięcie obrazów z danego katalogu.
    images = load_and_cut_images(texture_dir, 128)
    
    # Obliczenie cech tekstury dla wyciętych obrazów.
    features = texture_features(images, [1, 3, 5], [0, np.pi/4, np.pi/2, 3*np.pi/4], ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM'])
    
    # Dodanie obliczonych cech do listy wszystkich cech.
    all_features.append(features)
    
    # Dodanie etykiety kategorii do listy etykiet dla każdego obrazu w katalogu.
    labels.extend([os.path.basename(texture_dir)] * len(features))

# Konwersja listy cech do jednej tablicy NumPy i utworzenie DataFrame.
all_features = np.vstack(all_features)
df = pd.DataFrame(all_features)

# Dodanie kolumny z etykietami kategorii do DataFrame.
df['label'] = labels

# Zapisanie DataFrame do pliku CSV.
df.to_csv('texture_features.csv', index=False)

# Podział danych na zbiór treningowy i testowy.
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['label'], test_size=0.3, random_state=42)

# Skalowanie cech.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Inicjalizacja i trenowanie klasyfikatora SVM.
classifier = SVC()
classifier.fit(X_train_scaled, y_train)

# Klasyfikacja danych testowych i obliczenie dokładności.
y_pred = classifier.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

# Wyświetlenie dokładności klasyfikatora.
print(f'Accuracy: {accuracy:.2f}')


classify_new_image(classifier, scaler)
