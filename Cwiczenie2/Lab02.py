"""
Program do analizy chmur punktów 3D.

Program umożliwia wczytanie chmury punktów z pliku .xyz, dopasowanie do niej płaszczyzny za pomocą algorytmu RANSAC,
oraz podział punktów na rozłączne chmury za pomocą algorytmu k-średnich (k-means). Użytkownik może wybrać plik z danymi
za pomocą interfejsu graficznego. Następnie program wizualizuje wyniki, w tym najlepiej dopasowaną płaszczyznę i wynik
klasyfikacji punktów do poszczególnych chmur.

Funkcje:
- fit_plane_ransac: Dopasowuje płaszczyznę do chmury punktów za pomocą algorytmu RANSAC.
- find_clusters_with_kmeans: Dzieli chmurę punktów na rozłączne chmury za pomocą algorytmu k-średnich.
- plot_clusters: Wizualizuje wyniki klasteryzacji różnymi kolorami dla każdej chmury.
- plot_points: Wyświetla chmurę punktów z odróżnieniem punktów należących i nienależących do dopasowanej płaszczyzny.
- load_xyz: Wczytuje chmurę punktów z pliku .xyz.
- select_file: Otwiera okno dialogowe, umożliwiające użytkownikowi wybór pliku.

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import filedialog
from sklearn.cluster import KMeans

'''def find_clusters_with_kmeans(points, k=3, n_init=10):
    """
    Dzieli chmurę punktów na rozłączne chmury za pomocą algorytmu k-średnich.

    Args:
        points (np.array): Chmura punktów 3D.
        n_clusters (int): Liczba chmur do wyznaczenia.
        n_init (int): Liczba inicjalizacji algorytmu.

    Returns:
        np.array: Etykiety dla każdego punktu wskazujące przynależność do chmur.
    """
    # Ustawienie n_init jawnie na 10, aby uniknąć ostrzeżenia
    kmeans = KMeans(n_clusters=k, n_init=n_init)
    kmeans.fit(points)
    labels = kmeans.labels_
    return labels'''

def find_clusters_with_kmeans(points, k=2, max_iterations=100, tolerance=1e-4): #custom
    """
    Prosta implementacja algorytmu k-średnich, zwracająca etykiety klastrów dla punktów.
    Ta wersja jest dostosowana do bezpośredniego użycia z innymi funkcjami, które oczekują etykiet klastrów.

    Args:
        points (np.array): Chmura punktów 3D.
        k (int): Liczba klastrów do wyznaczenia.
        max_iterations (int): Maksymalna liczba iteracji algorytmu.
        tolerance (float): Próg tolerancji dla zmiany położenia centroidów, który decyduje o zakończeniu algorytmu.

    Returns:
        np.array: Etykiety dla każdego punktu wskazujące przynależność do klastrów.
    """
    # Inicjalizacja centroidów przez losowe wybranie k punktów z chmury punktów
    centroids = points[np.random.choice(points.shape[0], k, replace=False)]
    labels = np.zeros(points.shape[0])

    for iteration in range(max_iterations):
        # Przypisanie każdego punktu do najbliższego centroidu
        distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
        closest_centroids = np.argmin(distances, axis=0)

        # Aktualizacja centroidów
        new_centroids = np.array([points[closest_centroids == j].mean(axis=0) for j in range(k)])

        # Sprawdzenie, czy centroidy przestały się zmieniać (lub zmiany są poniżej zadanej tolerancji)
        if np.all(np.sqrt(((new_centroids - centroids) ** 2).sum(axis=1)) < tolerance):
            labels = closest_centroids
            break

        centroids = new_centroids
        labels = closest_centroids

    return labels

def plot_clusters(points, labels):
    """
    Wizualizuje chmury punktów z odróżnieniem kolorów dla każdej chmury.

    Args:
        points (np.array): Chmura punktów 3D.
        labels (np.array): Etykiety wskazujące przynależność punktów do chmur.
    """
    fig = plt.figure()  # Utworzenie nowego rysunku.
    ax = fig.add_subplot(111, projection='3d')  # Dodanie osi 3D do rysunku.

    colormap = plt.cm.viridis  # Użycie mapy kolorów viridis.
    colors = [colormap(i) for i in np.linspace(0, 1, len(np.unique(labels)))]  # Generowanie kolorów dla każdej chmury.

    for i in np.unique(labels):  # Iteracja po unikalnych etykietach chmur.
        ax.scatter(points[labels == i, 0], points[labels == i, 1], points[labels == i, 2], color=colors[i],
                   label=f'Cluster {i + 1}')
        # Rysowanie punktów dla każdej chmury z odpowiednim kolorem.

    ax.set_xlabel('X Label')  # Ustawienie etykiety osi X.
    ax.set_ylabel('Y Label')  # Ustawienie etykiety osi Y.
    ax.set_zlabel('Z Label')  # Ustawienie etykiety osi Z.
    plt.legend()  # Dodanie legendy.
    plt.show()  # Wyświetlenie wykresu.

def plot_points(points, inliers, inlier_color='g', outlier_color='r', point_size=1, inlier_label='Inliers', outlier_label='Outliers'):
    """
    Wyświetla chmurę punktów z odróżnieniem punktów należących i nienależących do dopasowanej płaszczyzny.

    Args:
        points (np.array): Chmura punktów 3D.
        inliers (np.array): Punkty należące do płaszczyzny.
        inlier_color (str): Kolor punktów należących do płaszczyzny.
        outlier_color (str): Kolor punktów nienależących do płaszczyzny.
        point_size (float): Rozmiar punktów na wykresie.
        inlier_label (str): Etykieta dla punktów należących do płaszczyzny.
        outlier_label (str): Etykieta dla punktów nienależących do płaszczyzny.
    """
    fig = plt.figure() # Utworzenie nowego rysunku.
    ax = fig.add_subplot(111, projection='3d') # Dodanie osi 3D do rysunku.

    # Punkty nie należące do płaszczyzny
    outliers = np.array([point for point in points if point not in inliers]) # Wyodrębnienie punktów nienależących do najlepszej płaszczyzny.

    if len(outliers) > 0:
        ax.scatter(outliers[:,0], outliers[:,1], outliers[:,2], color=outlier_color, s=point_size, label=outlier_label) # Rysowanie punktów nienależących na czerwono.
    if len(inliers) > 0:
        ax.scatter(inliers[:,0], inliers[:,1], inliers[:,2], color=inlier_color, s=point_size, label=inlier_label) # Rysowanie punktów należących na zielono.

    ax.set_xlabel('X Label') # Ustawienie etykiety osi X.
    ax.set_ylabel('Y Label') # Ustawienie etykiety osi Y.
    ax.set_zlabel('Z Label') # Ustawienie etykiety osi Z.
    plt.legend() # Dodanie legendy do wykresu.
    plt.show() # Wyświetlenie wykresu.
def fit_plane_ransac(points, iterations=100, distance_threshold=0.01):
    """
    Dopasowuje płaszczyznę do chmury punktów za pomocą algorytmu RANSAC.

    Args:
        points (np.array): Chmura punktów 3D.
        iterations (int): Liczba iteracji algorytmu.
        distance_threshold (float): Próg odległości dla punktów należących do płaszczyzny.

    Returns:
        tuple: Współczynniki najlepiej dopasowanej płaszczyzny i punkty należące do tej płaszczyzny.
    """
    best_plane = None   # Inicjalizacja zmiennej na najlepszą płaszczyznę.
    best_inliers = []   # Lista punktów najlepiej dopasowanych do płaszczyzny.

    for _ in range(iterations): # Wykonanie określonej liczby iteracji algorytmu.
        sample_points = points[np.random.choice(points.shape[0], 3, replace=False)]     # Losowe wybranie 3 punktów z chmury.
        v1 = sample_points[1] - sample_points[0]    # Obliczenie wektorów na płaszczyźnie.
        v2 = sample_points[2] - sample_points[0]
        normal_vector = np.cross(v1, v2)    # Obliczenie wektora normalnego do płaszczyzny.
        A, B, C = normal_vector     # Rozpakowanie współczynników wektora normalnego.
        D = -np.dot(normal_vector, sample_points[0])    # Obliczenie wartości D w równaniu płaszczyzny.

        distances = np.abs(A*points[:,0] + B*points[:,1] + C*points[:,2] + D) / np.linalg.norm(normal_vector)   # Obliczenie odległości punktów od płaszczyzny.
        inliers = points[distances < distance_threshold]    # Wybór punktów będących w określonej odległości od płaszczyzny.
        mean_distance = np.mean(distances)  # Średnia odległość punktów od płaszczyzny

        if len(inliers) > len(best_inliers):    # Aktualizacja najlepszej płaszczyzny, jeśli znaleziono lepszy zestaw punktów.
            best_inliers = inliers
            best_plane = (A, B, C, D)

    if mean_distance < distance_threshold:
        print("Chmura jest płaszczyzną.")
        # Określenie, czy płaszczyzna jest pionowa czy pozioma
        if np.abs(C) > np.abs(A) and np.abs(C) > np.abs(B):
            print("Płaszczyzna jest pozioma.")
        else:
            print("Płaszczyzna jest pionowa.")
    else:
        print("Chmura nie jest płaszczyzną.")

    return best_plane, best_inliers     # Zwrócenie najlepszej płaszczyzny i punktów do niej należących.

def load_xyz(filename):
    """
    Wczytuje chmurę punktów z pliku .xyz.

    Args:
        filename (str): Ścieżka do pliku .xyz.

    Returns:
        np.array: Wczytana chmura punktów 3D.
    """
    with open(filename, 'r') as file:  # Otwarcie pliku do odczytu.
        points = []  # Inicjalizacja listy na punkty.
        for line in file:  # Iteracja po każdej linii w pliku.
            parts = line.strip().split(',')  # Usunięcie białych znaków i podział linii na części.
            if len(parts) == 3:  # Sprawdzenie, czy linia zawiera 3 wartości.
                points.append(
                    [float(part) for part in parts])  # Konwersja stringów na floaty i dodanie do listy punktów.
        return np.array(points)  # Konwersja listy punktów na tablicę numpy i jej zwrócenie.

def select_file():
    """
    Otwiera okno dialogowe do wyboru pliku.

    Returns:
        str: Ścieżka do wybranego pliku.
    """


    root = tk.Tk()  # Inicjalizacja instancji Tkinter.
    root.withdraw()  # Ukrycie głównego okna Tkinter.
    file_path = filedialog.askopenfilename()  # Otwarcie okna dialogowego do wyboru pliku.
    return file_path  # Zwrócenie ścieżki do wybranego pliku.

# Wybór pliku przez użytkownika
filename = select_file()

# Wczytanie danych z pliku, jeśli użytkownik wybrał plik
if filename:
    points = load_xyz(filename)
    plane, inliers = fit_plane_ransac(points)
    print("Współczynniki płaszczyzny:", plane)
    print("Liczba punktów pasujących:", len(inliers))


    # Wyświetlanie chmury punktów z odróżnieniem kolorów
    labels = find_clusters_with_kmeans(points)
    plot_clusters(points, labels)
    plot_points(points, inliers)
else:
    print("Nie wybrano pliku.")

#### Punkt 6 ####

