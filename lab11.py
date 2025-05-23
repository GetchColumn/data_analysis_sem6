import numpy as np
import matplotlib.pyplot as plt  
from PIL import Image

# Если у вас нет библиотеки PIL, выполните команду:
#   pip install pillow


def main():
    with open('ex7data2.npy', 'rb') as f:
        X = np.load(f)
    K = 3

    ## 1. Присвоение кластеров
    initial_mu = np.array([[3, 3], [6, 2], [8, 5]])
    C = assign_clusters(X, initial_mu)
    print('\nБлижайшие центроиды к первым пяти примерам:', C[:5])
    print('Ожидаемое значение: [0 2 1 0 0]')
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 2. Вычисление центроидов кластеров
    mu = compute_mean_centroids(X, C, K)
    print('\nЦентроиды:\n', mu)
    print('Ожидаемое значение (приблизительно):')
    print(' [[ 2.428301 3.157924 ]')
    print(' [ 5.813503 2.633656 ]')
    print(' [ 7.119387 3.616684 ]]')
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 3. Начальная инициализация центроидов
    initial_mu = init_centroids(X, K)
    print('\nНачальная инициализация центродов:\n', initial_mu)
    Xf = np.round(X, 2)
    muf = np.round(initial_mu, 2)
    print('Инициализация верная') if np.sum((Xf[:, None] == muf).all(-1).any(1)) == K else print('Ошибка инициализации')
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 4. Алгоритм K-средних
    C, mu = kmeans(X, initial_mu)
    print('\nНайденные алгоритмом K-средних центроиды:\n', mu)
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 5. Отображение результата кластеризации
    draw_clusters(X, C, mu)
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 6. Функция ошибки K-средних
    test_C = np.array([0, 0, 1, 1, 2])
    test_mu = np.array([[2.0, 5.0], [3.0, 1.0], [6.0, 3.0]])
    J = distortion_function(X[:5], test_C, test_mu)
    print('\nВычисленное значение функции ошибки от тестовых данных:', J)
    print('Ожидаемое значение (приблизительно): 10.915')
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 7. Оптимальная кластеризация
    C, mu = iterative_kmeans(X, K, 10)
    print('\nНайденные итеративным алгоритмом K-средних центроиды:\n', mu)
    print('Ожидаемое значение (приблизительно, порядок может быть изменен):')
    print('[[1.954  5.026]')
    print(' [6.034  3.001]')
    print(' [3.044  1.015]]')
    draw_clusters(X, C, mu)
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 8. Использование K-средних для сжатия изображений
    source_image = np.array(Image.open('bird_small.png'))   # Загрузить изображение и представить массивом
    K = 16  # Уменьшим число цветов изображения до 16 (если программа работает долго, возьмите K=2 или K=3)
    Q = 10  # Количество итераций для поиска лучшей кластеризации (если программа работает долго, возьмите Q=1 или Q=2)
    X = source_image.reshape(-1, 3)  # Преобразуем изображение в одномерный массив точек для использовать в алгоритме K-средних
    C, mu = iterative_kmeans(X, K, Q)  # Выполним кластеризацию изображения
    compressed_X = generate_centroids_array(C, mu)  # Сконструируем массив из центроидов кластеров вместо точек
    compressed_image = compressed_X.reshape(source_image.shape).astype(np.uint8)  # Преобразуем массив в двумерное изображение
    draw_images([source_image, compressed_image], ['Исходное', 'Сжатое'])  # Отобразим оба изображения
    Image.fromarray(compressed_image).save('bird_compressed.png')  # Сохраним сжатое изображение

    ## Конец работы


# Присвоение кластеров
def assign_clusters(X, mu):
    [m, n] = X.shape
    K = mu.shape[0]
    C = np.zeros(m, dtype=int)
    # ------ добавьте свой код --------
    # ...
    # ---------------------------------
    return C


# Вычисление центроидов кластеров
def compute_mean_centroids(X, C, K):
    [m, n] = X.shape
    mu = np.zeros((K, n))
    # ------ добавьте свой код --------
    # ...
    # ---------------------------------
    return mu


# Начальная инициализация mu
def init_centroids(X, K):
    [m, n] = X.shape
    init_mu = np.zeros((K, n))
    # ------ добавьте свой код --------
    # ...
    # ---------------------------------
    return init_mu


# Алгоритм K-средних
def kmeans(X, initial_mu):
    [m, n] = X.shape
    K = initial_mu.shape[0]
    mu = initial_mu
    C = np.zeros(m, dtype=int)
    # ------ добавьте свой код --------
    # ...
    # ---------------------------------
    return C, mu


# Отображение кластеризации
def draw_clusters(X, C, mu):
    K = mu.shape[0]
    plt.figure()
    # ------ добавьте свой код --------
    # ...
    # ---------------------------------
    plt.show()


# Функция ошибки K-средних
def distortion_function(X, C, mu):
    J = 0
    # ------ добавьте свой код --------
    # ...
    # ---------------------------------
    return J


# Итеративный алгоритм K-средних
def iterative_kmeans(X, K, Q):
    [m, n] = X.shape
    optimum_C = np.zeros(m, dtype=int)
    optimum_mu = np.zeros((K, n))
    # ------ добавьте свой код --------
    # ...
    # ---------------------------------
    return optimum_C, optimum_mu


# Генерация массива, состоящего из центроидом кластеров каждой точки выборки
def generate_centroids_array(C, mu):
    K = mu.shape[0]
    X = np.zeros((C.shape[0], mu.shape[1]))
    for k in range(K):
        X[C == k, :] = mu[k, :]
    return X


# Показ изображений
def draw_images(images, labels):
    N = len(images)
    fig, axs = plt.subplots(1, 2)
    for i in range(N):
        axs[i].imshow(images[i])
        axs[i].set_title(labels[i])
    fig.show()


if __name__ == '__main__':
    plt.ion()
    main()
    input('Перейдите в терминал и нажмите Enter для завершения')
    plt.clf()
