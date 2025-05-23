import numpy as np
import matplotlib.pyplot as plt  


def main():
    ## 1. Загрузка, нормализация и отображение данных
    with open('ex7data1.npy', 'rb') as f:
        X = np.load(f)
    Xn, mu, sigma = feature_normalize(X)
    draw_data(Xn)
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 2. Метод главных компонент
    Z, R = pca(Xn, 1)
    print('\nПолученная матрица понижения размерности:\n', R)
    print('Ожидаемое значение (приближенно):\n [[-0.7071],\n [-0.7071]]')
    print('Первые три значения Z:\n', Z[:3])
    print('Ожидаемые значения (приближенно):\n [[ 1.4963]\n [-0.9222]\n [ 1.2244]]')
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 3. Восстановление данных
    Xr = reconstruct(Z, R)
    print('Первые три значения восстановленных данных:\n', Xr[:3])
    print('Ожидаемые значения (приближенно):\n [[-1.058 -1.058]\n [ 0.652  0.652]\n [-0.866 -0.866]]')
    draw_data(Xn, Xr)
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 4. Применение PCA для набора данных лиц
    with open('ex7faces.npy', 'rb') as f:
        X = np.load(f)
    m = X.shape[0]
    subset = np.random.permutation(m)[:25]
    display_faces(X[subset], 5, 5, 32, 32, 'Исходные изображения')
    Xn, mu, sigma = feature_normalize(X)
    Z, R = pca(Xn, 100)
    Xr = reconstruct(Z, R)
    display_faces(Xr[subset], 5, 5, 32, 32, 'Восстановленные по ' + str(R.shape[1]) + ' компонентам')
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 5. Подбор значения k
    print('\nПодбор значения k для сохранения 99% дисперсии')
    Z, R = pca_adaptive(Xn, 0.99)
    Xr = reconstruct(Z, R)
    print('Полученное значение k =', R.shape[1])
    display_faces(Xr[subset], 5, 5, 32, 32, 'Восстановленные по ' + str(R.shape[1]) + ' компонентам')
    print('Подбор значения k для сохранения 70% дисперсии')
    Z, R = pca_adaptive(Xn, 0.70)
    Xr = reconstruct(Z, R)
    print('Полученное значение k =', R.shape[1])
    display_faces(Xr[subset], 5, 5, 32, 32, 'Восстановленные по ' + str(R.shape[1]) + ' компонентам')

    ## Конец работы


# Отображение выборки
def draw_data(X, Xr=None):
    plt.figure()
    # ------ добавьте свой код --------

    # Для шага 1
    # ...

    # Для шага 3
    if Xr is not None:
        pass
        # ...

    plt.legend()
    # ---------------------------------
    plt.show()


# Нормализация данных
def feature_normalize(X):
    X_norm = X
    mu = 0
    sigma = 0
    # ------ добавьте свой код --------
    # ...
    # ---------------------------------
    return X_norm, mu, sigma


# Метод главных компонент
def pca(X, k):
    m, n = X.shape
    Z = np.zeros((m, k))
    R = np.zeros((n, n))
    # ------ добавьте свой код --------
    # ...
    # ---------------------------------
    return Z, R


# Восстановление размерности
def reconstruct(Z, R):
    m, k = Z.shape
    n = R.shape[0]
    Xr = np.zeros((m, n))
    # ------ добавьте свой код --------
    # ...
    # ---------------------------------
    return Xr


# Метод главных компонент
def pca_adaptive(X, threshold):
    m, n = X.shape
    Z = np.zeros(X.shape)
    R = np.zeros((n, n))
    # ------ добавьте свой код --------
    # ...
    # ---------------------------------
    return Z, R


# Отображение лиц
def display_faces(X, height, width, el_height, el_width, title=None):
    plt.figure()
    plt.gray()
    for i in range(height):
        for j in range(width):
            el = X[i * height + j].reshape(el_height, el_width).T
            row = np.hstack((row, el)) if j > 0 else el
        fig = np.vstack((fig, row)) if i > 0 else row
    plt.imshow(fig)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    plt.ion()
    main()
    input('Перейдите в терминал и нажмите Enter для завершения')
    plt.clf()
