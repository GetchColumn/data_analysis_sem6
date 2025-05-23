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
    # return

    ## 2. Метод главных компонент
    Z, R = pca(Xn, 1)
    print('\nПолученная матрица понижения размерности:\n', R)
    print('Ожидаемое значение (приближенно):\n [[-0.7071],\n [-0.7071]]')
    print('Первые три значения Z:\n', Z[:3])
    print('Ожидаемые значения (приближенно):\n [[ 1.4963]\n [-0.9222]\n [ 1.2244]]')
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    # return

    ## 3. Восстановление данных
    Xr = reconstruct(Z, R)
    print('Первые три значения восстановленных данных:\n', Xr[:3])
    print('Ожидаемые значения (приближенно):\n [[-1.058 -1.058]\n [ 0.652  0.652]\n [-0.866 -0.866]]')
    draw_data(Xn, Xr)
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    # return

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
    # return

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


# # Отображение выборки
# def draw_data(X, Xr=None):
#     plt.figure()
#     # ------ добавьте свой код --------
#
#     # Для шага 1
#     plt.scatter(X[:, 0], X[:, 1], c='blue', label='Данные', alpha=0.7, edgecolors='k')
#
#     # Для шага 3
#     if Xr is not None:
#         pass
#         # ...
#
#     plt.xlabel('Признак 1')
#     plt.ylabel('Признак 2')
#     plt.title('Отображение данных')
#     plt.legend()
#     plt.grid(True)
#     # ---------------------------------
#     plt.show()
# Отображение выборки
def draw_data(X, Xr=None):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c='blue', label='Данные', alpha=0.7, edgecolors='k')
    if Xr is not None:
        # Отображаем восстановленные точки другим цветом
        plt.scatter(Xr[:, 0], Xr[:, 1], c='red', label='Восстановленные', alpha=0.7, marker='x')
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.title('Отображение данных')
    plt.legend()
    plt.grid(True)
    plt.show()

# Нормализация данных
def feature_normalize(X):
    # Вычисляем среднее и стандартное отклонение по каждому признаку (столбцу)
    mu = np.mean(X, axis=0)  # Вектор средних (размерность n)
    sigma = np.std(X, axis=0, ddof=0)  # Вектор стандартных отклонений (размерность n)

    # Нормализация: (X - mu) / sigma
    # Добавляем небольшое значение к sigma, чтобы избежать деления на ноль
    sigma_safe = sigma.copy()
    sigma_safe[sigma_safe == 0] = 1.0

    X_norm = (X - mu) / sigma_safe

    return X_norm, mu, sigma


# Метод главных компонент
def pca(X, k):
    m, n = X.shape
    # Центрируем данные
    mu = np.mean(X, axis=0)
    X_centered = X - mu

    # Ковариационная матрица
    Sigma = (X_centered.T @ X_centered) / m

    # Собственные значения и векторы (или SVD)
    # Используем SVD для численной устойчивости
    U, S, Vt = np.linalg.svd(Sigma)
    # U — столбцы это собственные векторы (размер n x n)

    # Матрица понижения размерности (n x k)
    R = U[:, :k]

    # Проекция данных в новое пространство (m x k)
    Z = X_centered @ R

    return Z, R


# # Восстановление размерности
# def reconstruct(Z, R):
#     m, k = Z.shape
#     n = R.shape[0]
#     Xr = np.zeros((m, n))
#     # ------ добавьте свой код --------
#     # ...
#     # ---------------------------------
#     return Xr

# Восстановление размерности
def reconstruct(Z, R):
    # Z: (m, k), R: (n, k)
    # Восстановление: Xr = Z @ R.T
    Xr = Z @ R.T
    return Xr


# # Метод главных компонент
# def pca_adaptive(X, threshold):
#     m, n = X.shape
#     Z = np.zeros(X.shape)
#     R = np.zeros((n, n))
#     # ------ добавьте свой код --------
#     # ...
#     # ---------------------------------
#     return Z, R
def pca_adaptive(X, threshold):
    m, n = X.shape
    # Центрируем данные
    mu = np.mean(X, axis=0)
    X_centered = X - mu

    # Ковариационная матрица
    Sigma = (X_centered.T @ X_centered) / m

    # SVD
    U, S, Vt = np.linalg.svd(Sigma)
    # S — вектор собственных значений (дисперсий по компонентам)

    # Считаем долю сохранённой дисперсии для каждого k
    total_variance = np.sum(S)
    variance_ratio = np.cumsum(S) / total_variance

    # Находим минимальное k, при котором сохраняется не менее threshold дисперсии
    k = np.searchsorted(variance_ratio, threshold) + 1

    # Матрица понижения размерности (n x k)
    R = U[:, :k]

    # Проекция данных (m x k)
    Z = X_centered @ R

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
    # plt.ion()
    main()
    input('Перейдите в терминал и нажмите Enter для завершения')
    plt.clf()
