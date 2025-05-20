import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt  


def main():
    num_labels = 10  # 10 классов, от 0 до 9

    ## 1. Загрузка данных
    with open('ex3data1.npy', 'rb') as f:
        X = np.load(f)  # чтение матрицы X из бинарного файла формата numpy 
        y = np.load(f)  # чтение вектора y из бинарного файла формата numpy
    print('Отображение обучающего набора данных')
    # Выберем из обучающего набора 100 случайных изображений и покажем их
    indices = np.random.permutation(X.shape[0])[:100]  
    selected = X[indices, :]
    display_data(selected, 10, 10, 20, 20)
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    # return

    ## 2. Регуляризованная логистическая регрессия
    theta_t = np.array([-2, -1, 1, 2])
    X_t = np.hstack((np.ones(5).reshape(5, 1), np.arange(1, 16).reshape(3, 5).T / 10))
    y_t = (np.array([1, 0, 1, 0, 1]) >= 0.5)
    lamb_t = 3
    cost_t = cost_function(theta_t, X_t, y_t, lamb_t)
    grad_t = gradient_function(theta_t, X_t, y_t, lamb_t)
    print('\nФункция стоимости:', cost_t)
    print('Ожидаемое значение (приблизительно): 2.534819')
    print('Градиент:', grad_t)
    print('Ожидаемый градиент (приблизительно): [ 0.146561  -0.548558  0.724722  1.398003 ]')
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    # return

    ## 3. Обучение "один-против-всех"
    print('\nОбучение логистической регрессии "один-против-всех"')
    lamb = 0.1
    all_theta = one_vs_all(X, y, lamb, num_labels)
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    # return

    ## 4. Предсказание "один-против-всех"
    p = predict_one_vs_all(X, all_theta)
    e = np.mean((p == y)) * 100
    print('\nТочность на обучающем наборе, в %:', e)
    print('Ожидаемая точность: 95% и выше')


# Логистическая фукнция
def sigmoid(z):
    g = np.zeros(z.shape)
    # ------ добавьте свой код --------
    g = 1 / (1 + np.exp(-z))
    # ---------------------------------
    return g


# Функция стоимости регуляризованной логистической регрессии
def cost_function(theta, X, y, lamb):
    J = 0
    m = X.shape[0]
    h = sigmoid(np.dot(X, theta))

    # Вычисление стоимости с учетом регуляризации
    J = -1 / m * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h)) + (lamb / (2 * m)) * np.sum(theta[1:] ** 2)
    return J

# Функция вычисления градиента регуляризованной логистической регрессии
def gradient_function(theta, X, y, lamb):
    dth = np.zeros(theta.shape)
    m = X.shape[0]
    h = sigmoid(np.dot(X, theta))
    error = h - y

    # Градиент без регуляризации для intercept (нулевого параметра)
    dth[0] = 1 / m * np.sum(error * X[:, 0])

    # Градиент с регуляризацией для остальных параметров
    dth[1:] = (1 / m * np.dot(X[:, 1:].T, error)) + (lamb / m) * theta[1:]
    return dth.flatten()


# Обучение моделей логистической регрессии для всех классов
def one_vs_all(X, y, lamb, num_labels):
    [m, n] = X.shape
    all_theta = np.zeros((num_labels, n + 1))

    # Добавление столбца единиц для интерсепта
    X = np.hstack((np.ones((m, 1)), X))  # Объединяем единичный столбец с X

    # Обучение модели для каждого класса
    for k in range(num_labels):
        # Отображаем класс-целевое значение: 1 для класса k и 0 для остальных
        binary_y = (y == k).astype(int)
        # Инициализация случайных параметров
        initial_theta = np.zeros(n + 1)
        # Минимизация функции стоимости с помощью оптимизации
        result = op.minimize(fun=cost_function, x0=initial_theta, args=(X, binary_y, lamb),
                          method='TNC', jac=gradient_function)
        # Сохраняем параметры модели для класса k
        all_theta[k, :] = result.x
    return all_theta


# Функция предсказания значений по всем классам
def predict_one_vs_all(X, all_theta):
    m = X.shape[0]
    num_labels = all_theta.shape[0]
    # Добавляем столбец единиц для интерсепта
    X = np.hstack((np.ones((m, 1)), X))  # Объединяем единичный столбец с X
    # Вычисляем предсказания с использованием сигмоиды
    predictions = sigmoid(np.dot(X, all_theta.T))
    # Находим индекс класса с максимальным значением вероятности
    p = np.argmax(predictions, axis=1)
    return p


# Отображение обучающего набора
def display_data(X, width, height, el_width, el_height):
    plt.figure()
    for i in range(height):
        for j in range(width):
            el = X[i * height + j].reshape(el_height, el_width).T
            row = np.hstack((row, el)) if j > 0 else el
        fig = np.vstack((fig, row)) if i > 0 else row
    plt.imshow(fig)
    plt.show()
    plt.pause(10)


if __name__ == '__main__':
    plt.ion()
    main()
    input('Перейдите в терминал и нажмите Enter для завершения')
    plt.clf()
