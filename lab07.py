import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt  


def main():
    input_layer_size = 400  # 20x20 количество точек изображения одной цифры
    hidden_layer_size = 25  # Количество нейронов в скрытом слое
    num_labels = 10         # 10 классов, от 0 до 9

    ## 1. Загрузка данных
    with open('ex3data1_7.npy', 'rb') as f:
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

    ## 2. Загрузка весов нейронной сети
    print('\nЗагрузка весов нейронной сети')
    with open('ex3weights.npy', 'rb') as f:
        Theta1 = np.load(f)  # чтение матрицы Theta1
        Theta2 = np.load(f)  # чтение матрицы Theta2
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    # return

    ## 3. Алгоритм прямого распространения
    p = predict(X, Theta1, Theta2)
    e = np.mean((p == y)) * 100
    print('\nТочность на обучающем наборе:', e)
    print('Ожидаемая точность: > 95')
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    # return

    ## 4. Предсказание нейронной сетью случайных цифр
    indices = np.random.permutation(X.shape[0])
    for ind in indices:
        X_t = X[ind, :].reshape(1, -1)
        pred = predict(X_t, Theta1, Theta2)
        print('Предсказание сетью:', pred)
        display_data(X_t, 1, 1, 20, 20)
        c = input('Нажмите Enter для продолжения и q для выхода')
        if c == 'q' or c == 'Q':
            break


# Логистическая фукнция
def sigmoid(z):
    g = np.zeros(z.shape)
    # ------ добавьте свой код --------
    g = 1 / (1 + np.exp(-z))
    # ---------------------------------
    return g


# Функция предсказания значений по всем классам
def predict(X, Theta1, Theta2):
    m = X.shape[0]
    num_labels = Theta2.shape[0]
    p = np.zeros(m, dtype=int)

    # Добавляем столбец единиц для интерсепта в первый слой
    X = np.hstack((np.ones((m, 1)), X))  # Добавляем единичный столбец
    # Прямое распространение от входного слоя к скрытому слою
    z2 = np.dot(X, Theta1.T)
    a2 = sigmoid(z2)
    # Добавляем столбец единиц для интерсепта во второй слой
    a2 = np.hstack((np.ones((m, 1)), a2))  # Добавляем единичный столбец
    # Прямое распространение от скрытого слоя к выходному слою
    z3 = np.dot(a2, Theta2.T)
    a3 = sigmoid(z3)
    # Получаем индексы классов, соответствующие максимальным значениям выходного слоя
    p = np.argmax(a3, axis=1)  # Индексы начинаются с 0
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
    # plt.pause(10)


if __name__ == '__main__':
    # plt.ion()
    main()
    input('Перейдите в терминал и нажмите Enter для завершения')
    plt.clf()
