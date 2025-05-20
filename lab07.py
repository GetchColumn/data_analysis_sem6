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
    return

    ## 2. Загрузка весов нейронной сети
    print('\nЗагрузка весов нейронной сети')
    with open('ex3weights.npy', 'rb') as f:
        Theta1 = np.load(f)  # чтение матрицы Theta1
        Theta2 = np.load(f)  # чтение матрицы Theta2
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 3. Алгоритм прямого распространения
    p = predict(X, Theta1, Theta2)
    e = np.mean((p == y)) * 100
    print('\nТочность на обучающем наборе:', e)
    print('Ожидаемая точность: > 95')
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

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
    # ...  
    # ---------------------------------
    return g


# Функция предсказания значений по всем классам
def predict(X, Theta1, Theta2):
    m = X.shape[0]
    num_labels = Theta2.shape[0]
    p = np.zeros(m, dtype=int)
    # ------ добавьте свой код --------
    # ...
    # ---------------------------------
    return p


# Отображение обучающего набора
def display_data(X, width, height, el_width, el_height):
    plt.figure()
    # ------ добавьте свой код --------
    # ...
    # ---------------------------------
    plt.show()


if __name__ == '__main__':
    plt.ion()
    main()
    input('Перейдите в терминал и нажмите Enter для завершения')
    plt.clf()
