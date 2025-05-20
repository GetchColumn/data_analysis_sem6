import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt  


def main():
    input_layer_size = 400  # 20x20 количество точек изображения одной цифры
    hidden_layer_size = 25  # Количество нейронов в скрытом слое
    num_labels = 10         # 10 классов, от 0 до 9

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
    return

    ## 2. Загрузка весов нейронной сети
    print('\nЗагрузка весов нейронной сети')
    with open('ex3weights.npy', 'rb') as f:
        Theta1 = np.load(f)  # чтение матрицы Theta1
        Theta2 = np.load(f)  # чтение матрицы Theta2
    weights = np.append(Theta1.flatten(), Theta2.flatten())  # превратим две матрицы в один вектор весов
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 3. Функция стоимости (алгоритм прямого распространения)
    print('\nАлгоритм прямого распространения')
    lamb = 0  # Параметр регуляризации устанавливаем в 0, она пока не используется
    J, _ = cost_function(weights, input_layer_size, hidden_layer_size, num_labels, X, y, lamb)
    print('Значение функции стоимости без регуляриизации при загруженных из файла параметрах:' , J)
    print('Ожидаемое значение (приблизительно): 0.287629')
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 4. Функция стоимости с регуляризацией
    lamb = 1  # Задаем параметр регуляризации
    J, _ = cost_function(weights, input_layer_size, hidden_layer_size, num_labels, X, y, lamb)
    print('Значение функции стоимости без регуляриизации при загруженных из файла параметрах:' , J)
    print('Ожидаемое значение (приблизительно): 0.383770')
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 5. Производная функции сигмоиды
    print('\nВычисление производной сигмоиды')
    g = sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1]))
    print('Производная сигмоиды для [-1 -0.5 0 0.5 1]:', g)
    print('Ожидемое значение: [0.197 0.235 0.250 0.235 0.197]')
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 6. Начальная инициализация весов
    print('\nИнициализация параметров нейронной сети')
    initial_Theta1 = initialize_weights(input_layer_size, hidden_layer_size)
    initial_Theta2 = initialize_weights(hidden_layer_size, num_labels)
    initial_weights = np.append(initial_Theta1.flatten(), initial_Theta2.flatten())  # превратим две матрицы в один вектор весов
    print('Вектор параметров длиной:', initial_weights.shape[0], ' M:', np.mean(initial_weights), ' Std:', np.std(initial_weights))
    print('Ожидаемая длина: 10285  M(приблизительно): 0.0  Std(приблизительно): 0.07')
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 7. Алгоритм обратного распространения (без регуляризации)
    print('\nПроверка алгоритма обратного распространения (без регуляризации)')
    lamb = 0
    # Создадим тестовые значения меньшей размерности 
    t_inp_size = 3
    t_hidden_size = 5
    t_out_size = 3
    t_num_labels = 3
    t_m = 5
    t_X = np.random.rand(t_m, t_inp_size) * 2 * 0.12 - 0.12
    t_y = np.random.randint(0, t_num_labels, t_m)
    t_weights = np.random.rand((t_inp_size + 1) * t_hidden_size + (t_hidden_size + 1) * t_out_size) * 2 * 0.12 - 0.12
    # Вычислим градиент алгоритмом обратного распространения и численным методом и сравним
    _, weights_grad = cost_function(t_weights, t_inp_size, t_hidden_size, t_num_labels, t_X, t_y, lamb)
    weights_grad_num = numerical_gradient(t_weights, t_inp_size, t_hidden_size, t_num_labels, t_X, t_y, lamb)
    diff = np.linalg.norm(weights_grad_num - weights_grad) / np.linalg.norm(weights_grad_num + weights_grad)
    print('Относительное расхождение:', diff)
    print('Ожидаемое значение: < 1e-9')
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 8. Регуляризация параметров нейронной сети
    print('\nПроверка алгоритма обратного распространения (с регуляризацией)')
    lamb = 3
    # Вычислим градиент алгоритмом обратного распространения и численным методом и сравним
    _, weights_grad = cost_function(t_weights, t_inp_size, t_hidden_size, t_num_labels, t_X, t_y, lamb)
    weights_grad_num = numerical_gradient(t_weights, t_inp_size, t_hidden_size, t_num_labels, t_X, t_y, lamb)
    diff = np.linalg.norm(weights_grad_num - weights_grad) / np.linalg.norm(weights_grad_num + weights_grad)
    print('Относительное расхождение:', diff)
    print('Ожидаемое значение: < 1e-9')
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 9. Обучение нейронной сети
    print('\nОбучение нейронной сети')
    lamb = 1
    weights, _, _ = op.fmin_tnc(func=cost_function, x0=initial_weights, args=(input_layer_size, hidden_layer_size, num_labels, X, y, lamb), maxfun=150)

    ## 10. Проверка качества обучения
    print('\nПроверка качества обучения')
    Theta1 = weights[: (hidden_layer_size * (input_layer_size + 1))].reshape(hidden_layer_size, input_layer_size + 1)
    Theta2 = weights[(hidden_layer_size * (input_layer_size + 1)) :].reshape(num_labels, hidden_layer_size + 1)
    p = predict(X, Theta1, Theta2)
    e = np.mean((p == y)) * 100
    print('Точность на обучающем наборе:', e)
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 11. Предсказание нейронной сетью случайных цифр
    indices = np.random.permutation(X.shape[0])
    for ind in indices:
        X_t = X[ind, :].reshape(1, -1)
        pred = predict(X_t, Theta1, Theta2)
        print('Предсказание сетью:', pred)
        display_data(X_t, 1, 1, 20, 20)
        c = input('Нажмите Enter для продолжения и q для выхода')
        if c == 'q' or c == 'Q':
            break


# Логистическая функция
def sigmoid(z):
    g = np.zeros(z.shape)
    # ------ добавьте свой код --------
    # ...  
    # ---------------------------------
    return g


# Производная логистической функции
def sigmoid_gradient(z):
    g = np.zeros(z.shape)
    # ------ добавьте свой код --------
    # ...  
    # ---------------------------------
    return g


# Функция вычисления стоимости и градиента для нейронной сети
def cost_function(weights, input_size, hidden_size, num_labels, X, y, lamb):
    J = 0

    # Преобразование входного вектора весов weights в две матрицы Theta1 и Theta2
    Theta1 = weights[: (hidden_size * (input_size + 1))].reshape(hidden_size, input_size + 1)
    Theta2 = weights[(hidden_size * (input_size + 1)) :].reshape(num_labels, hidden_size + 1)

    dth1 = np.zeros(Theta1.shape)
    dth2 = np.zeros(Theta2.shape)
    # ------ добавьте свой код --------

    # Для шага 3: реализуйте алгоритм прямого распространения для вычисления значения функции стоимости J (без регуляризации)
    # ...

    # Для шага 4: реализуйте регуляризацию для функции стоимости 
    # ...

    #  Для шага 7: реализуйте алгоритм обратного распространения для вычисления значений частных производных переметров нейронной сети
    # ...

    # Для шага 8: добавьте регуляризацию к обучению переметров нейронной сети
    # ...

    # ---------------------------------
    dth = np.append(dth1.flatten(), dth2.flatten())
    return J, dth


# Начальная инициализация весов случайными значениями
def initialize_weights(in_layer_size, out_layer_size):
    W = np.zeros((out_layer_size, in_layer_size + 1))
    # ------ добавьте свой код --------
    # ...
    # ---------------------------------
    return W


# Численное определение градиента
def numerical_gradient(weights, input_size, hidden_size, num_labels, X, y, lamb):
    dth = np.zeros(weights.shape)
    e = 1e-4
    # ------ добавьте свой код --------
    # ...
    # ---------------------------------
    return dth



# Функция предсказания значений по нейронной сети
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
