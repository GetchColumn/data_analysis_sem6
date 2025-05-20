import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt  


def main():
    ## 1. Загрузка и отображение данных
    with open('ex5data1.npy', 'rb') as f:
        X = np.load(f)  # чтение обучающей выборки 
        y = np.load(f)  
        Xval = np.load(f)  # чтение валидационной выборки 
        yval = np.load(f)  
        Xtest = np.load(f)  # чтение тестовой выборки 
        ytest = np.load(f)  
    m = X.shape[0]
    print('Отображение обучающего набора данных')
    display_data(X, y)
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 2. Функция стоимости регуляризованной линейной регрессии
    X1 = np.hstack((np.ones((m, 1)), X))
    theta = np.array([1., 1.])
    lamb = 1.
    J = cost_function(theta, X1, y, lamb)
    print('\nФункция стоимости:', J)
    print('Ожидаемое значение (приблизительно): 303.993192')
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 3. Функция градиента регуляризованной линейной регрессии
    grad = gradient_function(theta, X1, y, lamb)
    print('\nГрадиент:', grad)
    print('Ожидаемое значение (приблизительно): [-15.303016; 598.250744]')
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 4. Обучение регуляризованной линейной регрессии 
    print('\nОбучение модели')
    lamb = 0
    theta = train_model(X1, y, lamb)
    print('\nTэта:', theta)
    print('Ожидаемое значение (приблизительно): [13.0879; 0.3678]')
    print('Отображение линейной регрессии')
    display_data(X, y, theta)
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 5. Построение и отображение кривых обучения
    print('\nПостроение кривых обучения')
    Xval1 = np.hstack((np.ones((Xval.shape[0], 1)), Xval))
    Jtrain, Jval = learning_curves(X1, y, Xval1, yval, lamb)
    # Отображение кривых
    mt = np.linspace(1, m , m)
    plt.figure()
    plt.plot(mt, Jtrain, label='Ошибка на обучении')
    plt.plot(mt, Jval, label='Ошибка на валидации')
    plt.title('Кривые обучения')
    plt.xlabel('Размер выборки')
    plt.ylabel('Ошибка')
    plt.legend()
    plt.show()
    print('Размер выборки\tОшибка на обучении\tОшибка на валидации')
    for i in range(m):
        print(i+1, '\t', Jtrain[i], '\t', Jval[i])
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 6. Формирование полиномиальной модели
    print('\nФормирование полиномиальной модели')
    p = 8  # используем полином 8-й степени
    Xp = poly_features(X, p)
    Xpoly_n, mu, sigma = feature_normalize(Xp)
    Xmin, Xmax = np.min(Xpoly_n), np.max(Xpoly_n)
    Xmean, Xstd = np.mean(Xpoly_n), np.std(Xpoly_n)
    print('Получена матрица размером', Xpoly_n.shape, 'Среднее значение:', Xmean, 'Отклонение:', Xstd)
    print('Ожидаемый размер', (m, p), 'Ожидаемые значения среднего и отклонения (приблизительно): 0.0  1.0')
    Xpoly_n = np.hstack((np.ones((Xpoly_n.shape[0], 1)), Xpoly_n))
    Xpoly_val_n = poly_features(Xval, p)
    Xpoly_val_n = (Xpoly_val_n - mu) / sigma
    Xpoly_val_n = np.hstack((np.ones((Xpoly_val_n.shape[0], 1)), Xpoly_val_n))
    Xpoly_test_n = poly_features(Xtest, p)
    Xpoly_test_n = (Xpoly_test_n - mu) / sigma
    Xpoly_test_n = np.hstack((np.ones((Xpoly_test_n.shape[0], 1)), Xpoly_test_n))
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 7. Кривые обучения полиномиальной модели
    print('\nОбучение полиномиальной модели')
    lamb = 0
    theta = train_model(Xpoly_n, y, lamb)
    # Отображение полиномиальной модели на данных
    plt.figure()
    plt.title('Полиномиальная модель')
    plt.xlabel("Изменение уровня воды")
    plt.ylabel("Поток воды через дамбу")
    plt.scatter(X, y, c='b', label='Выборка')
    Xd = np.linspace(np.min(X)-10, np.max(X)+10).reshape(-1, 1)
    Xdp = (poly_features(Xd, p) - mu) / sigma
    Xdp = np.hstack((np.ones((Xdp.shape[0], 1)), Xdp))
    yd = np.dot(Xdp, theta)
    plt.plot(Xd, yd, c='r', label='Модель')
    plt.legend()
    plt.show()
    # Кривые обучения полиномиальной модели
    Jtrain, Jval = learning_curves(Xpoly_n, y, Xpoly_val_n, yval, lamb)
    plt.figure()
    plt.plot(mt, Jtrain, label='Ошибка на обучении')
    plt.plot(mt, Jval, label='Ошибка на валидации')
    plt.title('Кривые обучения полиномиальной модели')
    plt.xlabel('Размер выборки')
    plt.ylabel('Ошибка')
    plt.legend()
    plt.show()
    print('Размер выборки\tОшибка на обучении\tОшибка на валидации')
    for i in range(m):
        print(i+1, '\t', Jtrain[i], '\t', Jval[i])
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 8. Валидация для подбора параметра регуляризации
    lambda_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    Jtrain, Jval = validation_curves(Xpoly_n, y, Xpoly_val_n, yval, lambda_vec)
    plt.figure()
    plt.plot(lambda_vec, Jtrain, label='Ошибка на обучении')
    plt.plot(lambda_vec, Jval, label='Ошибка на валидации')
    plt.title('Валидация параметра регуляризации')
    plt.xlabel('Лямбда')
    plt.ylabel('Ошибка')
    plt.legend()
    plt.show()
    print('Лямбда\tОшибка на обучении\tОшибка на валидации')
    for i in range(len(lambda_vec)):
        print(lambda_vec[i], '\t', Jtrain[i], '\t', Jval[i])
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 9. Оценка обобщающей способности модели
    print('\nОценка обобщающей способности')
    lamb = 0  
    # ------ добавьте свой код --------
    # Задайте оптимальное значение лямбда по результатам шага 8
    # ...
    # ---------------------------------
    theta = train_model(Xpoly_n, y, lamb)
    Jtest = cost_function(theta, Xpoly_test_n, ytest, lamb=0.)
    print('Ошибка модели на тесте:', Jtest)
    print('Ожидаемое значени (приблизительно): 3.6')

    ## Конец работы


# Отображение данных
def display_data(X, y, theta=None):
    plt.figure()
    # ------ добавьте свой код --------

    # Для шага 1: добавьте отображение обучающей выборки
    # ...

    # Для шага 4: добавьте отображение предсказанных значений для X
    if theta is not None:
        # ...
        pass  # уберите эту строку

    # ---------------------------------
    plt.show()


# Функция стоимости регуляризованной линейной регрессии
def cost_function(theta, X, y, lamb):
    J = 0
    # ------ добавьте свой код --------
    # ...
    # ---------------------------------
    return J


# Функция градиента регуляризованной линейной регрессии
def gradient_function(theta, X, y, lamb):
    dth = np.zeros(theta.shape)
    # ------ добавьте свой код --------
    # ...
    # ---------------------------------
    return dth.flatten()


# Построение кривых обучения
def learning_curves(X, y, Xval, yval, lamb):
    m = y.shape[0]
    Jtrain = np.zeros(m)
    Jval = np.zeros(m)
    # ------ добавьте свой код --------
    # ...
    # ---------------------------------
    return Jtrain, Jval


# Построение кривых валидации регуляризации
def validation_curves(X, y, Xval, yval, lambda_vec):
    n = len(lambda_vec)
    Jtrain = np.zeros(n)
    Jval = np.zeros(n)
    # ------ добавьте свой код --------
    # ...
    # ---------------------------------
    return Jtrain, Jval


# Формирование полиномиальных признаков
def poly_features(X, p):
    m = X.shape[0]
    Xpoly = np.zeros((m, p))
    # ------ добавьте свой код --------
    # ...
    # ---------------------------------
    return Xpoly


# Нормализация выборки
def feature_normalize(X):
    Xnorm = np.zeros(X.shape)
    mu = 0
    sigma = 0
    # ------ добавьте свой код --------
    # ...
    # ---------------------------------
    return Xnorm, mu, sigma


# Функция обучения модели (вам не нужно модифицировать эту функцию)
def train_model(X, y, lamb):
    initial_theta = np.zeros(X.shape[1])
    min_res = op.minimize(fun=cost_function, jac=gradient_function, x0=initial_theta, args=(X, y, lamb))
    if not min_res.success:
        print('Ошибка оптимизации:', min_res.message)
    return min_res.x


if __name__ == '__main__':
    plt.ion()
    main()
    input('Перейдите в терминал и нажмите Enter для завершения')
    plt.clf()
