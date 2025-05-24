import numpy as np
import matplotlib.pyplot as plt  


def main():
    ## 1. Загрузка и отображение данных
    print('Загрузка и отображение данных')
    with open('ex8data1.npy', 'rb') as f:
        X = np.load(f)
        Xval = np.load(f)
        yval = np.load(f)
    draw_data(X)
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    # return

    ## 2. Оценивание параметров распределений
    print('\nОценивание параметров нормального распределения')
    mu, sigma2 = estimate_gaussian(X)
    print('Полученный вектор матожидания:', mu)
    print('Ожидаемое значение (приближенно): [14.112 14.998]')
    print('Полученный вектор дисперсии:', sigma2)
    print('Ожидаемое значение (приближенно): [1.833 1.710]')
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    # return

    ## 3. Модель многомерного распределения
    print('\nМодель многомерного нормального распределения')
    pval = multivariate_gaussian(Xval, mu, sigma2)
    print('Первых пять значений:', pval[:5])
    print('Ожидаемые значения (приближенно): [0.042 0.082 0.041 0.062 0.0712]')
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    # return

    ## 4. Выбор порога
    print('\nВыбор порога обнаружения аномалий')
    epsilon, F1 = select_threshold(yval, pval)
    print('Полученное значение порога:', epsilon)
    print('Ожидаемое значение (приближенно): 0.00009')
    print('Полученное значение F1-меры:', F1)
    print('Ожидаемое значение (приближенно): 0.875')
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    # return

    ## 5. Отображение
    print('\nОтображение выборки с распределением и найденными аномалиями')
    draw_data_and_fit(X, mu, sigma2, epsilon)
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    # return

    ## 6. Проверка на наборе данных с большим числом параметров
    print('\nПроверка на наборе данных с большим числом параметров')
    with open('ex8data2.npy', 'rb') as f:
        X = np.load(f)
        Xval = np.load(f)
        yval = np.load(f)
    mu, sigma2 = estimate_gaussian(X)
    pval = multivariate_gaussian(Xval, mu, sigma2)
    epsilon, F1 = select_threshold(yval, pval)
    print('Полученное значение порога:', epsilon)
    print('Ожидаемое значение (приближенно): 1.3786e-18')
    print('Полученное значение F1-меры:', F1)
    print('Ожидаемое значение (приближенно): 0.61538')
    print('Найдено аномалий:', sum(pval < epsilon))

    ## Конец работы


# Отображение выборки
# def draw_data(X):
#     plt.figure()
#     # ------ добавьте свой код --------
#     # ...
#     # ---------------------------------
#     plt.show()
def draw_data(X):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c='blue', edgecolors='k', alpha=0.7, label='Обучающая выборка')
    plt.xlabel('Пропускная способность (Мбит/с)')
    plt.ylabel('Задержка ответа (мсек)')
    plt.title('Обучающая выборка')
    plt.legend()
    plt.grid(True)
    plt.show()


# Оценивание параметров Гауссовского распределения    
# def estimate_gaussian(X):
#     m, n = X.shape
#     mu = np.zeros(n)
#     sigma2 = np.zeros(n)
#     # ------ добавьте свой код --------
#     # ...
#     # ---------------------------------
#     return mu, sigma2
def estimate_gaussian(X):
    # Среднее по каждому признаку (столбцу)
    mu = np.mean(X, axis=0)
    # Несмещённая дисперсия по каждому признаку (столбцу)
    sigma2 = np.var(X, axis=0, ddof=0)
    return mu, sigma2

# Расчет оценки многомерного распределения
# def multivariate_gaussian(X, mu, sigma2):
#     m, n = X.shape
#     p = np.zeros(m, dtype=int)
#     # ------ добавьте свой код --------
#     # ...
#     # ---------------------------------
#     return p
def multivariate_gaussian(X, mu, sigma2):
    m, n = X.shape
    sigma2_diag = np.diag(sigma2)
    X_centered = X - mu
    # Вычисляем определитель и обратную матрицу ковариации
    det = np.prod(sigma2)
    inv = np.diag(1 / sigma2)
    # Формула плотности многомерного нормального распределения
    coef = 1 / (np.power((2 * np.pi), n / 2) * np.sqrt(det))
    exp_term = np.exp(-0.5 * np.sum(X_centered @ inv * X_centered, axis=1))
    p = coef * exp_term
    return p

# Выбор значения порога
# def select_threshold(yval, pval):
#     eps = 0
#     F1 = 0
#     # ------ добавьте свой код --------
#     # ...
#     # ---------------------------------
#     return eps, F1
# Выбор значения порога
def select_threshold(yval, pval):
    epsilons = np.linspace(np.min(pval), np.max(pval), 1000)
    f_best = 0
    epsilon_best = 0

    for epsilon in epsilons:
        preds = pval < epsilon
        preds = preds.reshape(len(yval), 1)
        tp = np.sum((preds == 1) & (yval == 1))
        fp = np.sum((preds == 1) & (yval == 0))
        fn = np.sum((preds == 0) & (yval == 1))
        prec = tp / (tp+fp)
        rec = tp / (tp+fn)
        f1 = 2 * prec * rec / (prec+rec)
        if f1 > f_best:
            f_best = f1
            epsilon_best = epsilon

    return epsilon_best, f_best


# Отображение выборки с графиком распределения
# def draw_data_and_fit(X, mu, sigma2, epsilon):
#     plt.figure()
#     # ------ добавьте свой код --------
#     # ...
#     # ---------------------------------
#     plt.show()
def draw_data_and_fit(X, mu, sigma2, epsilon):
    # 1. Вычисляем вероятности для всех точек
    p = multivariate_gaussian(X, mu, sigma2)
    # 2. Определяем аномалии
    outliers = p < epsilon

    # 3. Отображаем точки: нормальные и аномальные
    plt.figure()
    plt.scatter(X[~outliers, 0], X[~outliers, 1], c='b', edgecolors='k', label='Нормальные')
    plt.scatter(X[outliers, 0], X[outliers, 1], c='r', edgecolors='k', label='Аномалии')

    # 4. Строим сетку для отображения границы
    x1 = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    x2 = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
    X1, X2 = np.meshgrid(x1, x2)
    XX = np.column_stack([X1.ravel(), X2.ravel()])
    Z = multivariate_gaussian(XX, mu, sigma2)
    Z = Z.reshape(X1.shape)

    # 5. Рисуем границу уровня epsilon
    contour = plt.contour(X1, X2, Z, levels=[epsilon], linewidths=2, colors='g')
    plt.xlabel('Пропускная способность (Мбит/с)')
    plt.ylabel('Задержка ответа (мсек)')
    plt.title('Обнаружение аномалий и граница порога')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # plt.ion()
    main()
    input('Перейдите в терминал и нажмите Enter для завершения')
    plt.clf()
