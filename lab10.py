import numpy as np
from sklearn import svm 
import matplotlib.pyplot as plt  


def main():
    ## 1. Загрузка и отображение первого набора данных
    with open('ex6data1.npy', 'rb') as f:
        X = np.load(f)
        y = np.load(f)  
    print('Отображение первого набора обучающего набора данных')
    display_boundary(X, y, 'Набор 1')
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 2. Обучение SVM с линейным ядром
    print('Обучение SVM с линейным ядром и С=1')
    C = 1
    model = svm_learn(X, y, C)
    display_boundary(X, y, 'C='+str(C), model)
    input('Перейдите в терминал и нажмите Enter для продолжения...')
    print('Обучение SVM с линейным ядром и С=100')
    C = 100
    model = svm_learn(X, y, C)
    display_boundary(X, y, 'C='+str(C), model)
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 3. Ядро Гаусса
    print('Реализация ядра Гаусса')
    x1 = np.array([1, 2, 1])
    x2 = np.array([0, 4, -1])
    sigma = 2
    sim = gaussian_kernel(x1, x2, sigma)
    print('Ядро Гаусса между x1 = [1; 2; 1], x2 = [0; 4; -1] при sigma = 2 :', sim)
    print('Ожидаемое значени (приблизительно): 0.324652')
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 4. Загрузка и отображение второго набора данных
    with open('ex6data2.npy', 'rb') as f:
        X = np.load(f)
        y = np.load(f)  
    print('Отображение второго набора обучающего набора данных')
    display_boundary(X, y, 'Набор 2')
    input('Перейдите в терминал и нажмите Enter для продолжения...')
    C = 1
    model = svm_learn(X, y, C)
    display_boundary(X, y, 'SVM с линейным ядром', model)
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 5. Обучение SVM с ядром Гаусса
    print('Обучение SVM с ядром Гаусса')
    C = 1
    sigma = 0.1
    model = svm_learn(X, y, C, sigma)
    display_boundary(X, y, 'SVM с ядром Гаусса', model)
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 6. Загрузка и отображение третьего набора данных
    with open('ex6data3.npy', 'rb') as f:
        X = np.load(f)
        y = np.load(f)  
        Xval = np.load(f)
        yval = np.load(f)  
    print('Отображение третьего обучающего набора данных')
    display_boundary(X, y, 'Набор 3')
    input('Перейдите в терминал и нажмите Enter для продолжения...')
    C = 2
    sigma = 0.01
    model = svm_learn(X, y, C, sigma)
    display_boundary(X, y, 'SVM с ядром Гаусса (C = '+str(C)+', sigma = '+str(sigma)+')', model)
    input('Перейдите в терминал и нажмите Enter для продолжения...')
    C = 2
    sigma = 5
    model = svm_learn(X, y, C, sigma)
    display_boundary(X, y, 'SVM с ядром Гаусса (C = '+str(C)+', sigma = '+str(sigma)+')', model)
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 7. Подбор параметров для модели
    print('Подбор оптимальный параметров C и sigma для модели')
    C, sigma = adjust_params(X, y, Xval, yval)
    print('Найденные оптимальные параметры С =', C, 'sigma = ', sigma)
    input('Перейдите в терминал и нажмите Enter для продолжения...')

    # Удалите следующую строку для продолжения работы
    return

    ## 8. Оценка модели
    print('Оценка вероятности ошибки построенной оптимальной модели')
    model = svm_learn(X, y, C, sigma)
    display_boundary(X, y, 'SVM с ядром Гаусса (оптимальные параметры)', model)
    pred = model.predict(Xval)
    err = np.mean((pred != yval).astype(float))
    print('Вероятность ошибки модели:', err)

    ## Конец работы


# Функция ядра Гаусса
def gaussian_kernel(x1, x2, sigma):
    sim = 0
    # ------ добавьте свой код --------
    # ...
    # ---------------------------------
    return sim


# Подбор параметров для SVM-модели
def adjust_params(X, y, Xval, yval):
    C = 1
    sigma = 1
    # ------ добавьте свой код --------
    # ...
    # ---------------------------------
    return C, sigma


# Обучение SVM из библиотеки sklearn
def svm_learn(X, y, C, sigma=None, tol=1e-3, max_iter=-1):
    if sigma is not None:
        gamma = 1 / (2 * sigma**2)
        clf = svm.SVC(C=C, gamma=gamma, kernel='rbf', tol=tol, max_iter=max_iter)
    else:
        clf = svm.SVC(C=C, kernel='linear', tol=tol, max_iter=max_iter)
    return clf.fit(X, y)


# Отображение обучающей выборки и границы решения SVM-модели
def display_boundary(X, y, title, model=None):
    plt.figure()
    plt.scatter(X[(y == 0), 0], X[(y == 0), 1], c='r', marker='o')
    plt.scatter(X[(y == 1), 0], X[(y == 1), 1], c='g', marker='o')
    if model is not None and model.kernel == 'linear':
        w = model.coef_[0]
        a = -w[0] / w[1]
        xx = np.array([X[:,0].min(), X[:,0].max()])
        yy = a * xx - (model.intercept_[0]) / w[1]
        plt.plot(xx, yy, 'b-')
    if model is not None and model.kernel == 'rbf':
        x1plot = np.linspace(X[:,0].min(), X[:,0].max(), 100).T
        x2plot = np.linspace(X[:,1].min(), X[:,1].max(), 100).T
        X1, X2 = np.meshgrid(x1plot, x2plot)
        vals = np.zeros(X1.shape)
        for i in range(X1.shape[1]):
            this_X = np.column_stack((X1[:, i], X2[:, i]))
            vals[:, i] = model.predict(this_X)
        plt.contour(X1, X2, vals, colors="b")
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    plt.ion()
    main()
    input('Перейдите в терминал и нажмите Enter для завершения')
    plt.clf()
