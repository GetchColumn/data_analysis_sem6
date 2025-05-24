import numpy as np
import scipy.optimize as op


def main():
    ## 1. Загрузка данных
    print('Загрузка исходных данных рекомендательной системы фильмов')
    with open('ex8movies.npy', 'rb') as f:
        Y = np.load(f)
        R = np.load(f)
        X = np.load(f)
        Theta = np.load(f)
        num_users = np.load(f)
        num_movies = np.load(f)
        num_features = np.load(f)
        movies = np.load(f)
    print('Пользователей:', num_users, '  Фильмов:', num_movies, '  Признаков:', num_features)
    n = 3
    nu = 4
    nm = 5
    Xt = X[:nm, :n]
    Tt = Theta[:nu, :n]
    Yt = Y[:nm, :nu]
    Rt = R[:nm, :nu]
    params = np.append(Xt.flatten(), Tt.flatten())

    input('Перейдите в терминал и нажмите Enter для продолжения...')
    # Удалите следующую строку для продолжения работы
    # return

    ## 2. Нормализация рейтинга к нулевому среднему значению
    print('\nНормализация рейтинга к нулевому среднему значению')
    Ynorm, Ymean = normalize_ratings(Y, R)
    print('Средняя оценка первого фильма ({}): {}'.format(movies[0], Ymean[0]))
    print('Ожидаемое значение (приближенно): 3.88')

    input('Перейдите в терминал и нажмите Enter для продолжения...')
    # Удалите следующую строку для продолжения работы
    # return

    ## 3. Функция стоимости коллаборативной фильтрации без регуляризации
    print('\nФункция стоимости коллаборативной фильтрации без регуляризации')
    lamb = 0
    J = cofi_cost(params, Yt, Rt, nu, nm, n, lamb)
    print('Полученное значение функции стоимости:', J)
    print('Ожидаемое значение (приближенно): 22.22')

    input('Перейдите в терминал и нажмите Enter для продолжения...')
    # Удалите следующую строку для продолжения работы
    # return

    ## 4. Градиент коллаборативной фильтрации без регуляризации
    print('\nГрадиент коллаборативной фильтрации без регуляризации')
    grad = cofi_gradient(params, Yt, Rt, nu, nm, n, lamb)
    diff = check_gradient(lamb)
    print('Разница между найденным градиентом и численным:', diff)
    print('Ожидаемая разница: < 1e-9')

    input('Перейдите в терминал и нажмите Enter для продолжения...')
    # Удалите следующую строку для продолжения работы
    # return

    ## 5. Функция стоимости с регуляризацией
    print('\nФункция стоимости коллаборативной фильтрации с регуляризацией')
    lamb = 1.5
    J = cofi_cost(params, Yt, Rt, nu, nm, n, lamb)
    print('Полученное значение функции стоимости c регуляризацией:', J)
    print('Ожидаемое значение (приближенно): 31.34')

    input('Перейдите в терминал и нажмите Enter для продолжения...')
    # Удалите следующую строку для продолжения работы
    # return

    ## 6. Градиент с регуляризацией
    print('\nГрадиент коллаборативной фильтрации с регуляризацией')
    grad = cofi_gradient(params, Yt, Rt, nu, nm, n, lamb)
    diff = check_gradient(lamb)
    print('Разница между найденным градиентом и численным:', diff)
    print('Ожидаемая разница: < 1e-9')

    input('Перейдите в терминал и нажмите Enter для продолжения...')
    # Удалите следующую строку для продолжения работы
    # return

    ## 7. Обучение коллаборативной фильтрации
    print('\nОбучение коллаборативной фильтрации')
    lamb = 10
    Ynorm, Ymean = normalize_ratings(Y, R)
    X, Theta = cofi_train(Ynorm, R, num_users, num_movies, num_features, lamb)
    pred = X @ Theta.T + Ymean[:, np.newaxis]
    print('Рекомендуемые фильмы к просмотру для первого пользователя:')
    user_idx = 0
    rec_idx = np.argsort(-pred[R[:, user_idx] == 0, user_idx])[:10]
    rec_movies = movies[R[:, user_idx] == 0][rec_idx]
    for name in rec_movies:
        print(name)


# Нормализация рейтинга фильмов к нулевому среднему
# def normalize_ratings(Y, R):
#     Ynorm = np.zeros(Y.shape, dtype=float)
#     Ymean = np.zeros(Y.shape[0], dtype=float)
#     # ------ добавьте свой код --------
#     # ...
#     # ---------------------------------
#     return Ynorm, Ymean
def normalize_ratings(Y, R):
    num_movies = Y.shape[0]
    Ymean = np.zeros(num_movies)
    Ynorm = np.zeros_like(Y, dtype=float)
    for i in range(num_movies):
        idx = R[i, :] == 1
        if np.any(idx):
            Ymean[i] = np.mean(Y[i, idx])
            Ynorm[i, idx] = Y[i, idx] - Ymean[i]
        else:
            Ymean[i] = 0
    return Ynorm, Ymean

# Функция стоимости коллаборативной фильтрации
# def cofi_cost(params, Y, R, num_users, num_movies, num_features, lamb):
#     J = 0
#     # ------ добавьте свой код --------
#
#     # Для шага 3 (без регуляризации)
#     # ...
#
#     # Для шага 5 (с регуляризацией)
#     # ...
#
#     # ---------------------------------
#     return J
def cofi_cost(params, Y, R, num_users, num_movies, num_features, lamb):
    # Восстановление X и Theta из params
    X = params[:num_movies * num_features].reshape((num_movies, num_features))
    Theta = params[num_movies * num_features:].reshape((num_users, num_features))
    # Предсказания
    pred = X @ Theta.T
    # Ошибка только по тем, где R == 1
    error = (pred - Y) * R
    # Стоимость (без регуляризации)
    J = 0.5 * np.sum(error ** 2)
    J += (lamb / 2) * (np.sum(Theta ** 2) + np.sum(X ** 2))
    # Регуляризация
    return J

# Функция стоимости коллаборативной фильтрации
# def cofi_gradient(params, Y, R, num_users, num_movies, num_features, lamb):
#     grad = np.zeros(params.shape, dtype=float)
#     # ------ добавьте свой код --------
#
#     # Для шага 4 (без регуляризации)
#     # ...
#
#     # Для шага 6 (с регуляризацией)
#     # ...
#
#     # ---------------------------------
#     return grad
def cofi_gradient(params, Y, R, num_users, num_movies, num_features, lamb):
    # Восстановление X и Theta из params
    X = params[:num_movies * num_features].reshape((num_movies, num_features))
    Theta = params[num_movies * num_features:].reshape((num_users, num_features))
    # Ошибка только по тем, где R == 1
    error = (X @ Theta.T - Y) * R
    # Градиенты без регуляризации
    X_grad = error @ Theta + lamb * X
    Theta_grad = error.T @ X + lamb * Theta
    # Объединяем в один вектор
    grad = np.append(X_grad.flatten(), Theta_grad.flatten())
    return grad


# Обучение коллаборативной фиьтрации
# def cofi_train(Y, R, num_users, num_movies, num_features, lamb):
#     X = np.zeros((num_users, num_features), dtype=float)
#     Theta = np.zeros((num_movies, num_features), dtype=float)
#     # ------ добавьте свой код --------
#     # ...
#     # ---------------------------------
#     return X, Theta
def cofi_train(Y, R, num_users, num_movies, num_features, lamb):
    # Случайная инициализация параметров
    X_init = 0.01 * np.random.randn(num_movies, num_features)
    Theta_init = 0.01 * np.random.randn(num_users, num_features)
    params_init = np.append(X_init.flatten(), Theta_init.flatten())

    # Определяем функцию стоимости и градиента
    def cost_func(p):
        return cofi_cost(p, Y, R, num_users, num_movies, num_features, lamb)
    def grad_func(p):
        return cofi_gradient(p, Y, R, num_users, num_movies, num_features, lamb)

    # Минимизация
    res = op.minimize(cost_func, params_init, jac=grad_func, method='TNC', options={'maxiter': 100})

    # Восстановление X и Theta из результата
    params_opt = res.x
    X = params_opt[:num_movies * num_features].reshape((num_movies, num_features))
    Theta = params_opt[num_movies * num_features:].reshape((num_users, num_features))
    return X, Theta

# Функция определения численного градиента (вам не нужно менять код этой функции)
def numerical_gradient(J, theta):
    grad = np.zeros(theta.shape, dtype=float)
    p = np.zeros(theta.shape, dtype=float)
    eps = 1e-4
    for i in range(theta.shape[0]):
        p[i] = eps
        J1 = J(theta - p)
        J2 = J(theta + p)
        grad[i] = (J2 - J1) / (2 * eps)
        p[i] = 0
    return grad


# Функция проверки градиента, сравнение его с численным значением (вам не нужно менять код этой функции)
def check_gradient(lamb):
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)
    Y = X_t @ Theta_t.T
    Y[np.random.random_sample(Y.shape) > 0.5] = 0
    R = np.zeros(Y.shape, dtype=int)
    R[Y != 0] = 1
    X = np.random.random_sample(X_t.shape) - 0.5
    Theta = np.random.random_sample(Theta_t.shape) - 0.5
    nu, n = Theta.shape
    nm = X.shape[0]
    params = np.append(X, Theta)
    J = lambda t: cofi_cost(t, Y, R, nu, nm, n, lamb)
    numgrad = numerical_gradient(J, params)
    grad = cofi_gradient(params, Y, R, nu, nm, n, lamb)
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    return diff


if __name__ == '__main__':
    main()
    input('Перейдите в терминал и нажмите Enter для завершения')
