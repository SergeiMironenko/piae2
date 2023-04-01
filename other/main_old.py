import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import math


# Кубическая двухфакторная модель
def f(x):
    # return np.array([1, x[0], x[1], x[0]**2, x[0] * x[1], x[1]**2, x[0]**3, x[0]**2 * x[1], x[0] * x[1]**2, x[1]**3]).T
    return np.array([1, x[0], x[1], x[0] ** 2, x[0] * x[1], x[1] ** 2]).T


# 1. Создание начального плана
def create_plan(seq):
    x, p = [], []
    for i, _x in enumerate(seq):
        for j, _y in enumerate(seq):
            x.append((_x, _y))
            p.append(1.0 / len(seq)**2)
    return x, p


# 2. Поиск точки экстремума
def find_extremum(x, p, x0=np.array([1, 1])):
    result = sp.optimize.minimize(find_fi, x0, args=(x, p), bounds=sp.optimize.Bounds(-1, 1))
    return result.fun, tuple(result.x)


# 2.1. Поиск фи
def find_fi(x0, x, p):
    # fx = f(x0)
    # M = find_M(x, p)
    # D = find_D(M)
    # return np.dot(np.dot(fx.T, D), fx)
    fx = f(x0)
    M = find_M(x, p)
    D = find_D(M)
    Mx = np.outer(fx, fx.T)
    return np.trace(np.dot(D, Mx))


def find_fi_rev(x0, x, p):
    return -find_fi(x0, x, p)


# 2.2. Поиск M
def find_M(x, p, n=6):
    M = np.zeros((n, n))
    for i in range(len(x)):
        fx = f(x[i])
        M += p[i] * np.outer(fx, fx.T)
    return M


# 2.3. Поиск D
def find_D(M):
    return np.linalg.inv(M)


# 3. Проверка условий оптимальности планов
def find_opt(x, p, extr_fi, n=6):
    M = find_M(x, p)
    # D = find_D(M)
    opt = math.fabs(-extr_fi + n)  # n = 10 = np.trace(np.dot(M, D))
    return opt


# 4. Составление нового плана (пересчет весов и добавление новой точки)
def update_plan(x, p, a, extr_point):
    for i in range(len(x)):
        p[i] = (1 - a) * p[i]
    x.append(extr_point)
    p.append(a)
    return x, p


# 5. Поиск функционала
def find_psy(x, p):
    M = find_M(x, p)
    dt = np.linalg.det(M)
    return math.log(np.linalg.det(M))


# 6. Объединение точек
def union_points(x, p):
    eps = 0.01
    for i in range(len(x) - 1, -1, -1):
        j = 0
        while j < i:
            if np.dot(np.array(x[i]) - np.array(x[j]), np.array(x[i]) - np.array(x[j])) < eps:
                p[j] += p[i]
                x[j] = np.array(x[j]) * p[j] + np.array(x[i]) * p[i]
                x[j] /= p[j]
                x.pop(i)
                p.pop(i)
                break
            j += 1
    return x, p


# 7. Выброс точек
def delete_points(x, p):
    eps = 0.01
    p_del = 0
    for i in range(len(x) - 1, -1, -1):
        if p[i] < eps:
            x.pop(i)
            p_del += p.pop(i)
    for i in range(len(p)):
        p[i] += p_del / len(p)



    return x, p


# Вывод
def show_plan(x, p):
    plt.plot([i[0] for i in x], [i[1] for i in x], 'ro')
    plt.show()


# Последовательный алгоритм
def seq_algorithm():
    # x, p = create_plan([-1, -0.75, -0.5, 0, 0.5, 0.75, 1])
    x, p = create_plan([-1, -0.5, 0, 0.5, 1])
    show_plan(x, p)
    iter, max_iter = 0, 300

    while True:
        # Экстремум функции и точка экстремума
        extr_fi, extr_point = find_extremum(x, p)
        delta = extr_fi * 0.01
        opt = find_opt(x, p, extr_fi)
        if opt <= delta:
            break
            # x, p = union_points(x, p)
            # x, p = delete_points(x, p)
            # extr_fi, extr_point = find_extremum(x, p)
            # delta = extr_fi * 0.01
            # opt = find_opt(x, p, extr_fi)
            # if opt <= delta:
            #     break
        a = 1.0 / len(x)
        while True:
            psy1 = find_psy(x, p)
            x_new, p_new = x[:], p[:]
            update_plan(x_new, p_new, a, extr_point)  # 4
            psy2 = find_psy(x_new, p_new)
            # print(f'p1 = {psy1}, p2 = {psy2}')
            if psy2 >= psy1:
                print('hoba')
                a /= 2
            else:
                x, p = x_new, p_new
                break
        if iter % 20 == 0:
            print(f'opt = {opt}, delta = {delta}, extr_fi = {extr_fi}, extr_point = {extr_point}, n = {len(x)}, a = {a}')
        iter += 1

    print('konec')
    # show_plan(x, p)
    # print(f'x = {x}, p = {p}')
    # x, p = union_points(x, p)
    # print(f'x = {x}, p = {p}')
    # x, p = delete_points(x, p)
    # print(f'x = {x}, p = {p}')
    show_plan(x, p)


def fun(x):
    # z = (2 * x[0])**2 + x[1]**2
    z = 1 / (1 + x[0]**2) + 1 / (1 + x[1]**2)
    z = x[0]**2 + (x[1] - 1)**2
    return z


def test():
    x0 = np.array([1, 1])
    res = sp.optimize.minimize(fun, x0, bounds=sp.optimize.Bounds(-1, 1))
    print(f'x = {res.x}, fun = {res.fun}')


if __name__ == '__main__':
    seq_algorithm()
    # test()
    # a1 = np.array([1, 2])
    # b1 = np.array([5, -4])
    # print(a1 @ b1)
    # print(np.dot(a1, b1))
