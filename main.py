import numpy as np
import matplotlib.pyplot as plt
import math
import time


def f(x):
    return np.array([1, x[0], x[1], x[0]**2, x[0] * x[1], x[1]**2, x[0]**3, x[0]**2 * x[1], x[0] * x[1]**2, x[1]**3]).T


N = 10
P = 20


def create_plan(seq):
    x, p = [], []
    for i, _x in enumerate(seq):
        for j, _y in enumerate(seq):
            x.append((_x, _y))
            p.append(1.0 / len(seq)**2)
    return x, p


def find_extr(x, p):
    grid = np.arange(-1, 1.1, 0.2)
    max_fi = fi(x[0], x, p)
    min_dot = x[0]
    for _x in grid:
        for _y in grid:
            fi_cur = fi((_x, _y), x, p)
            if fi_cur > max_fi:
                max_fi = fi_cur
                min_dot = (_x, _y)
    return max_fi, min_dot


def fi(x0, x, p):
    fx = f(x0)
    M = find_m(x, p)
    D = find_d(M)
    return np.dot(np.dot(fx.T, D), fx)


def find_m(x, p):
    M = np.zeros((N, N))
    for i in range(len(x)):
        fx = f(x[i])
        M += p[i] * np.outer(fx, fx.T)
    return M


def find_d(M):
    return np.linalg.inv(M)


def is_optimal(max_fi, iter):
    delta = math.fabs(max_fi) * 0.01
    return math.fabs(-max_fi + N) <= delta


def update(x, p, a, dot):
    for i in range(len(x)):
        p[i] = (1 - a) * p[i]
    x.append(dot)
    p.append(a)
    return x, p


def psy(x, p):
    M = find_m(x, p)
    return math.log(np.linalg.det(M))


def union_points(x, p):
    eps = 0.01
    for i in range(len(x) - 1, -1, -1):
        j = 0
        while j < i:
            if np.sqrt(np.sum(np.square(np.array(x[i]) - np.array(x[j])))) < eps:
                x[j] = np.array(x[j]) * p[j] + np.array(x[i]) * p[i]
                p[j] += p[i]
                x[j] /= p[j]
                x.pop(i)
                p.pop(i)
                break
            j += 1
    return x, p


def delete_points(x, p):
    eps = 0.01  # 1 / len(p)
    p_del = 0
    for i in range(len(x) - 1, -1, -1):
        if p[i] < eps:
            x.pop(i)
            p_del += p.pop(i)
    for i in range(len(p)):
        p[i] += p_del / len(p)
    return x, p


def clean(x, p):
    show_plan(x, p, 'output/fig_unclean.png')
    x, p = union_points(x, p)
    x, p = delete_points(x, p)
    return x, p


def alg(x, p):
    iter = 0
    while True:
        a = 1 / len(x)
        max_fi, min_dot = find_extr(x, p)
        if iter % P == 0:
            print(f'iter = {iter}, max_fi = {max_fi}', end='\n')
        if is_optimal(max_fi, iter):
            clean(x, p)
            if is_optimal(max_fi, iter):
                break
        else:
            while True:
                x_new, p_new = x[:], p[:]
                update(x_new, p_new, a, min_dot)
                if psy(x_new, p_new) <= psy(x, p):
                    a /= 2
                else:
                    x, p = x_new, p_new
                    iter += 1
                    break
    return x, p


def show_plan(x, p, fig):
    plt.clf()
    plt.title(fig)
    for i in range(len(x)):
        plt.scatter(x[i][0], x[i][1])
    plt.savefig(fig)


def output_plan(x, p):
    f = open('output/final_plan.txt', 'w+')
    for i in range(len(x)):
        f.write(f'({x[i][0]:+.2f}, {x[i][1]:+.2f}) : {p[i]:+.4f}\n')


if __name__ == '__main__':
    time_0 = time.time()
    N = len(f((0, 0)))
    dots = [-1, -0.75, -0.5, 0, 0.5, 0.75, 1]
    x, p = create_plan(dots)
    show_plan(x, p, 'output/fig_start.png')
    x, p = alg(x, p)
    show_plan(x, p, 'output/fig_final.png')
    output_plan(x, p)
    print(time.time() - time_0)
