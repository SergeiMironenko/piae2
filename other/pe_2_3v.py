import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import Bounds

def show_plan(x):
    x0 = []
    x1 = []
    for i in range(len(x)):
        x0.append(x[i][0])
        x1.append(x[i][1])
    plt.plot(x0, x1, 'ro')
    plt.show()


def model(x1,x2):
    return np.array([[1], [x1],[x2],[x1*x2], [x1*x1],[x2*x2]])

# Построение информационной матрицы
def get_m(x, p):
    M = np.zeros((6, 6))
    for i in range(len(p)):
        f_value = model(x[i][0], x[i][1])
        M += p[i] * np.dot(f_value, f_value.T)
    return M


def fi(x, M):
    M_inv = np.linalg.inv(M)
    f_value = model(x[0], x[1])
    return -np.dot(f_value.T,  np.dot(np.linalg.matrix_power(M_inv, 2).T, f_value))


def minimization(x0, M):
    result = optimize.minimize(fi, x0, method="Nelder-Mead", args=(M), bounds=Bounds(-1, 1)).x
    return result


def a_functional(M):
    M_inv = np.linalg.inv(M)
    return np.trace(M_inv)

# Последовательный алгоритм
def sequential_algorithm(x, p):
    iter = 0
    while True:
        a = 1 / len(p)
        M = get_m(x, p)
        if iter % 10 == 0:
            print("iter=", iter)
            print("a_functional=", a_functional(M))
        new_point = minimization(np.random.uniform(-1, 1, 2), M)
        if is_A_optimal(new_point, M):
            break
        x_new = x[:]
        x_new.append(new_point)
        while True:
            p_new = p[:]
            for i in range(len(p)):
                p_new[i] = (1-a)*p[i]
            p_new.append(a)
            M_new = get_m(x_new, p_new)
            if a < 0.001:
                break
            elif a_functional(M) <= a_functional(M_new):
                a /= 2
            else:
                x, p = plan_cleaning(x_new, p_new)
                break
        iter += 1
    print("iter=", iter)
    print("a_functional=", a_functional(M))
    return x, p


def is_A_optimal(x, M):
    M_inv = np.linalg.inv(M)
    sigma = abs(fi(x, M)) * 0.0001 # Должен быть множитель 0.01, но т.к. из-за такого условия рано 
                                   # прекращается алгоритм поменял множитель на меньший
    #print("sigma = " + str(sigma))
    #print('expression = ' + str(abs(-fi(x, M) + -np.trace(np.dot(M, np.linalg.matrix_power(M_inv, 2))))))
    if abs(-fi(x, M) + -np.trace(np.dot(M, np.linalg.matrix_power(M_inv, 2)))) <= sigma:
        return True
    return False


# Очистка плана
def plan_cleaning(x, p):
    q = len(p)
    i = q-1
    while i >= 0:
        j = 0
        while j < i:
            if (x[i] - x[j]).T @ (x[i] - x[j]) < 0.01:
                x[j] = x[i]*p[i] + x[j]*p[j]
                p[j] += p[i]
                x[j] /= p[j]
                p.pop(i)
                x.pop(i)
                q -= 1
                break
            j += 1
        if j == i and p[i] < 0.01:
            p_sum = np.sum([p[k] for k in range(len(p)) if i!=k])
            p = [p[k]/p_sum for k in range(len(p))]
            p.pop(i)
            x.pop(i)
            q -= 1
        i -= 1
    return x, p


if __name__ == "__main__":
    m=5
    n = m*m  # количество точек плана
    p = [1/n] * n # веса

    # dots = np.linspace(-1, 1, m)
    dots = np.array([-1, -0.5, 0, 0.5, 1])
    x = []
    for i in range(n):
        x.append(np.array([dots[i // m], dots[i % m]]))
    show_plan(x)
    x, p = sequential_algorithm(x, p)
    show_plan(x)
