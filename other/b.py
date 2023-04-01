import numpy as np
import matplotlib.pyplot as plt
import random

#возвращает вектор со значениями нашей функции
def func(x):
    return np.array([1.0, x[0], x[1], x[0]*x[1], x[0]**2, x[1]**2])

#region работа с файлами
#считывание из файла: сначала количество элементов, потом все точки x, потом все веса p
def read(filename):
    f = open(filename, "r")
    Xmass = []
    Pmass = []
    count = int(f.readline())
    for i in range(count):
        Xmass.append([float(item) for item in f.readline().split(" ")])
    for i in range(count):
        Pmass.append(float(f.readline()))
    return Xmass, Pmass

#Запись в файл, записывается так, чтобы потом считать
def write(filename, x, p):
    f = open(filename, "w")
    f.write(str(len(x)) + "\n")
    for i in x:
        f.write(str(i[0]) + " " + str(i[1]) + "\n")
    f.write(str(p[0]))
    for i in p[1:]:
        f.write("\n" + str(i))
    f.close()

#функция которая создает файл, с первоначальными данными
def first_input(filename):
    f = open(filename, "w")
    # x1 = np.arange(-1, 1.1, 0.5)
    x1 = np.array([-1, -0.5, 0, 0.5, 1])
    l = len(x1)**2
    f.write(str(len(x1)*len(x1)) + "\n")
    for i in x1:
        for j in x1:
            f.write(str(i) + " " + str(j) + "\n")
    for i in range(l - 1):
        f.write(str(1.0/l) + "\n")
    f.write((str(1.0/l)))
    f.close()
#endregion

#region Создание матрицы М, D и работа с ними
#создание матрицы М
def makeM(x, p):
    M = np.zeros((len(func(x[0])), len(func(x[0]))))
    for i in range(len(x)):
        M += p[i] * make_partM(func(x[i]))
    return M

#создание части матрицы, для каждого из весов
def make_partM(fx):
    M = np.zeros((len(fx), len(fx)))
    for i in range(len(fx)):
        for j in range(len(fx)):
            M[i][j] = fx[i] * fx[j]
    return M

#создание матрицы D из матрицы М
def makeD(M):
    return np.linalg.inv(M)

#Проверка на D-оптимальность ( вычисление критерия для заданной матрицы)
def D_optim(M):
    return np.linalg.det(M)
#endregion

#нахождение в массиве X индекса элемента, который близок к элементу x
def findClose(x, X):
    for i in range(len(X)):
        vec = np.array([x[0]-X[i][0],x[1]-X[i][1]])
        scal = np.dot(vec, vec)
        if np.sqrt(scal)**2 < 0.1:
            return i
    return -1

#функция нахождения максимума на сетке, для добавления новой точки
def findMaxFi(grid, D):
    max = np.dot(np.dot(func([grid[0], grid[0]]), D), func([grid[0], grid[0]]).T)
    maxdot = [grid[0], grid[0]]

    for i in grid:
        for j in grid:
            f = np.dot(np.dot(func([i, j]), D), func([i, j]).T)
            if f > max:
                max = f
                maxdot = [i, j]
    return max, maxdot

#Подбор веса для новой точки
def addToP(curientD, p, x):
    newD = curientD - 1
    ksy = 1
    while curientD > newD:
        newP = p.copy()
        for i in range(len(newP)):
            newP[i] = (1.0 - ksy / len(newP)) * newP[i]
        newP.append(ksy / len(newP))
        newM = makeM(x, newP)
        newD = D_optim(newM)
        ksy /= 2
    return newP

#Добавление новой точки
def addNewPoint(x, p, grid):
    M = makeM(x, p)
    D = makeD(M)
    curientD = D_optim(M)
    max, maxdot = findMaxFi(grid, D)
    x.append(maxdot)
    p = addToP(curientD, p, x)
    eps = max * 0.01
    print("max ", max)
    delta = abs(max - np.trace(np.dot(M, D)))
    print(delta)
    return delta, eps, x, p

#объединение близких точек
def unionCloseDots(x, p):
    newX = [x[0]]
    newP = [p[0]]
    for i in range(1, len(x)):
        index = findClose(x[i], newX)
        if index == -1:
            newX.append(x[i])
            newP.append(p[i])
        else:
            newP[index] += p[i]
    x = newX
    p = newP
    return x, p

#удаление точек с малыми весами
def removeDotsWithSmallWeitgh(x, p):
    sum = 0
    index = 0
    for i in range(len(p)):
        if p[i] < 0.02:
            sum += p[i]
            p[i] = 0
            x[i] = [0, 0]
            index += 1
    for i in range(index):
        p.remove(0)
        x.remove([0, 0])
    sum /= len(p)
    for i in range(len(p)):
        p[i] += sum
    return x, p

first_input('input.txt')
x, p = read("input.txt")
grid = np.arange(-1, 1.005, 0.01)
delta = 1
eps = 0.01
max = 0
bigeps = 0.01
exit = 1
m = 6
iteration = 1
while abs(exit) > bigeps:
    while delta > eps:
        delta, eps, x, p = addNewPoint(x, p, grid)

    if (iteration % 10 == 0) or (iteration == 1):
        write("first" + str(iteration) + ".txt", np.around(x, 5), np.around(p, 5))
        #график сразу после нахождения всех точек
        for i in x:
            plt.scatter(i[0], i[1])
        plt.show()

    x, p = unionCloseDots(x, p)

    if (iteration % 10 == 0) or (iteration == 1):
        write("second" + str(iteration) + ".txt", np.around(x, 5), np.around(p, 5))
        #график после объединения
        for i in x:
            plt.scatter(i[0], i[1])
        plt.show()

    x, p = removeDotsWithSmallWeitgh(x, p)

    if (iteration % 10 == 0) or (iteration == 1):
        write("third" + str(iteration) + ".txt", np.around(x, 5), np.around(p, 5))
        #график после удаления
        for i in x:
            plt.scatter(i[0], i[1])
        plt.show()

    exit, bigeps, x, p = addNewPoint(x, p, grid)

    print(exit, " ", bigeps)
    iteration += 1
