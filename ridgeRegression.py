import numpy as np
from tqdm.autonotebook import tqdm


def constrH(dataset_with_description, thetas):
    """
    Функция конструирования матрицы H -- матрицы функций \phi

    :param x: входные вектора
    :param thetas: дескрипторы (нелинейные параметры)
    :return: матрица для линейной регрессии и ее нормировочный член
    """
    dataset, struct_description = dataset_with_description
    pairs = struct_description[4]
    atoms = struct_description[5]
    m = dataset[0].shape[0]  # кол-во строк равно кол-ву элементов датасета
    n = len(thetas) - len(atoms) + len(pairs) # кол-во столбцов = кол-ву различных функций \phi
    H = np.zeros((m, n))
    b = dataset[0].shape[1]
    a = dataset[1].shape[1]
    t = dataset[2].shape[1]
    p = dataset[3].shape[1]

    qq = np.zeros(len(pairs))  #  q[i] -- заряд i-го атома в списке всех зарядов
    q = thetas[a + b + t + p: a + b + t + p + len(atoms)]
    for i, pair in enumerate(pairs):
        qq[i] = q[pair[0] - 1] * q[pair[1] - 1]

    assert a + b + t + p * 2 == n
    H[:, :b] = (dataset[0] - thetas[0:b]) ** 2
    H[:, b: a + b] = (dataset[1] - thetas[b: a + b]) ** 2
    H[:, a + b : a + b + t] = (1 + np.cos(dataset[2] - thetas[a + b : a + b + t]))
    H[:, a + b + t : a + b + t + p] = (thetas[a + b + t: a + b + t + p] / dataset[3]) ** 12 - \
                                      2 * (thetas[a + b + t: a + b + t + p] / dataset[3]) ** 6
    H[:, a + b + t + p: a + b + t + p + p] = qq / dataset[3]  # thetas[a + b + t + p: a + b + t + p + p]
    stdH = H.std(axis=0, )
    H /= stdH  # нормировка на среднее квадратическое значение, в H_test нужно учесть
    return H, stdH


def RidgeRegression(H, y, l):
    """
    Функция, осуществляющая поиск линейных параметров 'C' с помощью МНК

    :param H: матрица функций
    :param y: вектор выходных значений (энергии)
    :param l: параметр регуляризации
    :return: линейные коэффициенты, приближенное значение энергий (y)
    """
    Nf = H.shape[1]
    I = np.eye(Nf)
    C = np.linalg.inv(H.T.dot(H) + 2 * l * I).dot(H.T.dot(y))
    y_est = H.dot(C)
    return C, y_est


def LOOCV(H, y, y_est):
    M = H.shape[0]
    varH = np.linalg.norm(H - H.mean(axis=0), axis=1) ** 2
    h = 1 / M + varH / varH.sum()
    loocv = 1 / M * (((y - y_est) / (1 - h)) ** 2).sum()
    return loocv


def RR_LOOCV(x, y, l, thetas):
    H, _ = constrH(x, thetas)
    _, y_est = RidgeRegression(H, y, l)
    L = LOOCV(H, y, y_est)
    return L


def myLOO(x, y, H):
    M = len(x)
    MSE = 0
    for i in range(M):
        H_loo = np.delete(H, i, 0)
        y_loo = np.delete(y, i)
        C, _ = RidgeRegression(H_loo, y_loo, l)
        y_est = H[i].dot(C)
        MSE += (y[i] - y_est) ** 2
        print(f'MSE myLOO: {y[i] - y_est}')
    return MSE/M