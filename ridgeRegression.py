import numpy as np
from moleculToVector import DataSetMatrix, StructDescription
from tqdm.autonotebook import tqdm


def constrH(dataset_matrix: DataSetMatrix, struct_description: StructDescription, thetas):
    """
    Функция конструирования матрицы H -- матрицы функций \phi (только энергии)

    :param dataset_with_description: тупл (dataset_matrix, struct_description), возвращаемый get_dataset
    :param thetas: нелинейные параметры
    :return: матрица для линейной регрессии и ее нормировочный член
    """

    pairs = struct_description.pairs
    atoms = struct_description.atoms
    m = dataset_matrix.bonds_matrix.shape[0]  # кол-во строк равно кол-ву элементов датасета
    n = len(thetas) - len(atoms) + len(pairs)  # кол-во столбцов = кол-ву различных функций \phi
    H = np.zeros((m, n))
    b = dataset_matrix.bonds_matrix.shape[1]
    a = dataset_matrix.angles_matrix.shape[1]
    t = dataset_matrix.torsions_matrix.shape[1]
    p = dataset_matrix.pairs_matrix.shape[1]

    qq = np.zeros(len(pairs))  # q[i] -- заряд i-го атома в списке всех зарядов
    q = thetas[a + b + t + p: a + b + t + p + len(atoms)]
    for i, pair in enumerate(pairs):
        qq[i] = q[pair[0]] * q[pair[1]]

    assert a + b + t + p * 2 == n
    H[:, :b] = (dataset_matrix.bonds_matrix - thetas[0:b]) ** 2
    H[:, b: a + b] = (dataset_matrix.angles_matrix - thetas[b: a + b]) ** 2
    H[:, a + b: a + b + t] = (1 + np.cos(dataset_matrix.torsions_matrix - thetas[a + b: a + b + t]))
    H[:, a + b + t: a + b + t + p] = (thetas[a + b + t: a + b + t + p] / dataset_matrix.pairs_matrix) ** 12 - \
                                     2 * (thetas[a + b + t: a + b + t + p] / dataset_matrix.pairs_matrix) ** 6
    H[:, a + b + t + p: a + b + t + p + p] = qq / dataset_matrix.pairs_matrix
    # thetas[a + b + t + p: a + b + t + p + p]
    stdH = H.std(axis=0, )
    # H /= stdH  # нормировка на среднее квадратическое значение, в H_test нужно учесть
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


def RR_LOOCV(x, y, l, thetas, descr):
    H, _ = constrH(x, descr, thetas)
    _, y_est = RidgeRegression(H, y, l)
    L = LOOCV(H, y, y_est)
    return L


# def myLOO(x, y, l, H):
#     M = len(x)
#     MSE = 0
#     for i in range(M):
#         H_loo = np.delete(H, i, 0)
#         y_loo = np.delete(y, i)
#         C, _ = RidgeRegression(H_loo, y_loo, l)
#         y_est = H[i].dot(C)
#         MSE += (y[i] - y_est) ** 2
#         print(f'MSE myLOO: {y[i] - y_est}')
#     return MSE / M


def constrHH(dataset: DataSetMatrix, struct_descr: StructDescription, thetas):
    """
    Функция конструирования матрицы H -- матрицы функций \phi (энергии и силы)

    :param dataset:
    :param struct_descr
    :param thetas: нелинейные параметры
    :return: матрица для линейной регрессии и ее нормировочный член
    """

    pairs = struct_descr.pairs
    atoms = struct_descr.atoms
    m = dataset.bonds_matrix.shape[0]  # кол-во строк равно кол-ву элементов датасета
    n = len(thetas) - len(atoms) + len(pairs)  # кол-во столбцов = кол-ву различных функций \phi
    b = dataset.bonds_matrix.shape[1]
    a = dataset.angles_matrix.shape[1]
    t = dataset.torsions_matrix.shape[1]
    p = dataset.pairs_matrix.shape[1]

    #  Для кулоновских сил сконструируем массив парных переменоженных зарядов в соответствии с описанием датасета:
    qq = np.zeros(len(pairs))  # q[i] -- заряд i-го атома в списке всех зарядов
    q = thetas[a + b + t + p: a + b + t + p + len(atoms)]

    for i, pair in enumerate(pairs):
        qq[i] = q[pair[0] - 1] * q[pair[1] - 1]

    HH = np.zeros((3 * len(atoms), m, n))

    for j in range(b):
        for k in range(3 * len(atoms)):

            if k // 3 == struct_descr.bonds[j][0]:
                for i in range(m):
                    bond = dataset.bonds_matrix[i][j]
                    r = dataset.coords[i, struct_descr.bonds[j][1]] - dataset.coords[i, struct_descr.bonds[j][0]]
                    HH[k][i][j] = 2 * (bond - thetas[j]) * r[k % 3] / bond

            if k // 3 == struct_descr.bonds[j][1]:
                for i in range(m):
                    bond = dataset.bonds_matrix[i][j]
                    r = dataset.coords[i, struct_descr.bonds[j][0]] - dataset.coords[i, struct_descr.bonds[j][1]]
                    HH[k][i][j] = 2 * (bond - thetas[j]) * r[k % 3] / bond

    for j in range(a):
        for k in range(3 * len(atoms)):
            for i in range(m):

                angle = dataset.angles_matrix[i, j]
                angle_s = np.sin(angle)
                angle_c = np.cos(angle)

                r12 = dataset.coords[i, struct_descr.angles[j][1]] - dataset.coords[i, struct_descr.angles[j][0]]
                r32 = dataset.coords[i, struct_descr.angles[j][1]] - dataset.coords[i, struct_descr.angles[j][2]]

                r12_n = np.linalg.norm(r12)
                r32_n = np.linalg.norm(r32)

                num1 = 2 * (angle - thetas[j + b])
                if k // 3 == struct_descr.angles[j][0]:
                    HH[k][i][j + b] = - num1 / (angle_s * r12_n) * (r12[k % 3] * angle_c / r12_n - r32[k % 3] / r32_n)

                if k // 3 == struct_descr.angles[j][2]:
                    HH[k][i][j + b] = - num1 / (angle_s * r32_n) * (r32[k % 3] * angle_c / r32_n - r12[k % 3] / r12_n)

                if k // 3 == struct_descr.angles[j][1]:
                    first =   num1 / (angle_s * r12_n) * (r12[k % 3] * angle_c / r12_n - r32[k % 3] / r32_n)
                    second =  num1 / (angle_s * r32_n) * (r32[k % 3] * angle_c / r32_n - r12[k % 3] / r12_n)
                    HH[k][i][j + b] = first + second

    for j in range(t):
        for k in range(3 * len(atoms)):
            for i in range(m):

                rij = dataset.coords[i, struct_descr.torsions[j][1]] - dataset.coords[i, struct_descr.torsions[j][0]]
                rjk = dataset.coords[i, struct_descr.torsions[j][2]] - dataset.coords[i, struct_descr.torsions[j][1]]
                rkl = dataset.coords[i, struct_descr.torsions[j][3]] - dataset.coords[i, struct_descr.torsions[j][2]]

                tors = dataset.torsions_matrix[i, j] / struct_descr.ns[j]
                tors_c = np.cos(tors)

                K = struct_descr.ns[j] * np.sin(tors * struct_descr.ns[j] + thetas[j + b + a]) * 1 / np.sin(tors)

                A = np.cross(rij, rjk)
                B = np.cross(rjk, rkl)
                dif = B - tors_c * A
                A_n = np.linalg.norm(A)
                B_n = np.linalg.norm(B)

                if k // 3 == struct_descr.torsions[j][0]:
                    HH[k][i][j + b + a] = K / A_n * (rjk[(k + 1) % 3] * dif[(k + 2) % 3] - rjk[(k + 2) % 3] * dif[(k + 1) % 3])

                if k // 3 == struct_descr.torsions[j][1]:
                    HH[k][i][j + b + a] = K * ( 1 / A_n * (rij[(k + 2) % 3] * dif[(k + 1) % 3] - rij[(k + 1) % 3] * dif[(k + 2) % 3]) +
                                                1 / B_n * (rkl[(k + 1) % 3] * dif[(k + 2) % 3] - rkl[(k + 2) % 3] * dif[(k + 1) % 3])) - \
                                          K / A_n * (rjk[(k + 1) % 3] * dif[(k + 2) % 3] - rjk[(k + 2) % 3] * dif[(k + 1) % 3])

                if k // 3 == struct_descr.torsions[j][2]:
                    HH[k][i][j + b + a] = - K / A_n * (rjk[(k + 2) % 3] * dif[(k + 1) % 3] - rjk[(k + 1) % 3] * dif[(k + 2) % 3]) - \
                                          K * (1 / A_n * (rij[(k + 2) % 3] * dif[(k + 1) % 3] - rij[(k + 1) % 3] * dif[(k + 2) % 3]) +
                                               1 / B_n * (rkl[(k + 1) % 3] * dif[(k + 2) % 3] - rkl[(k + 2) % 3] * dif[(k + 1) % 3]))

                if k // 3 == struct_descr.torsions[j][3]:
                    HH[k][i][j + b + a] = K / B_n * (rjk[(k + 2) % 3] * dif[(k + 1) % 3] - rjk[(k + 1) % 3] * dif[(k + 2) % 3])

    return HH
