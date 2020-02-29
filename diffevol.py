import random
import time
import numpy as np
import scipy.sparse

from ridgeRegression import RidgeRegression, LOOCV
from moleculToVector import StructDescription, AmberCoefficients
from xyz2bat import xyz2bat2constr_H_map, xyz2bat2constr_HH_map
import matplotlib.pyplot as plt
from fast_ridge_regression import FastJaxRidgeRegression


class DE:
    def __init__(self, all_coords, struct_description: StructDescription, amber_coeffs: AmberCoefficients, y,
                 test_structs):
        self.all_coords = all_coords
        self.y = y
        self.struct_description = struct_description
        self.amber_coeffs = amber_coeffs
        self.test_structs = test_structs

        self.fjrr = FastJaxRidgeRegression()
        self.forces = np.array(y.reshape(-1, 57, 3))

    def genes2thetas(self, x):
        l_bonds = len(self.amber_coeffs.bonds_zero_values)
        l_angles = len(self.amber_coeffs.angles_zero_values)
        l_torsions = len(self.amber_coeffs.torsions_zero_phase)
        l_q = len(self.amber_coeffs.qs)
        l_sigma = len(self.amber_coeffs.sigma_for_vdw)
        l_epsilon = len(self.amber_coeffs.epsilons_for_vdw)

        l, x = x[0], x[1:]
        bonds, x = x[:l_bonds], x[l_bonds:]
        angles, x = x[:l_angles], x[l_angles:]
        torsions, x = x[:l_torsions], x[l_torsions:]
        q, x = x[:l_q], x[l_q:]
        sigma_for_vdw, x = x[:l_sigma], x[l_sigma:]
        epsilon_for_vdw, x = x[:l_epsilon], x[l_epsilon:]

        assert len(x) == 0

        thetas = {'bonds': np.abs(bonds),
                  'angles': angles,
                  'torsions': torsions,
                  'q': q,
                  'sigma_for_vdw': np.abs(sigma_for_vdw),
                  'epsilon_for_vdw': np.abs(epsilon_for_vdw)}
        return l, thetas

    def f(self, x):
        start_time = time.time()
        l, thetas = self.genes2thetas(x)

        HH = xyz2bat2constr_HH_map(self.all_coords, self.struct_description.as_dict(), thetas)
        # HH = HH.reshape(-1, HH.shape[-1])
        # HH = scipy.sparse.csr_matrix(HH)
        # _, y_est = RidgeRegression(HH, self.y, l)
        # err = LOOCV(HH, self.y, y_est)

        C, y_est, err = self.fjrr.calculate(HH, self.forces, l)

        print(f'err: {err}')
        print(f'time: {time.time() - start_time}')

        return err

    def f_for_population(self, P):
        b = len(self.struct_description.bonds)
        a = len(self.struct_description.angles)
        t = len(self.struct_description.torsions)
        p = len(self.struct_description.pairs)

        # TODO: считывать теты из файла и варьировать их относительно считанных: |delta bond_0| < 0.5 A,
        #  |delta angle_0| < 15 grad, |q| < 0.5 Col and sum(q) < 1, |sigma| < 0.8 A

        # ограничение для длин связей (только положительные)
        P[:, 0:b] = np.abs(P[:, 0:b])
        # ограничение для зарядов (от -0.5 до 0.5)
        P[:, b + a + t + p:b + a + t + p * 2] = np.clip(P[:, b + a + t + p:b + a + t + p * 2], -0.5, 0.5)

        return np.array([self.f(p) for p in P])

    def mutation(self, P, F):
        V = np.zeros_like(P)
        N = P.shape[0]
        for i in range(N):
            p = random.randint(0, N - 1)
            q = random.randint(0, N - 1)
            r = random.randint(0, N - 1)
            V[i] = P[p] + F * (P[q] - P[r])
        return V

    def crossover(self, V, P, Cr):
        U = np.zeros_like(V)
        N = P.shape[0]
        k = P.shape[1]
        j_rand = np.random.randint(0, k - 1)
        for i in range(N):
            for j in range(k):
                r = np.random.randn()
                if r <= Cr or j == j_rand:
                    U[i, j] = V[i, j]
                else:
                    U[i, j] = P[i, j]
        return U

    def selection(self, P, fp, U):
        fu = self.f_for_population(U)
        to_replace = fp > fu
        P[to_replace] = U[to_replace]
        fp[to_replace] = fu[to_replace]

        return P, fp

    def run(self, k, N, F=0.7, Cr=0.85):
        """
        Запуск алгоритма дифференциальной эволюции
        :param k: количество элементов в векторе пробного решения (количество оптимизируемых гиперпараметров: [l, thetas])
        :param N: размер популяции (количество векторов пробного решения)
        :param F: дифференциальный вес (вероятность мутации донорного вектора)
        :param Cr:  коэффициент кроссовера (скорость кроссовера)
        :return:
        """

        P = self.amber_coeffs.get_theta() + 0.001 * np.random.randn(N, k)
        fp = self.f_for_population(P)
        while True:
            V = self.mutation(P, F)
            U = self.crossover(V, P, Cr)
            P, fp = self.selection(P, fp, U)
            self.best_p = P[np.argmin(fp)]
            # self.test_for_best_p()
            # print(np.min(fp))

    def test(self, C, thetas):
        """
        Функция, которая рассчитывает корреляцию между полученными энергиями для тестовых структур и QM-энергиями
        :return:
        """

        energy_test_qm = np.array([struct.energy for struct in self.test_structs])
        forces_test_qm = np.array([struct.forces for struct in self.test_structs])

        test_all_coords = np.array([struct.coords for struct in self.test_structs])
        H = xyz2bat2constr_H_map(test_all_coords, self.struct_description.as_dict(), thetas)
        HH = xyz2bat2constr_HH_map(test_all_coords, self.struct_description.as_dict(), thetas)

        energy_test_mm = H.dot(C)
        forces_test_mm = HH.dot(C)

        print(f'forces error for test train:\t{((forces_test_qm - forces_test_mm) ** 2).sum(axis=(1, 2)).mean()}')
        print(f'energy correlation for test train:\t{np.corrcoef(energy_test_qm, energy_test_mm)[0][1]}')
        plt.scatter(energy_test_qm, energy_test_mm)
        plt.show()
        print()
        print()

    def test_for_best_p(self):
        l, thetas = self.genes2thetas(self.best_p)
        HH = xyz2bat2constr_HH_map(self.all_coords, self.struct_description.as_dict(), thetas)
        C, energy_est = RidgeRegression(scipy.sparse.csr_matrix(HH.reshape(-1, HH.shape[-1])), self.y, l)

        predicted = HH.dot(C)
        true = self.y.reshape(predicted.shape)
        print(f'forces error for train train:\t{((predicted - true) ** 2).sum(axis=(1, 2)).mean()}')

        self.test(C, thetas)
