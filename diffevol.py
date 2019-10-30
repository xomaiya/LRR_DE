import random
from ridgeRegression import *


class DE:
    def __init__(self, dataset, struct_description, y):
        self.dataset = dataset
        self.y = y
        self.struct_description = struct_description

    def f(self, x):
        l = x[0]
        thetas = x[1:]
        err = RR_LOOCV(self.dataset, self.y, l, thetas)
        # print(err)
        return err

    def f_for_population(self, P):
        b = len(self.struct_description[0])
        a = len(self.struct_description[1])
        t = len(self.struct_description[2])
        p = len(self.struct_description[3])
        # ограничение для длин связей (> 0)
        P[:, 0:b] = np.abs(P[:, 0:b])
        # ограничение для зарядов (от -5 до 5)
        P[:, b + a + t + p:b + a + t + p*2] = np.clip(P[:, b + a + t + p:b + a + t + p*2], -5, 5)
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

        P = np.random.randn(N, k)
        fp = self.f_for_population(P)
        while True:
            V = self.mutation(P, F)
            U = self.crossover(V, P, Cr)
            P, fp = self.selection(P, fp, U)
            self.best_p = P[np.argmin(fp)]
            print(np.min(fp))
