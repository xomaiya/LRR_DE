import jax
from jax import numpy as np
from tqdm.autonotebook import trange, tqdm
import numpy as onp


class FastJaxRidgeRegression:
    def __init__(self):
        self.inited = False

    def init(self, HH_map):
        if self.inited:
            assert True
        else:
            self.inited = True

            nonzeros = []
            for i in trange(HH_map.shape[-1], desc='constructing nonzero masks'):
                flag = HH_map[..., i] != 0
                assert np.all(flag == flag[None, 0, ...])
                assert np.all(flag[0] == flag[0, :, 0, None])
                nonzeros.append(np.any(flag, axis=(0, 2)))

            ij, intersections = [], []
            for i in trange(len(nonzeros), desc='constructing intersections'):
                for j in range(i + 1):
                    ij.append((i, j))
                    intersections.append(onp.where(np.logical_and(nonzeros[i], nonzeros[j]))[0])

            descriptors = {}
            for (i, j), intr in zip(ij, intersections):
                descriptors.setdefault(len(intr), []).append(((i, j), intr))

            self.constructed_descrs = []
            for cnt, descr in tqdm(descriptors.items(), desc='constructing descriptors'):
                all_i = np.array([i for (i, _), _ in descr])
                all_j = np.array([j for (_, j), _ in descr])
                all_inds = np.array([inds for _, inds in descr])

                self.constructed_descrs.append((all_i, all_j, all_inds))

            self.method = jax.jit(
                lambda HH_map, forces, l: self.run_ridge_regression(HH_map, forces, l, self.constructed_descrs))

    @staticmethod
    def get_matrix_sqr(HH_map, descrs):
        P = np.zeros((HH_map.shape[-1], HH_map.shape[-1]))

        for all_i, all_j, all_inds in descrs:
            i_part = HH_map[:, all_inds, :, all_i[:, None]]
            j_part = HH_map[:, all_inds, :, all_j[:, None]]
            ij_vals = (i_part * j_part).sum(axis=(1, 2, 3))
            P = jax.ops.index_update(P, jax.ops.index[all_i, all_j], ij_vals)
            P = jax.ops.index_update(P, jax.ops.index[all_j, all_i], ij_vals)

        return P

    @staticmethod
    def run_ridge_regression(HH_map, forces, l, descr):
        A = FastJaxRidgeRegression.get_matrix_sqr(HH_map, descr) + 2 * l * np.identity(HH_map.shape[-1])
        B = (HH_map * forces[..., None]).sum(axis=(0, 1, 2))
        C = np.linalg.inv(A).dot(B)

        y_est = HH_map.dot(C)

        M = np.prod(HH_map.shape[:-1])
        varH = np.linalg.norm(HH_map - HH_map.mean(axis=(0, 1, 2)), axis=-1) ** 2
        h = 1 / M + varH / varH.sum()
        loocv = 1 / M * (((forces - y_est) / (1 - h)) ** 2).sum()

        return C, y_est, loocv

    def calculate(self, HH_map, forces, l):
        self.init(HH_map)
        return self.method(HH_map, forces, l)
