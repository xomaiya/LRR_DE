from jax import numpy as np
from moleculToVector import StructDescription
from jax import vmap, jacfwd, jit, lax


def calc_bonds(xyz: np.ndarray, struct_descr_dict):
    return np.linalg.norm(xyz[struct_descr_dict['bonds'][:, 0]] - xyz[struct_descr_dict['bonds'][:, 1]], axis=1)


def calc_angles(xyz: np.ndarray, struct_descr_dict):
    r_ij = xyz[struct_descr_dict['angles'][:, 1]] - xyz[struct_descr_dict['angles'][:, 0]]
    r_kj = xyz[struct_descr_dict['angles'][:, 1]] - xyz[struct_descr_dict['angles'][:, 2]]
    cos = np.sum(r_ij * r_kj, axis=1) / (np.linalg.norm(r_ij, axis=1) * np.linalg.norm(r_kj, axis=1))
    return np.arccos(np.clip(cos, -1, 1))


def calc_torsions(xyz: np.ndarray, struct_descr_dict):
    b1 = xyz[struct_descr_dict['torsions'][:, 1]] - xyz[struct_descr_dict['torsions'][:, 0]]
    b2 = xyz[struct_descr_dict['torsions'][:, 2]] - xyz[struct_descr_dict['torsions'][:, 1]]
    b3 = xyz[struct_descr_dict['torsions'][:, 3]] - xyz[struct_descr_dict['torsions'][:, 2]]

    b12 = np.cross(b1, b2)
    b23 = np.cross(b2, b3)
    return np.arctan2((np.cross(b12, b23) * b2).sum(axis=-1) / np.linalg.norm(b2, axis=-1), (b12 * b23).sum(axis=-1))


def calc_pairs(xyz: np.ndarray, struct_descr_dict):
    return np.linalg.norm(xyz[struct_descr_dict['pairs'][:, 0]] - xyz[struct_descr_dict['pairs'][:, 1]], axis=1)


def calc_qq(thetas, struct_descr_dict):
    return thetas['q'][struct_descr_dict['pairs'][:, 0]] * thetas['q'][struct_descr_dict['pairs'][:, 1]]


def calc_A_B_for_vdw(thetas, struct_descr_dict):
    sigma_ij = (thetas['sigma_for_vdw'][struct_descr_dict['pairs'][:, 0]] + thetas['sigma_for_vdw'][struct_descr_dict['pairs'][:, 1]]) / 2
    epsilon_ij = np.sqrt(thetas['epsilon_for_vdw'][struct_descr_dict['pairs'][:, 0]] * thetas['epsilon_for_vdw'][struct_descr_dict['pairs'][:, 1]])
    A = 4 * sigma_ij ** 12 * epsilon_ij
    B = 4 * sigma_ij ** 6 * epsilon_ij
    return A, B


def xyz2bat(xyz: np.ndarray, struct_description):
    bonds = calc_bonds(xyz, struct_description)
    angles = calc_angles(xyz, struct_description)
    torsions = calc_torsions(xyz, struct_description)
    pairs = calc_pairs(xyz, struct_description)
    return {'bonds': bonds, 'angles': angles, 'torsions': torsions, 'pairs': pairs}


def constr_H(bat, struct_description, thetas):
    bonds_part = (bat['bonds'] - thetas['bonds']) ** 2
    angles_part = (bat['angles'] - thetas['angles']) ** 2
    torsions_part = 1 + np.cos(bat['torsions'] + thetas['torsions'])
    A, B = calc_A_B_for_vdw(thetas, struct_description)
    vdw_part = A / bat['pairs'] ** 12 - B / bat['pairs'] ** 6
    coulomb_part = calc_qq(thetas, struct_description) / bat['pairs']
    H = np.concatenate([bonds_part, angles_part, torsions_part, vdw_part.sum(keepdims=True), coulomb_part.sum(keepdims=True)])
    return H


def xyz2bat2constr_H(coords, struct_description, thetas):
    bat = xyz2bat(coords, struct_description)
    return constr_H(bat, struct_description, thetas)


xyz2bat2constr_H_map = vmap(xyz2bat2constr_H, (0, None, None))


def xyz2bat2constr_HH(coords, struct_description, thetas):
    constr_HH = jacfwd(xyz2bat2constr_H)
    return np.transpose(constr_HH(coords, struct_description, thetas), (1, 2, 0))


xyz2bat2constr_HH_map = jit(vmap(xyz2bat2constr_HH, (0, None, None)))
