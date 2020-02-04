from jax import numpy as np
from moleculToVector import StructDescription
from jax import vmap, jacfwd


def calc_bonds(xyz: np.ndarray, struct_description: StructDescription):
    return np.linalg.norm(xyz[struct_description.bonds[:, 0]] - xyz[struct_description.bonds[:, 1]], axis=1)


def calc_angles(xyz: np.ndarray, struct_description: StructDescription):
    r_ij = xyz[struct_description.angles[:, 1]] - xyz[struct_description.angles[:, 0]]
    r_kj = xyz[struct_description.angles[:, 1]] - xyz[struct_description.angles[:, 2]]
    cos = np.sum(r_ij * r_kj, axis=1) / (np.linalg.norm(r_ij, axis=1) * np.linalg.norm(r_kj, axis=1))
    return np.arccos(np.clip(cos, -1, 1))


def calc_torsions(xyz: np.ndarray, struct_description: StructDescription):
    r_ij = xyz[struct_description.torsions[:, 1]] - xyz[struct_description.torsions[:, 0]]
    r_jk = xyz[struct_description.torsions[:, 2]] - xyz[struct_description.torsions[:, 1]]
    r_kl = xyz[struct_description.torsions[:, 3]] - xyz[struct_description.torsions[:, 2]]
    A = np.cross(r_ij, r_jk)
    B = np.cross(r_jk, r_kl)
    cos = np.sum(A * B, axis=1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))
    return struct_description.ns * np.arccos(np.clip(cos, -1, 1))


def calc_pairs(xyz: np.ndarray, struct_description: StructDescription):
    return np.linalg.norm(xyz[struct_description.pairs[:, 0]] - xyz[struct_description.pairs[:, 1]], axis=1)


def calc_qq(thetas, struct_description: StructDescription):
    return thetas['q'][struct_description.pairs[:, 0]] * thetas['q'][struct_description.pairs[:, 1]]


def calc_A_B_for_vdw(thetas, struct_description: StructDescription):
    sigma_ij = (thetas['sigma_for_vdw'][struct_description.pairs[:, 0]] + thetas['sigma_for_vdw'][struct_description.pairs[:, 1]]) / 2
    epsilon_ij = np.sqrt(thetas['epsilon_for_vdw'][struct_description.pairs[:, 0]] * thetas['epsilon_for_vdw'][struct_description.pairs[:, 1]])
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
    torsions_part = (1 + np.cos(bat['torsions'] + thetas['torsions']))
    A, B = calc_A_B_for_vdw(thetas, struct_description)
    vdw_part = A / bat['pairs'] ** 12 - B / bat['pairs'] ** 6
    coulomb_part = calc_qq(thetas, struct_description) / bat['pairs']
    H = np.concatenate([bonds_part, angles_part, torsions_part, vdw_part, coulomb_part])
    return H


def xyz2bat2constr_H(coords, struct_description, thetas):
    bat = xyz2bat(coords, struct_description)
    return constr_H(bat, struct_description, thetas)


xyz2bat2constr_H_map = vmap(xyz2bat2constr_H, (0, None, None))


def xyz2bat2constr_HH(coords, struct_description, thetas):
    constr_HH = jacfwd(xyz2bat2constr_H)
    return np.transpose(constr_HH(coords, struct_description, thetas), (1, 2, 0))


xyz2bat2constr_HH_map = vmap(xyz2bat2constr_HH, (0, None, None))

