import openbabel as ob
from tqdm.autonotebook import tqdm
import numpy as np

class StructXYZ:
    def __init__(self, coords, atoms, energy, forces):
        self.coords = coords
        self.atoms = atoms
        self.energy = energy
        self.forces = forces


def read_dataset(path='test-Olesya/sample_coords.xyz', path_to_forces=None):
    """
    Считывание из файла xyz структур в виде структур StructXYZ. Энергия в ккал.

    :param path: путь до файла со структурами
    :return: list(StructXYZ)
    """

    structs = []
    with open(path, 'r') as file:
        lines = file.readlines()

    if path_to_forces is not None:
        with open(path_to_forces, 'r') as file:
            force_line = file.readlines()

    a = tqdm()

    pointer = 0  # Сдвиг указателя на начало списка, чтобы не перекопировать его
    pointer_force = 4
    while True:
        a.update(1)

        N = int(lines[0 + pointer])
        E = 630 * float(lines[1 + pointer].split()[8])
        coords = np.zeros((N, 3))
        for i in range(N):
            coords[i, :] = [float(j) for j in lines[i + 2 + pointer].split()[1:4]]
        atoms = []
        for i in range(N):
            atoms.append(lines[i + 2 + pointer].split()[0])

        if path_to_forces is not None:
            forces = np.zeros((N, 3))
            for i in range(N):
                forces[i, :] = [float(j) for j in force_line[i + pointer_force].split()[3:6]]
        else:
            forces = None

        dict_atoms = {'C': 6, 'H': 1, 'N': 7, 'O': 8}
        atoms = [dict_atoms[atom] for atom in atoms]
        struct = StructXYZ(coords, atoms, E, forces)
        structs.append(struct)

        pointer += N + 2
        pointer_force += N + 5
        if len(lines) - pointer <= 1:
            break

    return structs


class StructDescription:
    def __init__(self, bonds, angles, torsions, ns, pairs, atoms):
        self.bonds = bonds
        self.angles = angles
        self.torsions = torsions
        self.ns = ns
        self.pairs = pairs
        self.atoms = atoms


class AmberCoefficients:
    def __init__(self, bonds_linear_coeffs, bonds_zero_values, angles_linear_coeffs, angles_zero_values,
                 torsions_linear_coeffs, torsions_zero_phase, sigma_for_vdw, epsilons_for_vdw, qs, qq_coeffs, ns):
        self.bonds_linear_coeffs = bonds_linear_coeffs
        self.bonds_zero_values = bonds_zero_values
        self.angles_linear_coeffs = angles_linear_coeffs
        self.angles_zero_values = angles_zero_values
        self.torsions_linear_coeffs = torsions_linear_coeffs
        self.torsions_zero_phase = torsions_zero_phase
        self.sigma_for_vdw = sigma_for_vdw
        self.epsilons_for_vdw = epsilons_for_vdw
        self.qs = qs
        self.qq_coeffs = qq_coeffs
        self.ns = ns



def get_struct_description(path='test-Olesya/Initial_parameters_with_numbers_and_dihedrals_only.txt'):
    """
    Считывание топологии молекулы (между какими атомами есть связи, углы, торсионные углы,
    а также записывается список всех пар атомов)
    Вместе с этим, считываются проверочные значения для линейных и нелинейных параметров в виде объекта класса
    AmberCoefficients.

    :param path: путь к файлу с описанием топологии молекулы
    :return: описание топологии в виде tuple(list)
    """

    with open(path, 'r') as file:
        lines = file.readlines()

    splt = []
    for i in range(len(lines)):
        if len(lines[i].strip()) == 0:
            splt.append(i)
    splt.append(len(lines))

    bonds = []
    bonds_linear_coeffs = []
    bonds_zero_values = []
    for i in range(1, splt[0]):
        bonds.append([int(j) - 1 for j in lines[i].split()[0:2]])
        bonds_linear_coeffs.append(float(lines[i].split()[2]))
        bonds_zero_values.append(float(lines[i].split()[3]))

    angles = []
    angles_linear_coeffs = []
    angles_zero_values = []
    for i in range(splt[0] + 2, splt[1]):
        angles.append([int(j) - 1 for j in lines[i].split()[0:3]])
        angles_linear_coeffs.append(float(lines[i].split()[3]))
        angles_zero_values.append(float(lines[i].split()[4]))
    angles_zero_values = [angle * 0.0174533 for angle in angles_zero_values]

    torsions = []
    torsions_linear_coeffs = []
    torsions_zero_phase = []
    ns = []
    for i in range(splt[1] + 2, splt[2]):
        torsions.append([int(j) - 1 for j in lines[i].split()[0:4]])
        torsions_linear_coeffs.append(float(lines[i].split()[4]))
        torsions_zero_phase.append(float(lines[i].split()[5]))
        ns.append(int(float(lines[i].split()[6])))
    torsions_zero_phase = [phase * 0.0174533 for phase in torsions_zero_phase]

    atoms = []
    rmin_single = []
    epsilon_single = []
    for i in range(splt[2] + 2, splt[3]):
        atoms.append(int(lines[i].split()[0]) - 1)
        rmin_single.append(float(lines[i].split()[1]))
        epsilon_single.append(float(lines[i].split()[2]))

    pairs_from_angles = [[i, k] for i, j, k in angles]
    pairs = []
    for i in range(0, len(atoms)):
        for j in range(0, i):
            if [i, j] not in bonds and [j, i] not in bonds \
                    and [i, j] not in pairs_from_angles and [j, i] not in pairs_from_angles:
                pairs.append([i, j])

    eps14_for_coloumb = 0.83333333333 * 332.0636
    qq_coeffs = []
    for i, j in pairs:
        if [None for (a, _, _,b) in torsions if a == i and b == j or a == j and b == i]:
             qq_coeffs.append(eps14_for_coloumb)
        else:
            qq_coeffs.append(1 * 332.0636)

    # epsilons_for_vdw = []  # массив значений глубины ямы для сил Ван-дер-Ваальса
    # for i, j in pairs:
    #     epsilons_for_vdw.append(np.sqrt(epsilon_single[i] * epsilon_single[j]))

    # sigma_for_vdw = []  # массив парных Rmin -- нелинейные коэффициенты для сил Ван-дер-Ваальса (сигмы)
    # for i, j in pairs:
    #     sigma_for_vdw.append(0.5 * (rmin_single[i] + rmin_single[j]))

    qs = [0.538850, -0.140384, 0.037195, 0.037195, -0.005530, 0.031484, 0.031484, -0.184033, 0.069999, 0.069999,
          0.273603, -0.404638, 0.137648, 0.111164, -0.390786, 0.185296, 0.331572, -0.433865, 0.189131, 0.186116,
          0.138703, -0.501163, 0.219208, 0.485403, -0.584667, 0.208140, 0.255950, 0.200007, -0.339941, -0.471452,
          0.109898, 0.109898, 0.109898, -0.471452, 0.109898, 0.109898, 0.109898, - 0.429466, 0.118775, 0.118775,
          0.118775, -0.296884, 0.103144, 0.103144, 0.103144, -0.300885, 0.103898, 0.103898, 0.103898, -0.209174,
          0.104587, 0.104587, 0.104587, -0.209174, 0.104587, 0.104587, 0.104587]

    struct_description = StructDescription(bonds=np.array(bonds), angles=np.array(angles), torsions=np.array(torsions),
                                           ns=np.array(ns), pairs=np.array(pairs), atoms=np.array(atoms))

    amber_coefficients = AmberCoefficients(bonds_linear_coeffs=np.array(bonds_linear_coeffs),
                                           bonds_zero_values=np.array(bonds_zero_values),
                                           angles_linear_coeffs=np.array(angles_linear_coeffs),
                                           angles_zero_values=np.array(angles_zero_values),
                                           torsions_linear_coeffs=np.array(torsions_linear_coeffs),
                                           torsions_zero_phase=np.array(torsions_zero_phase),
                                           sigma_for_vdw=np.array(rmin_single),
                                           epsilons_for_vdw=np.array(epsilon_single),
                                           qs=np.array(qs),
                                           qq_coeffs=np.array(qq_coeffs),
                                           ns=np.array(ns))

    return struct_description, amber_coefficients


def get_el_for_dataset(struct, struct_description):
    """
    Генератор отдельных объектов датасета

    :param struct: структура StructXYZ
    :param struct_description: объект класса StructDescription с полями: bonds, angles, ns, torsions, pairs
    :return: объект датасета типа tuple, содержащий списки значений связей, углов, торсионных углов
    и попарных расстояний в виде списоков ([bonds_val], [angles_val], [n * torsions_val], [pairs_length_val])
    в ангстремах и радианах
    """

    obMol = ob.OBMol()

    for atom, xyz in zip(struct.atoms, struct.coords):
        obAtom = ob.OBAtom()
        obAtom.SetVector(*xyz)
        obAtom.SetAtomicNum(atom)
        obMol.AddAtom(obAtom)

    x = ([], [], [], [])
    for bond in struct_description.bonds:
        x[0].append(obMol.GetAtom(int(bond[0]) + 1).GetDistance(obMol.GetAtom(int(bond[1]) + 1)))
    for angle in struct_description.angles:
        x[1].append(0.0174533 * obMol.GetAngle(obMol.GetAtom(int(angle[0]) + 1), obMol.GetAtom(int(angle[1]) + 1),
                                               obMol.GetAtom(int(angle[2]) + 1)))
    for n, torsion in zip(struct_description.ns, struct_description.torsions):
        x[2].append(0.0174533 * n * obMol.GetTorsion(obMol.GetAtom(int(torsion[0]) + 1), obMol.GetAtom(int(torsion[1]) + 1),
                                                     obMol.GetAtom(int(torsion[2]) + 1), obMol.GetAtom(int(torsion[3]) + 1)))
    for pair in struct_description.pairs:
        x[3].append(obMol.GetAtom(int(pair[0]) + 1).GetDistance(obMol.GetAtom(int(pair[1]) + 1)))

    return x


class DataSetMatrix:
    """
    Класс матриц датасета отдельно для связей, углов, торсионных углов и пар атомов
    """

    def __init__(self, bonds_matrix, angles_matrix, torsions_matrix, pairs_matrix, coords):
        self.bonds_matrix = bonds_matrix
        self.angles_matrix = angles_matrix
        self.torsions_matrix = torsions_matrix
        self.pairs_matrix = pairs_matrix
        self.coords = coords

    def __getitem__(self, item):
        return DataSetMatrix(
            self.bonds_matrix[item],
            self.angles_matrix[item],
            self.torsions_matrix[item],
            self.pairs_matrix[item],
            self.coords[item]
        )


def get_dataset(structs, struct_description):
    """
    Генератор датасета

    :param structs: список структур, содержащих coords, atoms, E и forces
    :param struct_description: tuple списков с описанием топологии структуры (bonds, angles, torsions, pairs)
    :return: list(tuple(list)) -- список входных векторов (типа tuple), содержащий списки значений
    связей, углов, торсионных углов и попарных расстояний
    """

    coords = np.zeros((len(structs), len(struct_description.atoms), 3))
    dataset = []
    for i, struct in enumerate(tqdm(structs)):
        dataset.append(get_el_for_dataset(struct, struct_description))
        coords[i] = struct.coords

    dataset_matrix = DataSetMatrix(bonds_matrix=np.array([el[0] for el in dataset]),
                                   angles_matrix=np.array([el[1] for el in dataset]),
                                   torsions_matrix=np.array([el[2] for el in dataset]),
                                   pairs_matrix=np.array([el[3] for el in dataset]),
                                   coords=coords)

    return dataset_matrix
