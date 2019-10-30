from diffevol import *
import openbabel as ob
from tqdm.autonotebook import tqdm


def read_dataset(path='test-Olesya/sample_coords.xyz'):
    """
    Считывание из файла xyz структур в виде (coords, atoms, E)

    :param path: путь до файла со структурами
    :return: list((coords, atoms, E))
    """

    structs = []
    with open(path, 'r') as file:
        lines = file.readlines()

    a = tqdm()

    pointer = 0 # Сдвиг указателя на начало списка, чтобы не перекопировать его
    while True:
        a.update(1)

        N = int(lines[0 + pointer])
        E = 2625.50 * float(lines[1 + pointer].split()[8])
        coords = np.zeros((N, 3))
        for i in range(N):
            coords[i, :] = [float(j) for j in lines[i + 2 + pointer].split()[1:4]]
        atoms = []
        for i in range(N):
            atoms.append(lines[i + 2 + pointer].split()[0])

        dict_atoms = {'C': 6, 'H': 1, 'N': 7, 'O': 8}
        atoms = [dict_atoms[atom] for atom in atoms]
        struct = (coords, atoms, E)
        structs.append(struct)

        pointer += N + 2
        if len(lines) - pointer <= 1:
            break
    return structs


def get_struct_description(path='test-Olesya/Initial_parameters_with_numbers_and_dihedrals_only.txt'):
    """
    Считывание топологии молекулы (между какими атомами есть связи, углы, торсионные углы,
    а также записывается список всех пар атомов)

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
    for i in range(1, splt[0]):
        bonds.append([int(j) for j in lines[i].split()[0:2]])

    angles = []
    for i in range(splt[0] + 2, splt[1]):
        angles.append([float(j) for j in lines[i].split()[0:3]])

    torsions = []
    ns = []
    for i in range(splt[1] + 2, splt[2]):
        torsions.append([float(j) for j in lines[i].split()[0:4]])
        ns.append(float(lines[i].split()[6]))

    atoms = []
    for i in range(splt[2] + 2, splt[3]):
        atoms.append(int(lines[i].split()[0]))

    pairs = []
    for i in range(1, len(atoms) + 1):
        for j in range(1, i):
            if [i, j] not in bonds and [j, i] not in bonds:
                pairs.append([i, j])
    assert len(bonds) + len(pairs) == len(atoms) * (len(atoms) - 1) / 2
    struct_description = (bonds, angles, torsions, ns, pairs, atoms)
    return struct_description


def get_el_for_dataset(struct, struct_description):
    """
    Генератор отдельных объектов датасета

    :param struct: структура в виде tuple с элементами-списками (coords, atoms, E)
    :param struct_description: tuple списков с описанием топологии структуры (bonds, angles, torsions, pairs)
    :return: объект датасета типа tuple, содержащий списки значений связей, углов, торсионных углов
    и попарных расстояний
    """

    obMol = ob.OBMol()
    coords, atoms, _ = struct
    bonds, angles, torsions, ns, pairs, atoms = struct_description

    for atom, xyz in zip(atoms, coords):
        obAtom = ob.OBAtom()
        obAtom.SetVector(*xyz)
        obAtom.SetAtomicNum(atom)
        obMol.AddAtom(obAtom)

    x = ([], [], [], [])
    for bond in bonds:
        x[0].append(obMol.GetAtom(int(bond[0])).GetDistance(obMol.GetAtom(int(bond[1]))))
    for angle in angles:
        x[1].append(obMol.GetAngle(obMol.GetAtom(int(angle[0])), obMol.GetAtom(int(angle[1])),
                                obMol.GetAtom(int(angle[2]))))
    for n, torsion in zip(ns, torsions):
        x[2].append(n * obMol.GetTorsion(obMol.GetAtom(int(torsion[0])), obMol.GetAtom(int(torsion[1])),
                                  obMol.GetAtom(int(torsion[2])), obMol.GetAtom(int(torsion[3]))))
    for pair in pairs:
        x[3].append(obMol.GetAtom(int(pair[0])).GetDistance(obMol.GetAtom(int(pair[1]))))

    return x


def get_dataset(structs, struct_description):
    """
    Генератор датасета

    :param structs: список структур в виде tuple с элементами-списками (coords, atoms, E)
    :param struct_description: tuple списков с описанием топологии структуры (bonds, angles, torsions, pairs)
    :return: list(tuple(list)) -- список входных векторов (типа tuple), содержащий списки значений
    связей, углов, торсионных углов и попарных расстояний
    """

    dataset = []
    for struct in tqdm(structs):
        dataset.append(get_el_for_dataset(struct, struct_description))

    dataset_matrix = (np.array([el[0] for el in dataset]),
                      np.array([el[1] for el in dataset]),
                      np.array([el[2] for el in dataset]),
                      np.array([el[3] for el in dataset]))

    return dataset_matrix, struct_description

