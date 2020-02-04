

def get_initial_thetas(path='test-Olesya/Initial_parameters.txt'):
    with open(path, 'r') as file:
        lines = file.readlines()

    splt = []
    for i in range(len(lines)):
        if len(lines[i].strip()) == 0:
            splt.append(i)
    splt.append(len(lines))

    thetas = []
    for i in range(1, splt[0]):
        thetas.append(float(lines[i].split()[3]))
    for i in range(splt[0] + 2, splt[1]):
        thetas.append(float(lines[i].split()[4]))
    for i in range(splt[1] + 2, splt[2]):
        thetas.append(float(lines[i].split()[5]))
    # тут запишем сигмы для вандервальсовых взаимодействий
    # тут запишем заряды для атомов
    # for i in range(splt[2] + 2, splt[3]):
    #     thetas.append(0)
    return thetas

def get_initial_linear_params(path='test-Olesya/Initial_parameters.txt'):

    with open(path, 'r') as file:
        lines = file.readlines()

    splt = []
    for i in range(len(lines)):
        if len(lines[i].strip()) == 0:
            splt.append(i)
    splt.append(len(lines))

    linear_params = []
    for i in range(1, splt[0]):
        linear_params.append(float(lines[i].split()[2]))
    for i in range(splt[0] + 2, splt[1]):
        linear_params.append(float(lines[i].split()[3]))
    for i in range(splt[1] + 2, splt[2]):
        linear_params.append(float(lines[i].split()[4]))

    # тут запишем сигмы для вандервальсовых взаимодействий
    for i in range(splt[2] + 2, splt[3]):
        linear_params.append(float(lines[i].split()[1]))
    # тут запишем заряды для атомов
    # for i in range(splt[2] + 2, splt[3]):
    #     thetas.append(0)

    return linear_params