# from matlab
import random
import time
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

from algorithms.AntLionAlgorithm import UAV_SENSOR_NUM, objective_function_value

MAX = 1
MIN = 0
SOLUTION_TEMPLATE = [0 for _ in range(UAV_SENSOR_NUM)]
MIN_VALUE_TEMPLATE = np.zeros(UAV_SENSOR_NUM)
MAX_VALUE_TEMPLATE = np.ones(UAV_SENSOR_NUM)
OBJECTIVE_NUM = 2
ARCHIVE_SIZE = 100
DIM_SIZE = UAV_SENSOR_NUM


def initialization(SearchAgents_no, dim, ub, lb):
    Boundary_no = ub.size

    if Boundary_no == 1:
        ub_new = np.ones(dim) * ub
        lb_new = np.ones(dim) * lb
    else:
        ub_new = ub
        lb_new = lb

    Positions = np.zeros((SearchAgents_no, dim))
    for i in range(dim):
        ub_i = ub_new[i]
        lb_i = lb_new[i]
        Positions[:, i] = np.random.rand(SearchAgents_no) * (ub_i - lb_i) + lb_i

    return Positions


def zd1(x):
    # ZD1
    dim = len(x)
    g = 1 + 9 * sum(x[2:dim]) / (dim - 1)
    # o1 = round(x[0], 4)
    # o2 = round(g * (1 - np.sqrt(x[0] / g)), 4)
    o1 = x[0]
    o2 = g * (1 - np.sqrt(x[0] / g))

    return o1, o2


def dominates(x, y):
    if np.all(x <= y):
        if np.any(x < y):
            return True

    return False


# def update_archive(Archive_X, Archive_F, Particles_X, Particles_F, Archive_member_no):
#     Archive_X_temp = np.vstack((Archive_X, Particles_X))
#     Archive_F_temp = np.vstack((Archive_F, Particles_F))
#
#     non_dominated_indices = []
#     for i in range(Archive_F_temp.shape[0]):
#         dominated = False
#         for j in range(Archive_F_temp.shape[0]):
#             if i != j and dominates(Archive_F_temp[j], Archive_F_temp[i]):
#                 dominated = True
#                 break
#         if not dominated:
#             non_dominated_indices.append(i)
#
#     Archive_X_updated = Archive_X_temp[non_dominated_indices]
#     Archive_F_updated = Archive_F_temp[non_dominated_indices]
#     Archive_member_no = len(non_dominated_indices)
#
#     return Archive_X_updated, Archive_F_updated, Archive_member_no


def update_archive(Archive_X, Archive_F, Particles_X, Particles_F, Archive_member_no):
    Archive_X_temp = np.vstack((Archive_X, Particles_X))
    Archive_F_temp = np.vstack((Archive_F, Particles_F))
    # o = np.zeros((1, Archive_F_temp.shape[0]))
    o = np.zeros(Archive_F_temp.shape[0], dtype=int)

    for i in range(Archive_F_temp.shape[0]):
        o[i] = 0
        for j in range(i):
            if np.any(Archive_F_temp[i, :] != Archive_F_temp[j, :]):
                if dominates(Archive_F_temp[i, :], Archive_F_temp[j, :]):
                    o[j] = 1
                elif dominates(Archive_F_temp[j, :], Archive_F_temp[i, :]):
                    o[i] = 1
                    break
            else:
                o[j] = 1
                o[i] = 1

    Archive_member_no = 0
    Archive_X_updated = list()
    Archive_F_updated = list()
    # Archive_X_updated = np.ones((1, Archive_X_temp.shape[1]))
    # Archive_F_updated = np.ones((1, Archive_F_temp.shape[1]))
    for i in range(Archive_X_temp.shape[0]):
        if o[i] == 0:
            Archive_X_updated.append(Archive_X_temp[i, :])
            Archive_F_updated.append(Archive_F_temp[i, :])
            # Archive_X_updated = np.append(Archive_X_updated, Archive_X_temp[i, :], axis=0)
            # Archive_F_updated = np.append(Archive_F_updated, Archive_F_temp[i, :], axis=0)
            Archive_member_no += 1
    Archive_X_updated_new = np.array(Archive_X_updated)
    Archive_F_updated_new = np.array(Archive_F_updated)

    return Archive_X_updated_new, Archive_F_updated_new, Archive_member_no


my_min, my_max = None, None


def ranking_process(archive_function, archive_max_size, obj_no=OBJECTIVE_NUM):
    global my_min, my_max
    if archive_function.shape == (1, 2):
        my_min = archive_function
        my_max = archive_function
    else:
        my_min = np.min(archive_function, axis=0)
        my_max = np.max(archive_function, axis=0)

    r = (my_max - my_min) / 20
    ranks = np.zeros(archive_function.shape[0])

    for i in range(archive_function.shape[0]):
        ranks[i] = np.sum(np.all(np.abs(archive_function - archive_function[i]) < r, axis=1))
    return ranks


# def ranking_process(archive_function, archive_max_size, obj_no=OBJECTIVE_NUM):
#     global my_min, my_max
#     if archive_function.shape == (1, 2):
#         my_min = archive_function
#         my_max = archive_function
#     else:
#         my_min = np.min(archive_function, axis=0)
#         my_max = np.max(archive_function, axis=0)
#
#     r = (my_max - my_min) / 20
#     ranks = np.zeros(archive_function.shape[0])
#
#     for i in range(archive_function.shape[0]):
#         ranks[i] = 0
#         for j in range(archive_function.shape[0]):
#             flag = 0
#             for k in range(obj_no):
#                 if np.all(abs(archive_function[j, k] - archive_function[i, k]) < r[k]):
#                     flag += 1
#             if flag == obj_no:
#                 ranks[i] += 1
#     return ranks


def handle_full_archive(Archive_X, Archive_F, Archive_member_no, Archive_mem_ranks, ArchiveMaxSize):
    for i in range(Archive_X.shape[0] - ArchiveMaxSize):
        index = roulette_wheel_selection(Archive_mem_ranks)

        Archive_X = np.vstack((Archive_X[index + 1:], Archive_X[:index + 1]))
        Archive_F = np.vstack((Archive_F[index + 1:], Archive_F[:index + 1]))
        # Archive_mem_ranks = np.concatenate((Archive_mem_ranks[:index], Archive_mem_ranks[index+1:Archive_member_no]))
        Archive_mem_ranks = np.delete(Archive_mem_ranks, index)
        Archive_member_no -= 1

    Archive_X_Chopped = Archive_X
    Archive_F_Chopped = Archive_F
    Archive_mem_ranks_updated = Archive_mem_ranks
    return Archive_X_Chopped, Archive_F_Chopped, Archive_mem_ranks_updated, Archive_member_no


def roulette_wheel_selection(weights):
    accumulation = np.cumsum(weights)
    p = np.random.random() * accumulation[-1]
    choice = np.searchsorted(accumulation, p)
    return choice


# def roulette_wheel_selection(weights):
#     accumulation = np.cumsum(weights)
#     p = np.random.random() * accumulation[-1]
#     chosen_index = -1
#     for index in range(len(accumulation)):
#         if accumulation[index] > p:
#             chosen_index = index
#             break
#     choice = chosen_index
#     return choice


def random_walk_around_antlion(dim, max_iter, lb, ub, antlion, current_iter):
    # if lb.shape == (1, 1):
    #     lb = np.ones(dim) * lb
    #     ub = np.ones(dim) * ub
    # if lb.shape[0] > lb.shape[1]:
    #     lb = lb.T
    #     ub = ub.T
    I = 1
    if current_iter > max_iter / 10:
        I = 1 + 100 * (current_iter / max_iter)
    elif current_iter < max_iter / 2:
        I = 1 + 1000 * (current_iter / max_iter)
    elif current_iter > max_iter * 0.75:
        I = 1 + 10000 * (current_iter / max_iter)
    elif current_iter > max_iter * 0.9:
        I = 1 + 100000 * (current_iter / max_iter)
    elif current_iter > max_iter * 0.95:
        I = 1 + 1000000 * (current_iter / max_iter)

    lb_new = lb / I
    ub_new = ub / I

    if np.random.rand() < 0.5:
        lb_new = lb_new + antlion
    else:
        lb_new = -lb_new + antlion

    if np.random.rand() >= 0.5:
        ub_new = ub_new + antlion
    else:
        ub_new = -ub_new + antlion

    RWs = np.zeros((max_iter + 1, dim))
    for i in range(dim):
        X = np.concatenate(([0], np.cumsum(2 * (np.random.rand(max_iter) > 0.5) - 1))).reshape(-1, 1)

        a = np.min(X)
        b = np.max(X)

        c = lb_new[i]
        d = ub_new[i]

        X_norm = ((X - a) * (d - c)) / (b - a) + c
        RWs[:, i] = X_norm.flatten()

    return RWs


def moalo(dim=DIM_SIZE, lb=MIN_VALUE_TEMPLATE, ub=MAX_VALUE_TEMPLATE, obj_no=OBJECTIVE_NUM, max_iter=3000,
          colony_size=100, targetfunction=zd1):
    archive_x = np.zeros((ARCHIVE_SIZE, dim))
    archive_f = np.ones((ARCHIVE_SIZE, obj_no)) * np.inf
    archive_member_no = 0

    # r = (ub - lb) / 2
    # V_max = (ub[0] - lb[0]) / 10

    elite_fitness = np.ones(obj_no) * np.inf
    # elite_position = np.zeros(dim)

    ant_position = initialization(colony_size, dim, ub, lb)
    # fitness = np.zeros((colony_size, 2))

    # v = initialization(colony_size, dim, ub, lb)
    # iter = 0

    # position_history = np.zeros((max_iter, colony_size, dim))

    for iter in range(max_iter):
        start_time = time.time()
        particles_f = np.zeros((colony_size, obj_no))
        for i in range(colony_size):
            particles_f[i, :] = targetfunction(ant_position[i, :])
        for i in range(colony_size):
            # particles_f[i, :] = targetfunction(ant_position[i, :])
            if dominates(particles_f[i, :], elite_fitness):
                elite_fitness = particles_f[i, :]
                elite_position = ant_position[i, :]

        # archive_x[0,:] = elite_position
        # archive_f[0,:] = elite_fitness

        archive_x, archive_f, archive_member_no = update_archive(archive_x, archive_f, ant_position, particles_f,
                                                                 archive_member_no)

        if archive_member_no > ARCHIVE_SIZE:
            archive_mem_ranks = ranking_process(archive_f, ARCHIVE_SIZE, obj_no)
            archive_x, archive_f, archive_mem_ranks, archive_member_no = handle_full_archive(archive_x, archive_f,
                                                                                             archive_member_no,
                                                                                             archive_mem_ranks,
                                                                                             ARCHIVE_SIZE)
        # else:
        #     archive_mem_ranks = ranking_process(archive_f, ARCHIVE_SIZE, obj_no)

        archive_mem_ranks = ranking_process(archive_f, ARCHIVE_SIZE, obj_no)

        index = roulette_wheel_selection(1.0 / archive_mem_ranks)
        if index == -1:
            index = 0
        elite_fitness = archive_f[index, :]
        elite_position = archive_x[index, :]

        # rand_ant_no = np.random.randint(0, archive_member_no)
        rand_ant_no = 0
        random_antlion_fitness = archive_f[rand_ant_no, :]
        random_antlion_position = archive_x[rand_ant_no, :]

        for i in range(colony_size):
            # index = 0
            # neighbours_no = 0

            ra = random_walk_around_antlion(dim, max_iter, lb, ub, random_antlion_position, iter)
            re = random_walk_around_antlion(dim, max_iter, lb, ub, elite_position, iter)

            ant_position[i, :] = (re[iter, :] + ra[iter, :]) / 2
            ant_position[i, :] = np.clip(ant_position[i, :], lb, ub)
        end_time = time.time()
        print(f'At the iteration {iter + 1} there are {archive_member_no} non-dominated solutions in the archive\n')
        print(f"iteration{iter + 1} has run for {round(end_time - start_time, 2)} seconds\n")
    print(archive_x)
    return archive_f


# malo = moalo(dim=DIM_SIZE, lb=MIN_VALUE_TEMPLATE, ub=MAX_VALUE_TEMPLATE, obj_no=OBJECTIVE_NUM, max_iter=3000,
#              colony_size=100, targetfunction=objective_function)
malo_zd1 = moalo(dim=5, lb=np.zeros(5), ub=np.ones(5), obj_no=2, max_iter=100, colony_size=100, targetfunction=zd1)

now = datetime.now()
timestamp = now.strftime("%Y%m%d-%H%M")
filename = f"MOALO-ZDT1-{timestamp}"

plt.style.use('classic')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 8
plt.rcParams['figure.dpi'] = 200
plt.rcParams["axes.grid"] = True
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.plot(malo_zd1[:, 0], malo_zd1[:, 1], linestyle='', marker='o', markersize=4)
plt.title('MOALO-ZDT1', fontsize=12)
plt.savefig(f'./{filename}.png')
plt.show()
