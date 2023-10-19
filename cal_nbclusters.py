import numpy as np


# import matplotlib.pyplot as plt

def floyd(adj):
    n = len(adj)
    distance = list(map(lambda i: list(map(lambda j: j, i)), adj))
    for o in range(n):
        for p in range(n):
            if (o != p):
                if (distance[o][p] == 0):
                    distance[o][p] = 1000
    # print(distance)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                distance[i][j] = min(distance[i][j], distance[i][k] + distance[k][j])
    return distance


def cal_point(distance, cluster_dict):
    n = len(distance)
    diff_list = []
    for k, v in cluster_dict.items():
        inner_dis = 0
        outer_dis = 0
        for i in v:
            for j in v:
                inner_dis = inner_dis + distance[i][j]
        inner_dis = inner_dis / 2
        if len(v) == 1:
            avg_inner_dis = 0
        else:
            avg_inner_dis = inner_dis / (len(v) * (len(v) - 1) / 2)

        for i in v:
            for j in range(n):
                outer_dis = outer_dis + distance[i][j]
        outer_dis = outer_dis - 2 * inner_dis
        if len(v) == n:
            avg_outer_dis = 0
        else:
            avg_outer_dis = outer_dis / len(v) / (n - len(v))

        # print(inner_dis)
        # print(avg_inner_dis)
        # print(outer_dis)
        # print(avg_outer_dis)

        diff = avg_outer_dis - avg_inner_dis
        diff_list.append(diff)

    point = 0
    for diff in diff_list:
        point += diff
    point = point / len(diff_list)

    return point
