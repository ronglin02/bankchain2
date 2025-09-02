# -*- coding: utf-8 -*-
# @Project : PyCharm
# @File    : utils.py
# @Author  : Ronglin
# @Date    : 2025/3/5 15:30
import math
from statistics import stdev
from typing import Dict

import networkx as nx
from sklearn.cluster import SpectralClustering
import numpy as np

# import pymetis

def compute_score(conn_score, load_score, amount_score, weights=None):
    # If the paper is accepted, we will release the complete code.
    pass


def community_connectivity(intra_connections, inter_connections):
    """
    计算社区连接性
    :param intra_connections: 社区内连接数,节点n在分片中，节点n对于分片的出度
    :param inter_connections: 社区间连接数，节点n在分片中，其他的出度
    :return: 社区连接性，值越大表示社区内连接越多，范围[0, 1]
    """
    return intra_connections / (intra_connections + inter_connections + 1e-10)


def load_balance_asymmetric(actual_loads, ideal_load, under_power=1.5, over_power=2):
    """
    计算不对称负载分配的分片负载度量
    增大 over_power 会对超载更不宽容
    减小 under_power 会对负载不足更宽容

    :param actual_loads: 每个节点真实的实际负载list
    :param ideal_load:  理想负载
    :param under_power: 负载不足的权重
    :param over_power: 负载过载的权重
    :return: 负载度量，值越大表示负载越均衡，范围[0, 1]
    """

    # If the paper is accepted, we will release the complete code.
    pass


def load_balance_symmetric(actual_loads, ideal_load, power=2):
    """
    计算对称负载分配的分片负载度量
    增大 power 会对负载不平衡更不宽容

    :param actual_loads: 实际负载s
    :param ideal_load:  理想负载
    :param power: 负载不平衡的权重
    :return: 负载度量，值越大表示负载越均衡，范围[0, 1]
    """
    # If the paper is accepted, we will release the complete code.
    pass

def load_balance_exponential(actual_loads, ideal_load, sensitivity=1):
    """
    基于指数衰减的负载均衡度量函数
    :param actual_loads: 每个节点真实的实际负载list
    :param ideal_load:  理想负载
    :param sensitivity: 敏感度参数，越大对偏差越敏感
    :return: 负载度量，值越大表示负载越均衡，范围[0, 1]
    """
    # If the paper is accepted, we will release the complete code.
    pass


def balance_score_combined(flows, alpha=1, beta=0.5):
    """
    :param flows: 流量列表（有正有负）
    :param alpha: 控制L2距离的衰减系数，越大越快衰减，默认1
    :param beta: 控制最大最小差距的衰减系数，默认0.5
    :return: 平衡得分，范围在 [0, 1] 之间
    """
    # If the paper is accepted, we will release the complete code.
    pass



def balance_score_abs(pool_changes,alpha=5, scale=1e5):
    """
        :param pool_changes: 流量列表（有正有负）
        :param alpha: 控制L衰减系数，越大越快衰减，默认1
        :return: 平衡得分，范围在 [0, 1] 之间
    """
    # If the paper is accepted, we will release the complete code.
    pass

def balance_score_sigmoid(pool_changes, alpha=10, scale=1e5, k=3):
    """
    :param pool_changes: 流量列表（有正有负）
    :param alpha: 控制衰减速度的系数，越大衰减越快，默认10
    :param scale: 归一化参考值，默认1e5
    :param k: Sigmoid 平移参数，默认3
    :return: 平衡得分，范围在 (0, 1) 之间
    """
    # If the paper is accepted, we will release the complete code.
    pass
