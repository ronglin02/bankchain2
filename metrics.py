# -*- coding: utf-8 -*-
"""
@Project : bankchain
@File    : metrics.py
@Author  : Ronglin
@Date    : 2025/3/16 21:37
"""

import numpy as np
from dataclasses import dataclass
import networkx as nx
from typing import Dict, NamedTuple

from graph import utils as utils


class ScoreResult(NamedTuple):
    """
    Represents the result of scoring a shard configuration.

    This immutable class holds scores for various aspects of sharding performance.
    """
    total: float  # Total score
    connectivity: float  # Connectivity score
    load_balance: float  # Load balance score
    amount_balance: float  # Amount balance score

    def to_dict(self) -> Dict:
        """Convert score results to dictionary."""
        return {
            'total': self.total,
            'connectivity': self.connectivity,
            'load_balance': self.load_balance,
            'amount_balance': self.amount_balance
        }

    def __str__(self) -> str:
        """String representation for readable output."""
        return (f"Score(total={self.total:.4f}, connectivity={self.connectivity:.4f}, "
                f"load_balance={self.load_balance:.4f}, amount_balance={self.amount_balance:.4f})")


@dataclass
class ShardMetrics:
    """
    Represents metrics for shards in a blockchain network.

    This class stores and calculates various metrics related to shard performance,
    including transaction counts, cross-shard transactions, and fund pool changes.
    """
    num_shards: int  # Number of shards
    intra_count: np.ndarray  # Intra-shard transaction counts
    cross_count: np.ndarray  # Cross-shard transaction counts
    pool_changes: np.ndarray  # Fund pool changes for each shard
    load_value: np.ndarray  # Load balance values

    @classmethod
    def create_empty(cls, num_shards: int) -> 'ShardMetrics':
        """Create an empty metrics object with zeroed arrays."""
        return cls(
            num_shards=num_shards,
            intra_count=np.zeros(num_shards, dtype=np.int32),
            cross_count=np.zeros(num_shards, dtype=np.int32),
            pool_changes=np.zeros(num_shards, dtype=np.float64),
            load_value=np.zeros(num_shards, dtype=np.int32)
        )

    @classmethod
    def compute_from_graph(cls, G: nx.Graph, num_shards: int, partition_map: Dict[str, int]) -> 'ShardMetrics':
        """
        Compute metrics from a transaction graph and partition mapping.

        Args:
            G: Transaction graph
            num_shards: Number of shards
            partition_map: Mapping of nodes to shards

        Returns:
            ShardMetrics object
        """
        metrics = cls.create_empty(num_shards)

        for a_addr, b_addr, data in G.edges(data=True):
            small_addr = min(a_addr, b_addr)
            large_addr = max(a_addr, b_addr)
            small_addr_shard = partition_map[small_addr]
            large_addr_shard = partition_map[large_addr]
            count = data['count']
            amount = data['amount']

            if small_addr_shard == large_addr_shard:  # 分片内交易
                metrics.intra_count[small_addr_shard] += count
                metrics.load_value[small_addr_shard] += count

            else:  # 跨分片交易
                # 跨分片交易数量统计
                metrics.cross_count[small_addr_shard] += count
                metrics.cross_count[large_addr_shard] += count

                # 负载计算
                metrics.load_value[small_addr_shard] += count
                metrics.load_value[large_addr_shard] += count

                # 处理资金池的变化
                value = abs(amount)
                if amount > 0:
                    # 表示从small -> large
                    # small的资金池增多，large的资金池减少（因为从资金池进行转账）
                    metrics.pool_changes[small_addr_shard] += value  # small_addr -> pool;
                    metrics.pool_changes[large_addr_shard] -= value  # pool -> large_addr
                else:
                    # 表示从large -> small
                    # large的资金池增多，small的资金池减少
                    metrics.pool_changes[large_addr_shard] += value  # large_addr -> pool;
                    metrics.pool_changes[small_addr_shard] -= value  # pool -> small_addr
        return metrics

    def copy(self) -> 'ShardMetrics':
        """Create a deep copy of this metrics object."""
        return ShardMetrics(
            num_shards=self.num_shards,
            intra_count=self.intra_count.copy(),
            cross_count=self.cross_count.copy(),
            pool_changes=self.pool_changes.copy(),
            load_value=self.load_value.copy()
        )

    def calculate_score(self, ideal_flag: bool = False, asym: bool = True) -> 'ScoreResult':
        """
        Calculate score based on metrics.

        Args:
            utils_module: Module containing utility functions
            ideal_flag: Whether to consider ideal sharding
            asym: Whether to use asymmetric load balance calculation

        Returns:
            ScoreResult object
        """
        # Calculate connectivity score
        connectivity = utils.community_connectivity(
            self.intra_count.sum(),
            self.cross_count.sum() // 2
        )

        # Calculate ideal load
        if ideal_flag:
            ideal_load = (self.load_value.sum() - self.cross_count.sum() // 2) / self.num_shards
        else:
            ideal_load = self.load_value.sum() / self.num_shards

        # Calculate load balance score
        if asym:
            load_balance = utils.load_balance_asymmetric(self.load_value, ideal_load)
        else:
            load_balance = utils.load_balance_symmetric(self.load_value, ideal_load)

        # Calculate amount score
        # amount_score = utils.balance_score_combined(self.pool_changes)
        # amount_score = utils.balance_score_l2(self.pool_changes)
        amount_score = utils.balance_score_abs_max(self.pool_changes)
        # amount_score = utils.balance_sigmoid_score(self.pool_changes)

        # Calculate total score
        total_score = utils.compute_score(connectivity, load_balance, amount_score)

        # 为了消融实验，只能将总分分别设置为单独的一项分数
        total_score = connectivity
        # total_score = load_balance
        # total_score = amount_score

        return ScoreResult(total_score, connectivity, load_balance, amount_score)
