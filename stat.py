# -*- coding: utf-8 -*-
"""
@Project : bankchain
@File    : stat.py
@Author  : Ronglin
@Date    : 2025/3/16 21:22
"""
import json
# stats.py
from typing import Dict, NamedTuple, List
import numpy as np
from graph.metrics import ShardMetrics


class PartitionStats(NamedTuple):
    """
    Statistics about a particular partition strategy.

    This class aggregates various statistical information about
    the partition, making it easy to evaluate and compare different strategies.
    """
    total_count: int  # Total edge count
    intra_count: int  # Intra-shard edge count
    cross_count: int  # Cross-shard edge count
    cross_count_rate: float  # Rate of cross-shard transactions
    load_stats: Dict  # Load statistics
    amount_stats: Dict  # Amount flow statistics
    node_stats: Dict  # Node distribution statistics

    epoch_idx: int = 0  # epoch index
    run_time: float = 0.0  # this epoch time taken
    run_time_all: float = 0.0  # Total epoch time taken
    change_node: int = 0  # Number of nodes changed in this epoch

    total_score: float = 0.0  # Total score of this epoch
    connectivity_score: float = 0.0  # Connectivity score of this epoch
    load_balance_score: float = 0.0  # Load balance score of this epoch
    amount_balance_score: float = 0.0  # Amount balance score of this epoch

    def set_epoch_info(self, epoch_idx: int, run_time: float, run_time_all: float, change_node: int,
                       total_score: float, connectivity_score: float, load_balance_score: float,
                       amount_balance_score: float) -> 'PartitionStats':
        """Set epoch information and return new instance."""
        # 创建一个新的实例，所有其他字段保持不变，只更新epoch相关字段
        return self._replace(
            epoch_idx=epoch_idx,
            run_time=run_time,
            run_time_all=run_time_all,
            change_node=change_node,
            total_score=total_score,
            connectivity_score=connectivity_score,
            load_balance_score=load_balance_score,
            amount_balance_score=amount_balance_score,
        )

    def to_dict(self) -> Dict:
        """Convert statistics to dictionary."""
        return {
            'epoch_idx': self.epoch_idx,
            'run_time': self.run_time,
            'run_time_all': self.run_time_all,
            'change_node': self.change_node,
            'total_score': self.total_score,
            'connectivity_score': self.connectivity_score,
            'load_balance_score': self.load_balance_score,
            'amount_balance_score': self.amount_balance_score,
            'total_count': self.total_count,
            'intra_count': self.intra_count,
            'cross_count': self.cross_count,
            'cross_count_rate': self.cross_count_rate,
            'load_stats': self.load_stats,
            'amount_stats': self.amount_stats,
            'node_stats': self.node_stats
        }

    @classmethod
    def from_metrics(cls, metrics: ShardMetrics, shard_nodes: Dict[int, set] = None) -> 'PartitionStats':
        """
        Calculate statistics from metrics and shard node information.

        Args:
            metrics: ShardMetrics object
            shard_nodes: Dictionary mapping shard IDs to sets of nodes

        Returns:
            PartitionStats object
        """
        # Calculate basic metrics
        total_edges = metrics.intra_count.sum() + metrics.cross_count.sum() // 2
        cross_edge_rate = (metrics.cross_count.sum() // 2) / total_edges if total_edges > 0 else 0

        # Calculate load balance metrics
        mean_load = metrics.load_value.mean()
        load_std = metrics.load_value.std()
        load_min = metrics.load_value.min()
        load_max = metrics.load_value.max()
        load_cv = load_std / mean_load if mean_load > 0 else float('inf')  # Coefficient of variation

        # Calculate amount flow balance metrics
        amount_mean = abs(metrics.pool_changes).mean()
        amount_std = abs(metrics.pool_changes).std()
        amount_min = abs(metrics.pool_changes).min()
        amount_max = abs(metrics.pool_changes).max()

        # Calculate shard net flow
        shard_net_flow = [(i, metrics.pool_changes[i]) for i in range(metrics.num_shards)]

        # Calculate nodes per shard
        if shard_nodes is None:
            nodes_per_shard = [0] * metrics.num_shards
            nodes_mean = 0
            nodes_std = 0
        else:
            nodes_per_shard = [len(shard_nodes[i]) for i in range(metrics.num_shards)]
            nodes_mean = np.mean(nodes_per_shard)
            nodes_std = np.std(nodes_per_shard)

        # Create statistics object
        return cls(
            total_count=int(total_edges),
            intra_count=int(metrics.intra_count.sum()),
            cross_count=int(metrics.cross_count.sum() // 2),
            cross_count_rate=float(cross_edge_rate),
            load_stats={
                "mean": float(mean_load),
                "std_dev": float(load_std),
                "cv": float(load_cv),
                "min": int(load_min),
                "max": int(load_max),
                "per_shard": [int(load) for load in metrics.load_value]
            },
            amount_stats={
                "mean": float(amount_mean),
                "std_dev": float(amount_std),
                "min": float(amount_min),
                "max": float(amount_max),
                "net_flow": [(int(i), float(amount)) for i, amount in shard_net_flow]
            },
            node_stats={
                "mean": float(nodes_mean),
                "std_dev": float(nodes_std),
                "per_shard": nodes_per_shard
            }
        )

    def __str__(self) -> str:
        """String representation for readable output."""
        return (
            f"PartitionStats(total_edges={self.total_count}, "
            f"intra_edges={self.intra_count}, cross_edges={self.cross_count}, "
            f"cross_rate={self.cross_count_rate:.4f})"
        )

    def print_detailed(self) -> None:
        """Print detailed statistics."""
        print("\nPartition Statistics:")
        print(
            f"Epoch: {self.epoch_idx}, Time: {self.run_time:.2f}s, Total Time: {self.run_time_all:.2f}s,moved nodes: {self.change_node}")
        print(
            f"Total score: {self.total_score:.4f}, Connectivity: {self.connectivity_score:.4f}, load_balance: {self.load_balance_score:.4f}, amount_balance: {self.amount_balance_score:.4f}")
        print(f"Total edges: {self.total_count}, Intra-shard: {self.intra_count}, Cross-shard: {self.cross_count}")
        print(f"Cross-shard rate: {self.cross_count_rate:.4f}")
        print(f"Load balance: mean={self.load_stats['mean']:.1f}, "
              f"std_dev={self.load_stats['std_dev']:.1f}, "
              f"max={self.load_stats['max']:.2f}, "
              f"min={self.load_stats['min']:.2f}, "
              f"cv={self.load_stats['cv']:.4f}")
        print(f"Amount flow: mean={self.amount_stats['mean']:.1f}, "
              f"std_dev={self.amount_stats['std_dev']:.1f}, "
              f"max={self.amount_stats['max']:.2f}, "
              f"min={self.amount_stats['min']:.2f}")
        print("Load per shard:", self.load_stats['per_shard'])
        print("Net flow per shard:", self.amount_stats['net_flow'])
        print("Nodes per shard:", self.node_stats['per_shard'])

    @staticmethod
    def save_stats_to_jsonfile(stats: List['PartitionStats'], filepath: str) -> None:
        """Save partition statistics to a file."""
        with open(filepath, 'w') as file:
            json.dump([stat.to_dict() for stat in stats], file, indent=4)
