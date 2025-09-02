# -*- coding: utf-8 -*-
"""
@Project : bankchain
@File    : algorithm.py
@Author  : Ronglin
@Date    : 2025/3/16 21:38
"""

import random
import time
import networkx as nx
import numpy as np
import concurrent.futures
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Optional

from graph.metrics import ShardMetrics
from graph.stat import PartitionStats
from utils.logger import logger

class GreedyShardPA:
    """
    Greedy sharding algorithm .

    This class implements a graph-based algorithm for optimizing shard allocation
    in blockchain networks to minimize cross-shard transactions and balance load and pool.
    """

    def __init__(self, num_shards: int, iterations: int = 50, patience: int = 5, num_workers: int = 1024):
        """
        Initialize the sharding algorithm.

        Args:
            num_shards: Number of shards
            iterations: Maximum iterations
            patience: Early stopping patience
            num_workers: Number of concurrent workers
        """
        self.num_workers = num_workers
        self.num_shards = num_shards
        self.max_iterations = iterations
        self.patience = patience

        # Use undirected graph
        self.G = nx.Graph()
        self.partition_map: Dict[str, int] = {}  # Node to shard mapping

        # Shard node sets
        self.shard_nodes: Dict[int, Set[str]] = defaultdict(set)

        # Shard metrics
        self.metrics: Optional[ShardMetrics] = None

    def add_transaction(self, from_addr: str, to_addr: str, amount: float) -> None:
        """
        Add a transaction to the graph.

        Args:
            from_addr: Sender address
            to_addr: Recipient address
            amount: Transaction amount
        """
        if from_addr == to_addr:
            return

        self._add_edge(from_addr, to_addr, amount)

    def _add_edge(self, from_addr: str, to_addr: str, amount: float) -> None:
        """
        向图中添加边或更新现有边

        Args:
            from_addr: 交易发送方地址
            to_addr: 交易接收方地址
            amount: 交易金额
        """
        # If the paper is accepted, we will release the complete code.
        pass


    def initialize_partition(self) -> None:
        """
        初始化分片配置和相关数据结构
        """
        # If the paper is accepted, we will release the complete code.
        pass

    def simulate_node_move(self, node: str, new_shard: int) -> ShardMetrics:
        """
        模拟节点移动并计算新的指标

        Args:
            node: 要移动的节点
            new_shard: 目标分片ID

        Returns:
            模拟移动后的分片指标
        """
        # If the paper is accepted, we will release the complete code.
        pass

    def find_best_shard(self, node: str) -> Tuple[int, float]:
        """
        为节点找到最佳分片

        Args:
            node: 目标节点

        Returns:
            (最佳分片ID, 对应得分)的元组
        """
        # If the paper is accepted, we will release the complete code.
        pass

    def process_single_node(self, node: str) -> Optional[Tuple[str, int, float]]:
        """处理单个节点并返回移动建议"""
        # If the paper is accepted, we will release the complete code.
        pass

    def process_node_batch(self, nodes: List[str]) -> int:
        """
        使用多线程批量处理节点分片分配

        Args:
            nodes: 要处理的节点列表

        Returns:
            成功移动的节点数量
        """
        # If the paper is accepted, we will release the complete code.
        pass


    def run(self) -> None:
        """
        运行分片算法
        """
        start_time = time.time()
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # 初始化分片配置
        self.initialize_partition()

        # 初始化分数和统计信息
        initial_score = self.metrics.calculate_score()
        initial_stats = PartitionStats.from_metrics(self.metrics, self.shard_nodes)
        initial_stats = initial_stats.set_epoch_info(0, 0.0, 0.0, len(self.G.nodes),
                                                     initial_score.total, initial_score.connectivity,initial_score.load_balance,initial_score.amount_balance)

        logger.info(f"初始状态: {initial_score}")
        initial_stats.print_detailed()

        # 迭代优化
        patience_counter = 0
        nodes = list(self.G.nodes())

        history = []
        history.append(initial_stats)

        for iteration in range(self.max_iterations):
            cur_time = time.time()
            # 打乱节点顺序
            random.shuffle(nodes)
            total_changes = self.process_node_batch(nodes)
            # 计算当前指标
            current_score = self.metrics.calculate_score()
            current_stats = PartitionStats.from_metrics(self.metrics, self.shard_nodes)
            current_stats = current_stats.set_epoch_info(iteration + 1, time.time() - cur_time, time.time() - start_time,
                                                         total_changes, current_score.total, current_score.connectivity, current_score.load_balance, current_score.amount_balance)
            logger.info(f"迭代 {iteration + 1}: {current_score}, 变化: {total_changes}")
            # current_stats.print_detailed()
            history.append(current_stats)

            # 将histroy保存到文件
            filename = f'{self.num_shards}_{timestamp}.json'
            PartitionStats.save_stats_to_jsonfile(history, filename)

            # 检查早停条件
            if total_changes == 0:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info("早停条件满足，停止迭代")
                    break
            else:
                patience_counter = 0


        # 计算最终指标
        final_score = self.metrics.calculate_score()
        final_stats = PartitionStats.from_metrics(self.metrics, self.shard_nodes)
        final_stats.print_detailed()
        logger.info(f"优化后的LPA完成, 耗时: {time.time() - start_time:.2f}秒, 最佳得分: {final_score}")


if __name__ == "__main__":
    partitioner = GreedyShardPA(num_shards=2, num_workers=1024)

    transactions = [
        ("1", "2", 7),
        ("1", "3", 2),
        ("1", "4", 10),
        ("3", "1", 1),
        ("2", "3", 6),
        ("2", "5", 5),
        ("5", "2", 7),
        ("4", "5", 13),
        ("3", "4", 11),
    ]

    for tx in transactions:
        partitioner.add_transaction(*tx)

    # 运行分片算法
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - 开始运行优化后的LPA算法...")
    partitioner.run()
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - 运行完毕")

