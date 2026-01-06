import numpy as np

class MazeEnv:
    def __init__(self, grid, start_pos, end_pos):
        """
        初始化迷宫环境。
        :param grid: 二维 numpy 数组 (0=路径, 1=墙壁)
        :param start_pos: (行, 列)
        :param end_pos: (行, 列)
        """
        self.grid = grid
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.rows, self.cols = grid.shape
        self.state = start_pos
        
        # 动作: 0=上, 1=下, 2=左, 3=右
        self.action_space_n = 4

    def reset(self):
        """
        将环境重置为起始位置。
        :return: 初始状态 (行, 列)
        """
        self.state = self.start_pos
        return self.state

    def step(self, action):
        """
        执行动作。
        :param action: int (0=上, 1=下, 2=左, 3=右)
        :return: (next_state, reward, done)
        """
        # 当前状态
        r, c = self.state
        
        # 提议的新状态
        nr, nc = r, c
        if action == 0:   # 上
            nr -= 1
        elif action == 1: # 下
            nr += 1
        elif action == 2: # 左
            nc -= 1
        elif action == 3: # 右
            nc += 1
            
        # 检查边界
        if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
            # 撞到边界，原地不动
            reward = -10
            done = False
            return (r, c), reward, done
        
        # 检查墙壁
        if self.grid[nr, nc] == 1:
            # 撞到墙壁，原地不动
            reward = -10
            done = False
            return (r, c), reward, done
        
        # 有效移动
        self.state = (nr, nc)
        
        # 检查目标
        if (nr, nc) == self.end_pos:
            reward = 100
            done = True
        else:
            # 简化奖励函数：仅使用步数惩罚
            # 移除曼哈顿距离引导，防止智能体陷入局部最优（为了不远离目标而拒绝绕墙）
            reward = -1 
            done = False
            
        return self.state, reward, done
