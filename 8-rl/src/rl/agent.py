import numpy as np
import random

class QLearningAgent:
    def __init__(self, rows, cols, action_space_n, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.rows = rows
        self.cols = cols
        self.action_space_n = action_space_n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Q 表: (行, 列, 动作)
        self.q_table = np.zeros((rows, cols, action_space_n))

    def choose_action(self, state):
        r, c = state
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_space_n - 1)
        else:
            # 选择最佳动作
            # 添加小噪声以打破loop，而不是在全为 0 时默认选择第一个最大值
            values = self.q_table[r, c]
            max_val = np.max(values)
            # 查找具有最大值的所有索引
            candidates = np.where(values == max_val)[0]
            return np.random.choice(candidates)

    def learn(self, state, action, reward, next_state, done):
        r, c = state
        nr, nc = next_state
        
        old_value = self.q_table[r, c, action]
        if done:
            target = reward
        else:
            next_max = np.max(self.q_table[nr, nc])
            target = reward + self.gamma * next_max
            
        # 更新
        new_value = old_value + self.alpha * (target - old_value)
        self.q_table[r, c, action] = new_value

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def get_best_path(self, env):
        # 贪婪地从 Q 表中提取最佳路径，并进行循环检测
        path = []
        state = env.reset()
        path.append(state)
        
        # 跟踪已访问的状态以检测循环
        visited = set()
        visited.add(state)
        
        # 安全中断
        max_steps = self.rows * self.cols  
        steps = 0
        
        done = False
        while not done and steps < max_steps:
            r, c = state
            
            # 获取当前状态的所有 Q 值
            q_values = self.q_table[r, c]
            
            # 按 Q 值排序动作 (降序)
            sorted_actions = np.argsort(q_values)[::-1]
            
            # 尝试选择导向未访问状态的最佳动作
            chosen_action = sorted_actions[0] # 默认为最佳
            
            # 简单的向前看以避免立即循环,使用网格移动的简化模型。
            for action in sorted_actions:
                nr, nc = r, c
                if action == 0: nr -= 1
                elif action == 1: nr += 1
                elif action == 2: nc -= 1
                elif action == 3: nc += 1
                
                # 检查边界 (粗略检查，环境检查更严格)
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if (nr, nc) not in visited:
                        chosen_action = action
                        break
            
            # 执行
            next_state, reward, done = env.step(chosen_action)
            
            path.append(next_state)
            visited.add(next_state)
            
            state = next_state
            steps += 1
            
        return path
