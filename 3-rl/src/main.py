import sys
import os
# 确保如果直接运行，src 在路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
from src.utils.image_processor import ImageProcessor
from src.rl.environment import MazeEnv
from src.rl.agent import QLearningAgent

def main():
    # 1. 处理图像
    img_path = 'maze.jpg'
    processor = ImageProcessor(img_path, target_size=(49, 66))
    try:
        grid, start_pos = processor.process()
    except FileNotFoundError:
        print(f"错误: 找不到 {img_path}。")
        return
    
    print(f"网格大小: {grid.shape}")
    print(f"起始位置 (网格): {start_pos}")
    
    # 2. 选择目标
    # 查找所有有效 (0) 位置（0 为路径，1 为墙壁）
    valid_positions = np.argwhere(grid == 0)
    # 过滤掉起始位置并转换为元组列表
    valid_positions = [tuple(p) for p in valid_positions if tuple(p) != start_pos]
    
    if not valid_positions:
        print("错误: 未找到有效的路径单元！请检查图像阈值处理。")
        return

    # 35-45 lines logic replacement/enhancement
    # 改进的目标选择：找到距离起点最远的点
    max_dist = -1
    best_end_pos = None

    # 遍历所有有效位置以找到具有最大曼哈顿距离的位置
    for candidate in valid_positions:
        dist = abs(candidate[0] - start_pos[0]) + abs(candidate[1] - start_pos[1])
        if dist > max_dist:
            max_dist = dist
            best_end_pos = candidate
            
    print(f"建议的终点 (距离: {max_dist}): {best_end_pos}")

    # --- 用户交互部分 ---
    user_input = input("是否输入自定义终点坐标? (y/n) [默认 n]: ").strip().lower()
    
    end_pos = best_end_pos # 默认使用自动计算的最佳终点
    
    if user_input == 'y':
        while True:
            try:
                coords_str = input(f"请输入终点坐标 (行, 列)，范围 (0-{grid.shape[0]-1}, 0-{grid.shape[1]-1})，以逗号分隔: ")
                parts = coords_str.split(',')
                if len(parts) != 2:
                    print("格式错误。请输入两个数字，用逗号分隔。")
                    continue
                    
                r, c = int(parts[0].strip()), int(parts[1].strip())
                
                # 验证范围
                if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
                    print("坐标超出网格范围。")
                    continue
                    
                # 验证是否为墙壁
                if grid[r, c] == 1:
                    print("该位置是墙壁 (值为1)，请选择路径点 (值为0)。")
                    continue
                
                # 验证是否与起点重合
                if (r, c) == start_pos:
                    print("终点不能与起点相同。")
                    continue
                    
                end_pos = (r, c)
                print(f"已设置自定义终点: {end_pos}")
                break
            except ValueError:
                print("无效输入。请输入整数坐标。")

    if max_dist < 20 and end_pos == best_end_pos:
        print(f"警告: 最远的可到达点仅在 {max_dist} 步之外。")
        print("这表明迷宫可能被堵塞，或者起点位于一个小的隔离区域。")
    
    print(f"最终确认终点: {end_pos}")
    
    # 3. 设置强化学习
    env = MazeEnv(grid, start_pos, end_pos)
    # 为网格世界调整的超参数
    agent = QLearningAgent(
        rows=grid.shape[0], 
        cols=grid.shape[1], 
        action_space_n=4, 
        alpha=0.1, 
        gamma=0.99, 
        epsilon=1.0, 
        epsilon_decay=0.997 # 较慢的衰减以利于探索
    )
    
    episodes = 10000 
    print(f"开始训练 {episodes} 个回合...")
    
    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        # 每回合最大步数以防止陷入死循环
        max_steps = 5000
        
        while not done and steps < max_steps:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
            
        agent.decay_epsilon()
        
        if (e+1) % 500 == 0:
            print(f"Episode {e+1}/{episodes} - Reward: {total_reward} - Epsilon: {agent.epsilon:.3f}")

    # 4. 提取路径
    print("训练完成。正在提取最佳路径...")
    best_path = agent.get_best_path(env)
    print(f"最佳路径长度: {len(best_path)}")
    
    reached_goal = False
    if len(best_path) > 0 and best_path[-1] == end_pos:
        print("成功: 智能体到达了目标！")
        reached_goal = True
    else:
        print("警告: 智能体在最佳路径追踪中未能到达目标。")
    
    # 5. 可视化
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 尝试显示原始图像作为背景，这样可以看到真实的墙壁
    if processor.original_image is not None:
        # OpenCV 使用 BGR，Matplotlib 使用 RGB
        img_rgb = processor.original_image[:, :, ::-1]
        
        # 获取网格尺寸
        grid_h, grid_w = grid.shape
        
        # 使用 extent 参数将图像坐标映射到网格坐标系
        # extent = [left, right, bottom, top]
        # 这样轴标签就会显示网格索引而不是像素
        ax.imshow(img_rgb, extent=[0, grid_w, grid_h, 0])
        
        # 坐标转换函数：网格 (row, col) -> 绘图坐标 (x, y)
        # 现在图像已经缩放到网格坐标系，只需要偏移到单元中心 (+0.5)
        def to_img_coords(r, c):
            return (c + 0.5, r + 0.5)
            
        # 转换起点和终点
        start_plot = to_img_coords(*start_pos)
        end_plot = to_img_coords(*end_pos)
        
        ax.plot(start_plot[0], start_plot[1], 'yo', markersize=12, label='Start')
        ax.plot(end_plot[0], end_plot[1], 'ro', markersize=12, label='Goal')
        
        # 绘制路径
        if len(best_path) > 0:
            path_x = []
            path_y = []
            for r, c in best_path:
                px, py = to_img_coords(r, c)
                path_x.append(px)
                path_y.append(py)
            ax.plot(path_x, path_y, 'b-', linewidth=3, label='Path')
            
    else:
        # 如果没有原始图像，回退到显示网格
        # 用户偏好：恢复灰白色墙壁。
        # grid: 1=墙壁, 0=路径
        # cmap='gray': 0=黑色(路径), 1=白色(墙壁)
        ax.imshow(grid, cmap='gray')
        
        # 标记起点和终点 (y, x) -> (row, col)
        ax.plot(start_pos[1], start_pos[0], 'yo', markersize=12, label='Start') # 黄色
        ax.plot(end_pos[1], end_pos[0], 'ro', markersize=12, label='Goal')   # 红色
        
        # 如果存在，绘制最佳路径
        if len(best_path) > 0:
            rows = [p[0] for p in best_path]
            cols = [p[1] for p in best_path]
            ax.plot(cols, rows, 'b-', linewidth=2, label='Path') # 蓝色路径
    
    ax.legend(loc='upper right')
    ax.set_title(f"Q-Learning Maze Solver\nResult: {'Success' if reached_goal else 'Failed'}")
    
    output_file = 'maze_solution.png'
    plt.savefig(output_file)
    print(f"结果图像已保存至 {output_file}")

if __name__ == "__main__":
    main()
