import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, image_path, target_size=(100, 100)):
        """
        初始化 ImageProcessor。
        :param image_path: 迷宫图像的路径。
        :param target_size: 离散网格的 (高度, 宽度) 元组。
        """
        self.image_path = image_path
        self.target_size = target_size
        self.original_image = None
        self.grid = None
        self.start_pos = None # (行, 列)

    def process(self):
        """
        加载图像，检测起始位置，并离散化为二进制网格。
        :return: (grid, start_pos) 元组
        """
        # 1. 加载图像
        img = cv2.imread(self.image_path)
        if img is None:
            raise FileNotFoundError(f"无法在 {self.image_path} 加载图像")
            
        # 1.5 自动裁剪黑框
        gray_temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh_temp = cv2.threshold(gray_temp, 10, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(thresh_temp)
        
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            if w < img.shape[1] or h < img.shape[0]:
                print(f"检测到黑框，正在裁剪: x={x}, y={y}, w={w}, h={h}")
                img = img[y:y+h, x:x+w]
        
        self.original_image = img
        
        # 2. 检测起始位置 (黄色)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        yellow_points = cv2.findNonZero(mask_yellow)
        
        original_h, original_w = img.shape[:2]
        
        # 3. 核心修改：基于网格块的采样 (Grid-based Pooling)
        target_h, target_w = self.target_size
        self.grid = np.zeros((target_h, target_w), dtype=int)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        step_h = original_h / target_h
        step_w = original_w / target_w
        
        print("正在执行基于块的网格映射...")
        
        for r in range(target_h):
            for c in range(target_w):
                r_start = int(r * step_h)
                r_end = int((r + 1) * step_h)
                c_start = int(c * step_w)
                c_end = int((c + 1) * step_w)
                
                patch = gray[r_start:r_end, c_start:c_end]
                
                if patch.size == 0:
                    self.grid[r, c] = 1 
                    continue
                
                mean_val = np.mean(patch)
                
                # 调整判定阈值：< 60 视为黑墙，>= 60 视为灰/白路
                if mean_val < 51:
                    self.grid[r, c] = 1 # 墙
                else:
                    self.grid[r, c] = 0 # 路

        # 4. 计算起始位置
        if yellow_points is not None:
            mean_pt = np.mean(yellow_points, axis=0)[0]
            start_x, start_y = int(mean_pt[0]), int(mean_pt[1])
            start_row = int(start_y / original_h * target_h)
            start_col = int(start_x / original_w * target_w)
            self.start_pos = (max(0, min(start_row, target_h-1)), 
                              max(0, min(start_col, target_w-1)))
        else:
            self.start_pos = (0, 0)

        # 5. 修正与强制路
        if self.grid[self.start_pos] == 1:
            self.grid[self.start_pos] = 0

        resized_mask = cv2.resize(mask_yellow, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        self.grid[resized_mask > 0] = 0

        # === 6. 新增：保存调试图像 ===
        # 创建可视化调试图：路(0)->白色(255), 墙(1)->黑色(0)
        debug_img = np.zeros_like(self.grid, dtype=np.uint8)
        debug_img[self.grid == 0] = 255  # 路变白
        debug_img[self.grid == 1] = 0    # 墙变黑
        
        # 放大保存以便观察 (放大10倍)
        debug_scale = 10
        debug_h, debug_w = debug_img.shape
        debug_resized = cv2.resize(debug_img, (debug_w * debug_scale, debug_h * debug_scale), interpolation=cv2.INTER_NEAREST)
        
        cv2.imwrite('debug_grid.png', debug_resized)
        print("已保存调试图像: debug_grid.png (白色=路, 黑色=墙)")
        # ===========================

        return self.grid, self.start_pos

    def get_grid_from_coords(self, x, y):
        """
        将原始图像坐标 (x, y) 映射到网格坐标 (行, 列)。
        """
        if self.original_image is None:
            return 0, 0
        original_h, original_w = self.original_image.shape[:2]
        scale_row = self.target_size[0] / original_h
        scale_col = self.target_size[1] / original_w
        
        r = int(y * scale_row)
        c = int(x * scale_col)
        r = max(0, min(r, self.target_size[0]-1))
        c = max(0, min(c, self.target_size[1]-1))
        return r, c
