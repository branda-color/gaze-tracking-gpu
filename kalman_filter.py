import numpy as np

class KalmanFilter:
    def __init__(self, initial_state, state_covariance, process_noise, measurement_noise):
        """
        初始化卡爾曼濾波器
        :param initial_state: 初始狀態 [位置, 速度]
        :param state_covariance: 狀態協方差矩陣
        :param process_noise: 過程噪聲協方差矩陣
        :param measurement_noise: 測量噪聲協方差矩陣
        """
        self.state = np.array(initial_state, dtype=np.float32)  # 初始狀態 [位置, 速度]
        self.P = np.array(state_covariance, dtype=np.float32)  # 狀態協方差
        self.Q = np.array(process_noise, dtype=np.float32)  # 過程噪聲
        self.R = np.array(measurement_noise, dtype=np.float32)  # 測量噪聲
        self.F = np.array([[1, 1], [0, 1]], dtype=np.float32)  # 狀態轉換矩陣
        self.H = np.array([[1, 0]], dtype=np.float32)  # 測量矩陣

    def predict(self):
        """
        預測下一時刻的狀態
        """
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state

    def update(self, measurement):
        """
        更新狀態根據測量值
        :param measurement: 當前測量值
        """
        z = np.array([measurement], dtype=np.float32)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.state += K @ y
        self.P = (np.eye(len(self.P)) - K @ self.H) @ self.P
        return self.state
