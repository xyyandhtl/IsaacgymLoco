from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # only if 3D
import numpy as np
import os


class LatentVisualizerBatch:
    def __init__(self, dim=2, save_dir="latent_vis"):
        self.pca = PCA(n_components=dim)
        self.dim = dim
        self.step = 0

        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

        self.fig = plt.figure()
        if dim == 3:
            from mpl_toolkits.mplot3d import Axes3D
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.ax = self.fig.add_subplot(111)

        self.pca_fitted = False
        self.low_dim_history = []  # 存储历史降维结果

        # 🎨 生成固定颜色数组（使用 colormap tab20 或其他）
        # self.env_num = env_num
        # cmap = plt.get_cmap("tab20")  # 支持多达20个明显可区分颜色
        # self.colors = [cmap(i % 100) for i in range(self.env_num)]

    def update(self, latent_batch):
        """
        latent_batch: shape (n, 16) tensor or ndarray
        """
        latent_np = latent_batch.detach().cpu().numpy() if hasattr(latent_batch, 'detach') else latent_batch

        if latent_np.shape[0] < 2:
            return  # 至少两个点才有意义的投影方向

        if not self.pca_fitted:
            # 用第一批样本拟合 PCA
            self.pca.fit(latent_np)
            self.pca_fitted = True
        low_dim = self.pca.transform(latent_np)

        # 累加历史降维结果
        self.low_dim_history.append(low_dim)
        all_low = np.vstack(self.low_dim_history)

        self.ax.clear()
        if self.dim == 3:
            self.ax.scatter(all_low[:, 0], all_low[:, 1], all_low[:, 2],
                            c=np.arange(len(all_low)), cmap='viridis', s=8)
        else:
            self.ax.scatter(all_low[:, 0], all_low[:, 1],
                            c=np.arange(len(all_low)), cmap='viridis', s=8)

        self.step += 1
        if self.step % 10 == 0:
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}/frame_{self.step:05d}.png")

        # plt.draw()
        # plt.pause(0.001)


class LatentVisualizerSingle:
    def __init__(self, dim=2, buffer_size=1000):
        self.pca = PCA(n_components=dim)
        self.latents = []
        self.buffer_size = buffer_size

        # 初始化图形
        self.dim = dim
        self.fig = plt.figure()
        if dim == 3:
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.ax = self.fig.add_subplot(111)

        plt.ion()
        plt.show()

    def update(self, latent):  # latent: shape (1, 16)
        latent = latent.detach().cpu().numpy().reshape(1, -1)
        self.latents.append(latent)

        if len(self.latents) > self.buffer_size:
            self.latents.pop(0)

        latents_np = np.vstack(self.latents)

        # 若数据不足不能拟合降维，先跳过
        if latents_np.shape[0] < 5:
            return

        low_dim = self.pca.fit_transform(latents_np)

        self.ax.clear()
        if self.dim == 3:
            self.ax.scatter(low_dim[:, 0], low_dim[:, 1], low_dim[:, 2], c='blue')
        else:
            self.ax.scatter(low_dim[:, 0], low_dim[:, 1], c='blue')

        # plt.draw()
        # plt.pause(0.001)
        if len(self.latents) % 100 == 0:
            plt.tight_layout()
            plt.savefig(f"latent_vis/frame_{len(self.latents):05d}.png")
