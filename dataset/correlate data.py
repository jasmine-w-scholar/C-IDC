import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D


class DataGenerator:
    def __init__(self, num_samples=3200, num_features=15):
        self.num_samples = num_samples
        self.num_features = num_features
        self.num_clusters = 4  # 4个簇
        self.X, self.Y = self.generate_data()

    def generate_data(self):
        # 计算每个簇的样本数量
        samples_per_cluster = self.num_samples // self.num_clusters
        X = []
        Y = []

        # 生成每个簇的数据
        for cluster in range(self.num_clusters):
            # 创建簇中心A
            centers = np.array([
                [1, 1, 0],  # Cluster 0
                [1, -1, 0],  # Cluster 1
                [-5, 1, 0.5],  # Cluster 2
                [-5, 1, 1.5] # Cluster 3
            ])

            # 生成强线性相关特征
            feature_1 = centers[cluster, 0]  + np.random.normal(0, 0.5, samples_per_cluster)  # 特征 1
            feature_4 = 3*feature_1 + np.random.normal(0, 0.5, samples_per_cluster)  # 特征 4

            # 生成强非线性相关特征
            feature_3 = centers[cluster, 2] + np.random.uniform(-0.2, 0.2, samples_per_cluster)  # 特征 3
            feature_5 = np.power(feature_3, 2) + np.random.normal(0, 0.5, samples_per_cluster)  # 特征 5（非线性关系）

            # 其他特征 (随机生成)
            feature_2 = centers[cluster, 1] + np.random.normal(0, 0.1, samples_per_cluster)
            other_features = np.random.normal(0, 0.1, (samples_per_cluster, self.num_features - 5))

            # 将所有特征合并
            cluster_data = np.column_stack((feature_1, feature_2, feature_3, feature_4, feature_5, other_features))
            X.append(cluster_data)
            Y.extend([cluster] * samples_per_cluster)

        # 组合所有数据
        X = np.vstack(X)
        Y = np.array(Y)

        # 预处理：归一化特征
        X = MinMaxScaler().fit_transform(X)

        # 打印每个类别的样本数量
        unique_classes, counts = np.unique(Y, return_counts=True)
        print("样本数量分布：")
        for i, count in zip(unique_classes, counts):
            print(f"类别 {i}: {count} 样本")
        return X, Y

    def visualize_data(self):
        # 2D视角绘图1，X[1]和X[2]
        plt.style.use('classic')
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.set_facecolor('w')
        scatter = ax.scatter(self.X[:, 0], self.X[:, 1], c=self.Y, s=100, alpha=0.8, cmap='viridis', edgecolor='k',
                             linewidth=2)
        ax.set_xlabel('$X_1$', fontsize=30)
        ax.set_ylabel('$X_2$', fontsize=30)
        plt.tight_layout()
        plt.xticks([])
        plt.yticks([])

        handles, labels = scatter.legend_elements()
        legend_labels = [f'Cluster {i}' for i in range(self.num_clusters)]
        unique_handles = [handles[i] for i in range(len(handles))]  # 确保颜色点唯一

        legend = ax.legend(unique_handles, legend_labels, loc="upper right", fontsize=12)
        ax.add_artist(legend)
        plt.savefig("synth_X_1_X_2.png")
        plt.show()

        # 2D视角绘图2，X[1]和X[3]
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.set_facecolor('w')
        scatter2 = ax.scatter(self.X[:, 0], self.X[:, 2], c=self.Y, s=100, alpha=0.8, cmap='viridis', edgecolor='k',
                              linewidth=2)
        ax.set_xlabel('$X_1$', fontsize=30)
        ax.set_ylabel('$X_3$', fontsize=30)
        plt.tight_layout()
        plt.xticks([])
        plt.yticks([])

        handles2, labels2 = scatter2.legend_elements()
        unique_handles2 = [handles2[i] for i in range(len(handles2))]  # 确保颜色点唯一

        legend2 = ax.legend(unique_handles2, legend_labels, loc="upper right", fontsize=12)
        ax.add_artist(legend2)

        plt.savefig("synth_X_1_X_3.png")
        plt.show()

        # 3D 绘图
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 3D 散点
        scatter3d = ax.scatter(self.X[:, 0], self.X[:, 1], self.X[:, 2], c=self.Y, s=100, alpha=0.8, cmap='viridis', edgecolor='k', linewidth=2)
        ax.set_xlabel('$X_1$', fontsize=10)
        ax.set_ylabel('$X_2$', fontsize=10)
        ax.set_zlabel('$X_3$', fontsize=10)
        ax.set_title('3D Visualization of Clusters', fontsize=25)

        # 添加图例
        handles3d, labels3d = scatter3d.legend_elements()
        legend3d = ax.legend(handles3d, legend_labels, loc="upper right", fontsize=12)
        ax.add_artist(legend3d)

        plt.tight_layout()
        plt.savefig("3d_clusters.png")
        plt.show()

    def save_data(self, filename='dataset.npz'):
        # 保存数据为 npz 格式
        np.savez(filename, X=self.X, Y=self.Y)
        print(f"数据已保存到 {filename}")
if __name__ == "__main__":
    data_gen = DataGenerator(num_samples=3200)  # 每个簇约800个样本，总共3200个样本
    data_gen.visualize_data()
    data_gen.save_data('correlate_data.npz')