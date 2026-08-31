import torch
import math


def init_weights_normal(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, std=0.001)
        if 'bias' in vars(m).keys():
            m.bias.data.fill_(0.0)


def clustering_head(cfg):
    return torch.nn.Sequential(
        torch.nn.Linear(cfg.clustering_head[0], cfg.clustering_head[1]),
        torch.nn.BatchNorm1d(cfg.clustering_head[1]),
        torch.nn.ReLU(),
        torch.nn.Linear(cfg.clustering_head[1], cfg.n_clusters)).apply(init_weights_normal)


def aux_classifier_head(cfg):
    return torch.nn.Sequential(
        torch.nn.Linear(cfg.input_dim, cfg.aux_classifier[0]),
        torch.nn.BatchNorm1d(cfg.aux_classifier[0]),
        torch.nn.ReLU(),
        torch.nn.Linear(cfg.aux_classifier[0], cfg.n_clusters)).apply(init_weights_normal)


class EncoderDecoder(torch.nn.Module):
    def __init__(self, cfg):
        super(EncoderDecoder, self).__init__()
        self.cfg = cfg
        self.encoder = []
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.encoder.apply(init_weights_normal)
        self.decoder.apply(init_weights_normal)

    def build_encoder(self):
        layers = [
            torch.nn.Linear(self.cfg.input_dim, self.cfg.encdec[0]),
            torch.nn.BatchNorm1d(self.cfg.encdec[0]),
            torch.nn.ReLU()
        ]
        hidden_layers = len(self.cfg.encdec) // 2 + 1
        for layer_idx in range(1, hidden_layers):
            if layer_idx == hidden_layers - 1:
                layers += [torch.nn.Linear(self.cfg.encdec[layer_idx - 1], self.cfg.encdec[layer_idx])]
            else:
                layers += [
                    torch.nn.Linear(self.cfg.encdec[layer_idx - 1], self.cfg.encdec[layer_idx]),
                    torch.nn.BatchNorm1d(self.cfg.encdec[layer_idx]),
                    torch.nn.ReLU()
                ]
        return torch.nn.Sequential(*layers)

    def build_decoder(self):
        hidden_layers = len(self.cfg.encdec) // 2 + 1
        layers = []
        for layer_idx in range(hidden_layers, len(self.cfg.encdec)):
            layers += [
                torch.nn.Linear(self.cfg.encdec[layer_idx - 1], self.cfg.encdec[layer_idx]),
                torch.nn.BatchNorm1d(self.cfg.encdec[layer_idx]),
                torch.nn.ReLU()
            ]
        layers += [torch.nn.Linear(self.cfg.encdec[-1], self.cfg.input_dim)]
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))

class GatingNet(torch.nn.Module):
    def __init__(self, cfg, cholesky_L=None):
        super(GatingNet, self).__init__()
        self.cfg = cfg
        self._sqrt_2 = math.sqrt(2)
        self.sigma = 0.5
        # ✅[新增]注册 Cholesky 矩阵为 buffer
        # register_buffer 的三个优点：
        #   1. 自动随 model.to(device) 移动到 GPU，无需手动处理
        #   2. 不参与梯度计算（固定不变的统计量），节省显存
        #   3. model.save/load 时自动包含，保证复现性
        if cholesky_L is not None:
            self.register_buffer('cholesky_L', cholesky_L)
            print(f"✅ [SEFS] IDC + SEFS 相关噪声已启用，"
                  f"Cholesky L 形状: {cholesky_L.shape}")
        else:
            self.register_buffer('cholesky_L', None)
            print("⚠️ [SEFS] cholesky_L 未传入，退化为 IDC 原始独立高斯噪声")

        self.local_gates = torch.nn.Sequential(
            torch.nn.Linear(cfg.input_dim, cfg.gates_hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(cfg.gates_hidden_dim, cfg.input_dim),
            torch.nn.Tanh()
        )
        self.local_gates.apply(self.init_weights)
        self.global_gates_net = torch.nn.Embedding(self.cfg.n_clusters, self.cfg.input_dim)
        torch.nn.init.normal_(self.global_gates_net.weight, std=0.01)

    @staticmethod
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.001)
            if 'bias' in vars(m).keys():
                m.bias.data.fill_(0.0)

    def global_forward(self, y):
        """
                全局门控前向传播
                与 IDC 原版完全一致
                注意：全局门控保留 IDC 原始独立高斯噪声（全局统计量无需相关性）
                """
        noise = torch.normal(mean=0, std=self.sigma, size=(y.size(0), self.cfg.input_dim),
                             device=self.global_gates_net.weight.device)
        z = torch.tanh(self.global_gates_net(y)) + .5 * noise * self.training
        gates = self.hard_sigmoid(z)
        return torch.tanh(self.global_gates_net(y)), gates

    def open_global_gates(self):
        return self.hard_sigmoid(torch.tanh(self.global_gates_net.weight)).sum(dim=1).mean().cpu().item()

    def forward(self, x):
        """
        IDC 原版 forward，引入 SEFS Relaxed-MultiBern 相关噪声

        ═══════════════════════════════════════════════════════════════
        IDC 原版噪声生成逻辑（独立高斯，替换掉）：
            mu = local_gates(x)                    # 门控均值
            z  = mu + 0.5 * randn_like(mu) * training  # 加独立噪声
            gates = hard_sigmoid(z)

        ✅ SEFS Relaxed-MultiBern 完整实现（Algorithm 4）：

        Step 1: ε ~ N(0, I)  [标准独立高斯，与 IDC 原版起点相同]

        Step 2: v = L @ ε ~ N(0, R)  [Cholesky 变换得到相关高斯]
                → 高度相关的特征 j,k 满足 Cov(v_j, v_k) = R_jk ≈ 1
                → 它们会得到几乎相同的噪声，形成"同步竞争"

        Step 3: u_k = Φ(v_k)  [Gaussian CDF，将相关高斯映射到 (0,1)]
                → 对应 SEFS 源码：u = 0.5*(1 + erf(v/√2))

        Step 4: m̃_k = σ(1/τ * (logit(π_k) + logit(u_k)))
                其中 π_k = sigmoid(mu_k)  [IDC local_gates 输出作为选择概率]
                → 对应 SEFS Algorithm 4 的 reparameterization trick
                → 等价于 Gumbel-Softmax 在二值门控上的推广

        Step 5: gates = hard_sigmoid(m̃)  [与 IDC 原版接口保持一致]
        ═══════════════════════════════════════════════════════════════

        参数：
            x: [batch_size, input_dim]

        返回：(mu, z, gates) — 与 IDC 原版接口完全一致，下游代码无需改动
        """
        # ── Step 0: 通过 local_gates 网络得到门控均值 mu ──
        mu = self.local_gates(x)   # [batch, D]，值域 (-1, 1)（因为最后 Tanh）
        if self.training:
            use_sefs = getattr(self.cfg, 'use_sefs', True)  # yaml 中控制

            if use_sefs:
                # ── IDC + SEFS 相关噪声 ──
                epsilon = torch.randn_like(mu)
                if self.cholesky_L is not None:
                    v = torch.matmul(epsilon, self.cholesky_L.t())
                else:
                    v = epsilon
                u = 0.5 * (1.0 + torch.erf(v / self._sqrt_2))
                u = u.clamp(min=1e-6, max=1.0 - 1e-6)
                tau = getattr(self.cfg, 'sefs_tau', 0.5)

                logit_u = torch.log(u) - torch.log(1.0 - u)
                m_tilde = torch.sigmoid((mu + logit_u) / tau)
                z = m_tilde
                gates = m_tilde
            else:
                # ── IDC 原版独立高斯噪声 ──
                noise = torch.normal(mean=0, std=self.sigma, size=x.size(), device=x.device)
                z = mu + .5 * noise * self.training
                gates = self.hard_sigmoid(z)
                sparse_x = x * gates
        else:
            z = mu
            gates = self.hard_sigmoid(z)

        return mu, z, gates
        # if self.training:

        #     # ════════════════════════════════════════════════════════
        #     # ✅ SEFS Relaxed-MultiBern 相关噪声（仅训练时启用）
        #     # ════════════════════════════════════════════════════════
        #
        #     # Step 1: 生成标准独立高斯噪声 ε ~ N(0, I)
        #     # shape: [batch, D]
        #     epsilon = torch.randn_like(mu)
        #
        #     # Step 2: Cholesky 变换生成相关高斯 v ~ N(0, R)
        #     # v = ε @ L^T  等价于 v = (L @ ε^T)^T
        #     # 相关特征 j,k 满足 Cov(v_j, v_k) = R_jk，形成同向涨落
        #     # cholesky_L 已通过 register_buffer 自动移至正确 device
        #     if self.cholesky_L is not None:
        #         # [batch, D] @ [D, D]^T = [batch, D]
        #         v = torch.matmul(epsilon, self.cholesky_L.t())
        #     else:
        #         # 退化为原始独立高斯噪声（与 IDC 原版等价）
        #         v = epsilon
        #
        #     # Step 3: Gaussian CDF  u_k = Φ(v_k)
        #     # 将相关高斯 v 映射到 (0,1) 区间的相关均匀随机变量 u
        #     # 对应 SEFS 源码：u = Gaussian_CDF(q) = 0.5*(1 + erf(q/√2))
        #     # 数值稳定性：clamp 防止 log(0) 出现 NaN
        #     u = 0.5 * (1.0 + torch.erf(v / self._sqrt_2))          # [batch, D]
        #     u = u.clamp(min=1e-6, max=1.0 - 1e-6)                  # 数值稳定
        #
        #     # Step 4: Relaxed-MultiBern 重参数化
        #     # π_k = sigmoid(mu_k)  ← IDC local_gates 输出作为选择概率
        #     # m̃_k = σ(1/τ * (log π_k/(1-π_k) + log u_k/(1-u_k)))
        #     #
        #     # 对应 SEFS Algorithm 4：
        #     #   m̃_k = σ(1/τ * (log π - log(1-π) + log u - log(1-u)))
        #     #
        #     # 温度参数 τ = 1.0（与 SEFS Table S.1 一致，可在 cfg 中调整）
        #     tau = getattr(self.cfg, 'sefs_tau', 1.0)
        #
        #     # logit(π) = log(π/(1-π))，π = sigmoid(mu)
        #     # 由于 mu 经过 Tanh，直接用 mu 作为 logit(π) 的近似
        #     # 更精确：logit(sigmoid(mu)) = mu（恒等式，可直接用 mu）
        #     # 对应 SEFS 源码：log(pi) - log(1-pi) = pi_logit（直接存 logit）
        #     logit_pi = mu                                           # [batch, D]
        #
        #     # logit(u) = log(u/(1-u))
        #     logit_u = torch.log(u) - torch.log(1.0 - u)           # [batch, D]
        #
        #     # 合并：m̃ = σ(1/τ * (logit(π) + logit(u)))
        #     m_tilde = torch.sigmoid((logit_pi + logit_u) / tau)    # [batch, D]
        #
        #     # Step 5: 保持与 IDC 原版接口一致
        #     # IDC 原版返回 z（加噪后的 mu），这里用 m_tilde 替代 z 的语义
        #     z = m_tilde
        #     gates = m_tilde
        #
        #
        # else:
        #     # ── 推理/验证阶段：不加噪声，直接用确定性门控 ──
        #     # 与 IDC 原版 eval 模式完全一致
        #     z = mu
        #     gates = self.hard_sigmoid(z)
        #
        # return mu, z, gates   # 接口与 IDC 原版完全一致，下游代码零改动

    @staticmethod
    def hard_sigmoid(x):
        return torch.clamp(x + .5, 0.0, 1.0)

    def regularization(self, mu, reduction_func=torch.mean):
        return reduction_func(0.5 - 0.5 * torch.erf((-1 / 2 - mu) / self._sqrt_2))

    def get_gates(self, x):
        with torch.no_grad():
            gates = self.hard_sigmoid(self.local_gates(x))
        return gates

    def num_open_gates(self, x, ):
        return self.get_gates(x).sum(dim=1).cpu().median(dim=0)[0].item()