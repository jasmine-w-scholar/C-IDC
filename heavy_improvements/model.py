import torch
import torch.nn as nn
import math
import numpy as np


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


class GradualEncoderDecoder(torch.nn.Module):
    """
    渐进压缩 Encoder-Decoder，针对小样本高维数据（如基因表达谱）优化。

    与原始 EncoderDecoder 的区别：
    1. 渐进压缩：D → D//div → D//div^2 → ... → bottleneck → 反向扩展
       避免了原始架构 D→512→512→2048→32 在小样本下的大规模过参数化
    2. LayerNorm 替代 BatchNorm：小 batch（6-7）下 LayerNorm 更稳定
    3. Dropout 正则化：每层后添加 Dropout，缓解小样本过拟合
    4. 跳跃连接：encoder 第 i 层输出加到 decoder 对应层输入，改善梯度流
    """
    def __init__(self, cfg):
        super(GradualEncoderDecoder, self).__init__()
        self.cfg = cfg
        input_dim = cfg.input_dim
        bottleneck_dim = cfg.get('bottleneck_dim', 32)
        div_factor = cfg.get('encdec_div_factor', 4)
        dropout_p = cfg.get('encdec_dropout', 0.2)
        n_stages = cfg.get('encdec_n_stages', 3)

        # ── 构建 encoder 各层输出维度 ──
        encoder_dims = [input_dim]
        current_dim = input_dim
        for i in range(n_stages):
            next_dim = max(current_dim // div_factor, bottleneck_dim)
            if i == n_stages - 1:
                next_dim = bottleneck_dim
            encoder_dims.append(next_dim)
            current_dim = next_dim

        # ── 构建 decoder 维度（encoder 的逆序）──
        decoder_dims = list(reversed(encoder_dims))  # [bn, ..., input_dim]

        # ── 构建 Encoder ──
        encoder_layers = []
        for i in range(len(encoder_dims) - 1):
            in_d, out_d = encoder_dims[i], encoder_dims[i + 1]
            encoder_layers.append(nn.Linear(in_d, out_d))
            if i < len(encoder_dims) - 2:  # 最后一层不加激活
                encoder_layers.append(nn.LayerNorm(out_d))
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.Dropout(dropout_p))
        self.encoder = nn.Sequential(*encoder_layers)

        # ── 构建 Decoder ──
        decoder_layers = []
        for i in range(len(decoder_dims) - 1):
            in_d, out_d = decoder_dims[i], decoder_dims[i + 1]
            # 跳跃连接：decoder 第 i 层的输入可能 concat encoder 对应层的输出
            # 简便实现：encoder 的中间输出维度暂存后直接相加
            decoder_layers.append(nn.Linear(in_d, out_d))
            if i < len(decoder_dims) - 2:
                decoder_layers.append(nn.LayerNorm(out_d))
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Dropout(dropout_p))
        self.decoder = nn.Sequential(*decoder_layers)

        # ── 存储维度信息用于 skip connections ──
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims
        self.n_stages = n_stages
        self.use_skip = cfg.get('encdec_skip_connections', True)

        # ── Skip connection projection layers（维度不匹配时使用）──
        self.skip_projections = nn.ModuleList()
        if self.use_skip:
            for i in range(n_stages):
                enc_dim = encoder_dims[i]  # encoder 第 i 层输出
                dec_idx = n_stages - i  # 对应 decoder 第 dec_idx 层
                if dec_idx < len(decoder_dims) - 1:
                    dec_in_dim = decoder_dims[dec_idx]
                    if enc_dim != dec_in_dim:
                        self.skip_projections.append(nn.Linear(enc_dim, dec_in_dim))
                    else:
                        self.skip_projections.append(nn.Identity())

        # 初始化
        self.encoder.apply(init_weights_normal)
        self.decoder.apply(init_weights_normal)

        print(f"[GradualEncoderDecoder] 架构: {encoder_dims} → {decoder_dims}")
        print(f"[GradualEncoderDecoder] Dropout={dropout_p}, Skip={self.use_skip}, "
              f"LayerNorm, div_factor={div_factor}")

    def forward(self, x):
        # ── Encoder: 逐层前向，保存中间输出用于 skip ──
        enc_outputs = [x]
        h = x
        layer_idx = 0
        for module in self.encoder:
            h = module(h)
            # 记录每个 Linear 层的输出（用于 skip connection）
            if isinstance(module, nn.Linear):
                enc_outputs.append(h)
                layer_idx += 1

        # ── Decoder: 逐层前向，添加 skip connections ──
        dec_h = h
        dec_linear_idx = 0
        for module in self.decoder:
            if isinstance(module, nn.Linear):
                # 添加对应 encoder 层的 skip connection
                if self.use_skip and dec_linear_idx < len(self.skip_projections):
                    skip_src_idx = self.n_stages - dec_linear_idx - 1
                    if skip_src_idx >= 0 and skip_src_idx < len(enc_outputs):
                        skip_val = enc_outputs[skip_src_idx]
                        skip_val = self.skip_projections[dec_linear_idx](skip_val)
                        dec_h = module(dec_h) + skip_val
                        dec_linear_idx += 1
                        continue
                dec_linear_idx += 1
            dec_h = module(dec_h)

        return dec_h


class GatingNet(torch.nn.Module):
    def __init__(self, cfg, cholesky_L=None):
        super(GatingNet, self).__init__()
        self.cfg = cfg
        self._sqrt_2 = math.sqrt(2)
        self.sigma = 0.5
        # ✅ SEFS 温度参数注册为 buffer（可动态更新）
        init_sefs_tau = getattr(cfg, 'sefs_tau', 0.5)
        self.register_buffer('sefs_tau', torch.tensor(init_sefs_tau))
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
                tau = self.sefs_tau  # 使用动态可调节的 sefs_tau

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

    def set_sefs_tau(self, tau):
        """动态设置 SEFS 温度参数（用于退火调度）"""
        self.sefs_tau.fill_(tau)

    def regularization(self, mu, reduction_func=torch.mean):
        return reduction_func(0.5 - 0.5 * torch.erf((-1 / 2 - mu) / self._sqrt_2))

    def get_gates(self, x):
        with torch.no_grad():
            gates = self.hard_sigmoid(self.local_gates(x))
        return gates

    def num_open_gates(self, x, ):
        return self.get_gates(x).sum(dim=1).cpu().median(dim=0)[0].item()


class SEFSTemperatureScheduler:
    """
    SEFS 温度退火调度器。

    在预训练阶段将 sefs_tau 从高值退火到低值：
    - 高 tau（如 3.0）：软特征选择，给所有特征公平的竞争机会
    - 低 tau（如 0.3）：硬特征选择，相关特征间竞争激烈

    退火策略：余弦退火
        tau(epoch) = tau_end + 0.5*(tau_start - tau_end)*(1 + cos(π*epoch/total_epochs))

    使用方式：
        scheduler = SEFSTemperatureScheduler(gating_net, cfg)
        # 在每个 training_step 开头调用：
        tau = scheduler.step(current_epoch)
    """
    def __init__(self, gating_net, cfg):
        self.gating_net = gating_net
        self.tau_start = cfg.get('sefs_tau_start', 3.0)
        self.tau_end = cfg.get('sefs_tau_end', cfg.get('sefs_tau', 0.5))
        self.total_epochs = cfg.ae_pretrain_epochs
        self.enabled = cfg.get('use_sefs_annealing', False)

    def step(self, epoch):
        if not self.enabled or epoch > self.total_epochs:
            return self.tau_end

        # 余弦退火
        progress = epoch / max(self.total_epochs, 1)
        tau = self.tau_end + 0.5 * (self.tau_start - self.tau_end) * (
            1.0 + math.cos(progress * math.pi)
        )
        self.gating_net.set_sefs_tau(tau)
        return tau