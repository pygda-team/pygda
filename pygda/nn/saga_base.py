"""Author SAGA core; class rename and import cleanup only.

Source: f1shungry/SAGA, commit 55ff00a4e426a2d0fbeb913f11330ab69b68a506.
See model documentation for the public-code reproduction scope.
"""
# MIT License
# 
# Copyright (c) 2026 Jingyu Zhao
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 

import torch
import torch.nn as nn
import torch.nn.functional as F

class SAGABase(nn.Module):
    def __init__(
            self,
            feats_dim,
            sub_num,
            hidden_dim,
            embed_dim,
            tau,
            dropout,
            nnodes,
            dominant_index,
            device,
            nb_classes,
            config
    ):
        super(SAGABase, self).__init__()
        self.feats_dim = feats_dim
        self.embed_dim = embed_dim
        self.sub_num = sub_num
        self.tau = tau
        self.device = device
        self.dominant_index = dominant_index
        self.dropout = dropout
        self.lam_sad = 1
        self.config = config

        # -----------------------
        # 2.1) 编码 / 解码器
        # -----------------------
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feats_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

        self.decoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feats_dim)
        )

        # 投影头，用于对比学习 (MI loss)
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ELU(),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(self.sub_num)
        ])


        # -----------------------
        # 源域分类器
        # -----------------------
        self.classifier = nn.Linear(embed_dim * sub_num, nb_classes)


    # -----------------------------
    # forward 函数，核心逻辑
    # -----------------------------
    def forward(self, feat_t, f_list_t, feat_s, f_list_s, label_s, re_computing=False):
        """
        :param feat_t:    [num_target_batch, feats_dim] 或 None
        :param f_list_t:  list of [num_target_batch, feats_dim], 长度 = sub_num
        :param feat_s:    [num_source_batch, feats_dim] 或 None
        :param f_list_s:  list of [num_source_batch, feats_dim], 长度 = sub_num
        :param label_s:   [num_source_batch, nb_classes] (one-hot) 或 None
        :param re_computing: 是否更新 self.dominant_index_t / s
        :return: loss
        """

        # ===========  编码 =============
        z_list_t = None
        z_list_s = None

        if f_list_t is not None:
            z_list_t = self.encoder(f_list_t)  # list of [batch_size, embed_dim]

            dominant_index_t = self.dominant_view_mining(z_list_t, feat_t)
            ae_loss_t = self.ae_loss(z_list_t, f_list_t)
        else:
            ae_loss_t = 0.0


        # z_list_s = None
        # dominant_index_s, err_loss_s = 0, 0.0
        if f_list_s is not None:
            z_list_s = self.encoder(f_list_s)
            dominant_index_s = self.dominant_view_mining(z_list_s, feat_s)
            ae_loss_s = self.ae_loss(z_list_s, f_list_s)
        else:
            ae_loss_s = 0.0

        if re_computing:
            self.dominant_index_t = dominant_index_t
            self.dominant_index_s = dominant_index_s


        # ===========  源域分类损失 =============
        # 先把 z_list_s 拼成单个 embedding_s
        clf_loss_s = torch.tensor(0.0, device=self.device)
        embedding_s = torch.zeros(self.feats_dim, self.embed_dim * self.sub_num).to(self.device)
        embedding_t = torch.zeros(self.feats_dim, self.embed_dim * self.sub_num).to(self.device)
        clf_loss_t = torch.tensor(0.0, device=self.device)

        # z_list_s 是一个长度 = sub_num 的列表，每个元素 [batch_size, embed_dim]
        if z_list_s is not None:
            embedding_s = torch.cat(z_list_s, dim=1)  # [batch_size, embed_dim * sub_num]
            logits_s = self.classifier(embedding_s)  # [batch_size, nb_classes]
            label_indices_s = label_s.argmax(dim=-1)  # 若 label_s 是 one-hot
            # print(logits_s)
            clf_loss_s = F.cross_entropy(logits_s, label_indices_s)
        if z_list_t is not None:
            embedding_t = torch.cat(z_list_t, dim=1)  # [batch_size, embed_dim * sub_num]
            logits_t = self.classifier(embedding_t)  # [batch_size, nb_classes]
            # print(logits_t)
            clf_loss_t = torch.mean(Entropy(F.softmax(logits_t, dim=-1)))

        # ===========  交叉对齐损失=============
        ca_loss = torch.tensor(0.0, device=self.device)
        if z_list_t is not None and z_list_s is not None:
            ca_loss = self.ca_loss(z_list_t, z_list_s, dominant_index_t, dominant_index_s)

        ia_loss = torch.tensor(0.0, device=self.device)
        if z_list_t is not None and z_list_s is not None:
            ia_loss = self.ia_loss(z_list_t, z_list_s, dominant_index_t, dominant_index_s)

        # =========== 6) 最终loss 汇总=============
        weights = self.config.loss_weights

        loss = (ae_loss_s + ae_loss_t
                + weights['beta'] * clf_loss_s
                + weights['delta'] * clf_loss_t
                + (1 - weights['alpha']) * ca_loss
                + weights['alpha'] * ia_loss
                )

        return loss

    def encoder(self, x_list):
        """
        输入: x_list 是一个长度 = self.sub_num 的列表，
             每个元素形状 [batch_size, feats_dim]
        输出: z_list 同样是长度 = sub_num 的列表，
             每个元素形状 [batch_size, embed_dim]
        """
        z_list = []
        for i in range(len(x_list)):
            z = self.fc(x_list[i])  # [batch_size, embed_dim]
            z_list.append(z)
        return z_list

    def ae_loss(self, z_list, f_list):
        """
        将每个子视图的编码 z_list[i] 用 self.decoder 重建，然后与原特征 f_list[i] 做 MSE 或其他损失
        """
        loss_rec = 0.0
        for i in range(self.sub_num):
            fea_rec = self.decoder(z_list[i])  # [batch_size, feats_dim]
            loss_rec += F.mse_loss(fea_rec, f_list[i])
        loss_ae = loss_rec / self.sub_num
        return loss_ae


    def ca_loss(self, z_list_t, z_list_s, dominant_index_t, dominant_index_s):
        """
        交叉对齐损失（Cross-Alignment Loss）

        :param z_list_t: 目标域编码列表，长度=sub_num，每个元素[batch_size, embed_dim]
        :param z_list_s: 源域编码列表，长度=sub_num，每个元素[batch_size, embed_dim]
        :param dominant_index_t: 目标域的主视图索引
        :param dominant_index_s: 源域的主视图索引
        :return: Loss_CA
        """
        if z_list_t is None or z_list_s is None:
            return torch.tensor(0.0, device=self.device)

        # 获取主视图的投影嵌入
        proj_t = self.proj[dominant_index_t](z_list_t[dominant_index_t])
        proj_s = self.proj[dominant_index_s](z_list_s[dominant_index_s])

        # 计算两个方向的对比损失
        loss_s2t = self._contrastive_loss(proj_s, proj_t)
        loss_t2s = self._contrastive_loss(proj_t, proj_s)

        # 总的交叉对齐损失
        ca_loss = loss_s2t + loss_t2s

        return ca_loss

    def _contrastive_loss(self, pos_embed, neg_embed):
        """
        计算单向的对比损失
        ℓ(Z_vs^{S,ks*}, Z_vT^{T,kT*}) = -log(1/(N_T*N_S) * sum exp(cos(pos_i, neg_j)/tau))

        :param pos_embed: 正样本投影嵌入 [N_pos, embed_dim]
        :param neg_embed: 负样本投影嵌入 [N_neg, embed_dim]
        :return: 对比损失标量
        """
        # 归一化嵌入用于余弦相似度计算
        pos_embed_norm = F.normalize(pos_embed, dim=1, p=2)  # [N_pos, embed_dim]
        neg_embed_norm = F.normalize(neg_embed, dim=1, p=2)  # [N_neg, embed_dim]

        # 计算余弦相似度矩阵: [N_pos, N_neg]
        # sim[i,j] = cos(pos_i, neg_j)
        sim_matrix = torch.mm(pos_embed_norm, neg_embed_norm.t())  # [N_pos, N_neg]

        # 除以温度参数
        sim_matrix = sim_matrix / self.tau  # [N_pos, N_neg]

        # 计算指数和
        exp_sim = torch.exp(sim_matrix)  # [N_pos, N_neg]
        sum_exp = torch.sum(exp_sim, dim=1)  # [N_pos]

        # 计算对数均值
        # loss = -log(1/(N_pos*N_neg) * sum(exp(sim)))
        # = -log(sum(exp(sim))) + log(N_pos*N_neg)
        N_pos = pos_embed.size(0)
        N_neg = neg_embed.size(0)
        log_sum = torch.log(sum_exp + 1e-8)  # [N_pos]
        loss = -torch.mean(log_sum) + torch.log(torch.tensor(N_pos * N_neg, dtype=torch.float32, device=self.device))

        return loss


    def dominant_view_mining(self, z_list, feat):
        """
        通过比较每个视图的相似度图与原特征的相似度图，找出最优视图 index
        并返回该 index 与 视图平均误差 (err_loss)
        """
        if z_list is None or feat is None:
            return 0, 0.0

        err_list = []
        feat = F.normalize(feat, dim=1, p=2)
        z_list = [F.normalize(z, dim=1, p=2) for z in z_list]
        feat_sim = torch.mm(feat, feat.t())  # [batch_size, batch_size]

        for i in range(len(z_list)):
            embed = z_list[i]
            z_sim = torch.mm(embed, embed.t())
            err = F.mse_loss(z_sim, feat_sim)
            err_list.append(err)

        # 找到最小 MSE 的视图
        dominant_index = torch.argmin(torch.tensor(err_list))
        err_loss = sum(err_list) / len(err_list)

        return dominant_index

    def get_embeds(self, f_list):
        """
        推理阶段可用：将若干视图特征编码后拼接并归一化
        """
        z_list = self.encoder(f_list)
        z = torch.cat(z_list, dim=1)
        z = F.normalize(z, dim=1, p=2)
        return z.detach()

    def compute_sad(self, z_list_t, z_list_s):
        """
        计算 Structure-Aware Discrepancy (SAD)
        SAD_v^k = ||Z_v^{T,k_s}(Z_v^{T,k_s})^T - Z_v^{S,k_T}(Z_v^{S,k_T})^T||_F^2

        :param z_list_t: 目标域编码列表 [sub_num, [batch_size_t, embed_dim]]
        :param z_list_s: 源域编码列表 [sub_num, [batch_size_s, embed_dim]]
        :return: sad_v_k [sub_num]
        """
        if z_list_t is None or z_list_s is None:
            return None

        sad_v_k = []

        for v in range(self.sub_num):
            # Cov_t = Z_t^T @ Z_t / batch_size_t
            cov_t = torch.mm(z_list_t[v].t(), z_list_t[v]) / z_list_t[v].size(0)
            cov_s = torch.mm(z_list_s[v].t(), z_list_s[v]) / z_list_s[v].size(0)

            # 计算 Frobenius norm 的平方
            sad = torch.norm(cov_t - cov_s, p='fro') ** 2
            sad_v_k.append(sad)

        return torch.stack(sad_v_k)  # [sub_num]
    def compute_sad_weights(self, sad_v_k_t, sad_v_k_s):
        """
        计算 SAD 引导的权重
        ω_{v,k} = exp(-λ * SAD_v^k) / Σ exp(-λ * SAD_i^j)

        :param sad_v_k_t: 目标域SAD [sub_num]
        :param sad_v_k_s: 源域SAD [sub_num]
        :return: omega_t [sub_num], omega_s [sub_num]
        """
        # 目标域权重
        exp_t = torch.exp(-self.lam_sad * sad_v_k_t)
        omega_t = exp_t / (torch.sum(exp_t) + 1e-8)

        # 源域权重
        exp_s = torch.exp(-self.lam_sad * sad_v_k_s)
        omega_s = exp_s / (torch.sum(exp_s) + 1e-8)

        return omega_t, omega_s

    def ia_loss(self, z_list_t, z_list_s, dominant_index_t, dominant_index_s):
        """
        Intra-alignment Loss
        """
        if z_list_t is None or z_list_s is None:
            return torch.tensor(0.0, device=self.device)

        # 计算SAD
        sad_v_k_t = self.compute_sad(z_list_t, z_list_s)
        sad_v_k_s = self.compute_sad(z_list_s, z_list_t)

        # 计算权重
        omega_t, omega_s = self.compute_sad_weights(sad_v_k_t, sad_v_k_s)

        batch_size_t = z_list_t[0].size(0)
        batch_size_s = z_list_s[0].size(0)

        # 目标域损失
        ia_loss_t = torch.tensor(0.0, device=self.device)
        cov_hop1_t = torch.mm(z_list_t[0].t(), z_list_t[0]) / batch_size_t

        for v in range(self.sub_num):
            cov_v_t = torch.mm(z_list_t[v].t(), z_list_t[v]) / batch_size_t
            cov_diff_t = torch.norm(cov_v_t - cov_hop1_t, p='fro') ** 2
            ia_loss_t += omega_t[v] * cov_diff_t

        ia_loss_t = ia_loss_t / self.sub_num

        # 源域损失
        ia_loss_s = torch.tensor(0.0, device=self.device)
        cov_hop1_s = torch.mm(z_list_s[0].t(), z_list_s[0]) / batch_size_s

        for v in range(self.sub_num):
            cov_v_s = torch.mm(z_list_s[v].t(), z_list_s[v]) / batch_size_s
            cov_diff_s = torch.norm(cov_v_s - cov_hop1_s, p='fro') ** 2
            ia_loss_s += omega_s[v] * cov_diff_s

        ia_loss_s = ia_loss_s / self.sub_num

        return ia_loss_t + ia_loss_s

def Entropy(input):
    epsilon = 1e-4
    entropy = -input * torch.log(input + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy






__all__ = ["SAGABase", "Entropy"]
