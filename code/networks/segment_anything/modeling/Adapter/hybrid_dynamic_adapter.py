import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridBranchAttention(nn.Module):
    def __init__(self, d_model=768, d_k=64, d_v=64, bias: bool = False):
        """
        参数:
        d_model (int): 输入特征 T_t 和 T_c 的维度，也是最终输出 T_h 的维度。
        d_k (int): 查询 (Query) 和 键 (Key) 的投影维度。
        d_v (int): 值 (Value) 的投影维度。
        bias (bool): 线性投影层是否使用偏置项。默认为 False，
                     因为原始 Transformer 中的 Q, K, V 投影通常不使用偏置，
                     但输出投影 W_o 可能会使用。
        """
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        # 线性投影层
        self.W_q = nn.Linear(d_model, d_k, bias=bias) # T_t W^q
        self.W_k = nn.Linear(d_model, d_k, bias=bias) # T_c W^k
        self.W_v = nn.Linear(d_model, d_v, bias=bias) # T_c W^v

        # 输出投影层，将 d_v 维度的注意力上下文映射回 d_model 维度
        self.W_o = nn.Linear(d_v, d_model, bias=bias)

        # 可学习的平衡向量 gamma，形状为 (d_model,)，初始化为全0
        self.gamma = nn.Parameter(torch.zeros(d_model))

        # 缩放因子
        self.scale_factor = math.sqrt(d_k)

        self.AvgPool1d = nn.AvgPool1d(kernel_size=8, stride=8)

    def forward(self, T_t: torch.Tensor, T_c: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        参数:
        T_t (torch.Tensor): 目标序列张量，形状 (batch_size, seq_len_t, d_model)
        T_c (torch.Tensor): 上下文序列张量，形状 (batch_size, seq_len_c, d_model)
                           在自注意力中，T_t 和 T_c 可以是同一个张量。

        返回:
        T_n (torch.Tensor): 更新后的目标序列表示，形状 (batch_size, seq_len_t, d_model)
        """
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # T_t = T_t.to(device)
        # self.W_q = self.W_q.to(device)
        # self.W_k = self.W_k.to(device)
        # self.W_v = self.W_v.to(device)
        # self.W_o = self.W_o.to(device)


        B, H, W, C = T_t.shape
        T_t = T_t.reshape(B, H * W, C)

        # 1. 调整维度: (B, L_in, D) -> (B, D, L_in)
        #    例如 (4, 4096, 768) -> (4, 768, 4096)
        T_t = T_t.permute(0, 2, 1)
        # print(f"Permute后的张量形状 (用于nn.Pool1d): {T_t.shape}")

        # 2. 定义池化层并执行操作
        # 平均池化
        avg_pool_layer = self.AvgPool1d
        T_t_permuted = avg_pool_layer(T_t)  # 输出形状 (B, D, L_out)
        # print(f"nn.AvgPool1d 输出形状 (permuted): {T_t_permuted.shape}")

        # # 最大池化
        # max_pool_layer = nn.MaxPool1d(kernel_size=kernel_size, stride=kernel_size)
        # max_pooled_permuted = max_pool_layer(input_permuted)  # 输出形状 (B, D, L_out)
        # print(f"nn.MaxPool1d 输出形状 (permuted): {max_pooled_permuted.shape}")

        # 3. 调整维度回来: (B, D, L_out) -> (B, L_out, D)
        T_t = T_t_permuted.permute(0, 2, 1)
        # print(f"方法二 平均池化后形状 (恢复): {T_t.shape}")
        # assert T_t.shape == (B, 512, 768)

        # max_pooled_output_nn = max_pooled_permuted.permute(0, 2, 1)
        # print(f"方法二 最大池化后形状 (恢复): {max_pooled_output_nn.shape}")
        # assert max_pooled_output_nn.shape == (B, L_out, D)

        T_c = T_c.permute(0, 2, 1)
        # print(f"Permute后的张量形状 (用于nn.Pool1d): {T_c.shape}")
        avg_pool_layer = self.AvgPool1d
        T_c_permuted = avg_pool_layer(T_c)  # 输出形状 (B, D, L_out)
        # print(f"nn.AvgPool1d 输出形状 (permuted): {T_c_permuted.shape}")
        T_c = T_c_permuted.permute(0, 2, 1)
        # print(f"方法二 平均池化后形状 (恢复): {T_c.shape}")
        # assert T_c.shape == (B, 512, 768)

        batch_size, seq_len_t, _ = T_t.shape
        _, seq_len_c, _ = T_c.shape

        # 1. 投影得到 Q, K, V
        Q = self.W_q(T_t)  # (batch_size, seq_len_t, d_k)
        K = self.W_k(T_c)  # (batch_size, seq_len_c, d_k)
        V = self.W_v(T_c)  # (batch_size, seq_len_c, d_v)

        # 2. 计算注意力分数: Q * K^T / sqrt(d_k)
        # K.transpose(-2, -1) 交换最后两个维度: (batch_size, d_k, seq_len_c)
        # scores 形状: (batch_size, seq_len_t, seq_len_c)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale_factor

        # 3. 应用 Softmax 得到注意力权重
        # Softmax 在最后一个维度 (seq_len_c) 上进行，表示对于 T_t 中的每个查询，
        # T_c 中所有键的权重分布。
        attention_weights = F.softmax(scores, dim=-1) # (batch_size, seq_len_t, seq_len_c)

        # 4. 用注意力权重加权 V (值)
        # (batch_size, seq_len_t, seq_len_c) @ (batch_size, seq_len_c, d_v)
        # -> (batch_size, seq_len_t, d_v)
        attention_context = torch.matmul(attention_weights, V) # 这对应公式 (2) 中的 Attention(T_t, T_c)

        # 5. 将注意力上下文通过输出投影层 W_o
        # (batch_size, seq_len_t, d_v) -> (batch_size, seq_len_t, d_model)
        projected_attention_context = self.W_o(attention_context)

        # 6. 计算 T_h: T_t + gamma * projected_attention_context (公式 3)
        # self.gamma (形状 d_model) 会被广播以匹配 projected_attention_context 的形状
        # (1, 1, d_model) * (batch_size, seq_len_t, d_model)

        T_h = T_t + self.gamma * projected_attention_context

        repeat_factor = 8  # 4096 // 512 = 8
        # === 使用 torch.repeat_interleave 进行上采样 ===
        # torch.repeat_interleave 可以在指定维度上重复张量的元素。
        # - repeats: 指定每个元素重复的次数。
        # - dim: 指定在哪个维度上进行重复操作。对于 (B, SeqLen, Dim)，序列长度维度是 1。
        T_h = T_h.repeat_interleave(repeats=repeat_factor, dim=1)
        # print(f"上采样后的张量形状: {T_h.shape}")
        # assert T_h.shape == (B, 4096, 768)

        T_h = T_h.reshape(B, H, W, C)

        return T_h

class Hybrid_Dynamic_Adapter(nn.Module):
    def __init__(self,
                 d_model=768,
                 out_dim=None,
                 bottleneck=64,
                ):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck

        #vit tokens
        self.scale_t = nn.Linear(self.n_embd, 1)
        self.down_proj_d1 = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func_t = nn.GELU()

        #hybrid tokens
        self.scale_h = nn.Linear(self.n_embd, 1)
        self.down_proj_d2 = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func_h = nn.GELU()

        if out_dim is None:
            self.up_proj_u1 = nn.Linear(self.down_size, self.n_embd)
            self.up_proj_u2 = nn.Linear(self.down_size, self.n_embd)
        else:
            self.up_proj_u1 = nn.Linear(self.down_size, out_dim)
            self.up_proj_u2 = nn.Linear(self.down_size, out_dim)


        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj_d1.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj_u1.weight)
            nn.init.zeros_(self.down_proj_d1.bias)
            nn.init.zeros_(self.up_proj_u1.bias)
            nn.init.kaiming_uniform_(self.scale_t.weight, a=math.sqrt(5))
            nn.init.zeros_(self.scale_t.bias)

            nn.init.kaiming_uniform_(self.down_proj_d2.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj_u2.weight)
            nn.init.zeros_(self.down_proj_d2.bias)
            nn.init.zeros_(self.up_proj_u2.bias)
            nn.init.kaiming_uniform_(self.scale_h.weight, a=math.sqrt(5))
            nn.init.zeros_(self.scale_h.bias)

    def forward(self, x, h_token, residual=None):
        #vit tokens
        scale_t = F.relu(self.scale_t(x))
        down_t = self.down_proj_d1(x)
        down_t = self.non_linear_func_t(down_t)
        T_m = self.up_proj_u1(down_t)
        T_m = T_m * scale_t

        # hybrid tokens
        scale_h = F.relu(self.scale_h(h_token))
        down_h = self.down_proj_d2(h_token)
        down_h = self.non_linear_func_h(down_h)
        T_n = self.up_proj_u2(down_h)
        T_n = T_n * scale_h

        T = T_m + T_n
        return T

# if __name__ == '__main__':
#     model = HybridBranchAttention()
#     T_t = torch.randn(1, 4096, 768)
#     print(T_t.shape)
#     T_c = torch.randn(1, 4096, 768)
#     T_h = model(T_t, T_c)
#     print(T_h.shape)
