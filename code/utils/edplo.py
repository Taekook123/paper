import numpy as np

# 全局常量
NUM_CLASSES = 4  # 3个前景类别, 1个背景类别
EPS = 1e-8  # 用于数值稳定性的小epsilon (避免log(0)或除以零)


# --- 辅助函数 ---

def _calculate_entropy_single_pixel(prob_dist_pixel):
    """
    计算单个像素概率分布的熵。
    对应公式 9: H(p_i) = - sum_c (p_i(c) * log(p_i(c)))

    参数:
        prob_dist_pixel (np.ndarray): 单个像素的概率分布，形状为 (C,)。
                                      假设总和为1。
    返回:
        float: 像素的熵值。
    """
    # 在取对数之前过滤掉零概率，以避免 -inf * 0 = nan 的问题
    # 如果 p 为 0，则 p log p 为 0。
    valid_probs = prob_dist_pixel[prob_dist_pixel > EPS]
    if len(valid_probs) == 0:
        return 0.0  # 如果所有概率都有效地为零或其中一个为1.0，则熵为0。

    log_probs = np.log(valid_probs + EPS)  # 在log内部添加EPS以确保安全
    entropy = -np.sum(valid_probs * log_probs)
    return entropy


def calculate_pixel_wise_entropy(prob_map):
    """
    计算概率图中每个像素的熵。

    参数:
        prob_map (np.ndarray): 来自教师模型的概率图，形状为 (H, W, C)。
                               每个像素的分布 p_i 应总和为1。
    返回:
        np.ndarray: 熵图，形状为 (H, W)。
    """
    # 对每个像素沿类别维度应用单个像素熵计算
    entropy_map = np.apply_along_axis(_calculate_entropy_single_pixel, axis=-1, arr=prob_map)
    return entropy_map


def _normalize_probabilities_single_pixel(prob_dist_pixel):
    """
    归一化单个像素的概率分布，使其总和为1。

    参数:
        prob_dist_pixel (np.ndarray): 单个像素的可能未归一化的概率分布，
                                      形状为 (C,)。
    返回:
        np.ndarray: 归一化后的概率分布，形状为 (C,)。
    """
    sum_probs = np.sum(prob_dist_pixel)
    if sum_probs < EPS:  # 如果在过滤/加权后所有概率都变为零或非常小
        return np.zeros_like(prob_dist_pixel)  # 与公式中暗示输出为0一致
    return prob_dist_pixel / sum_probs


def normalize_pixel_wise_probabilities(prob_map):
    """
    归一化图中每个像素的概率分布。

    参数:
        prob_map (np.ndarray): 概率图，形状为 (H, W, C)，其中像素分布
                               在中间步骤后可能不总和为1。
    返回:
        np.ndarray: 归一化后的概率图，形状为 (H, W, C)。
    """
    normalized_map = np.apply_along_axis(_normalize_probabilities_single_pixel, axis=-1, arr=prob_map)
    return normalized_map


# --- CEAF (分类熵自适应过滤) 策略 ---

def ceaf_filter(prob_map, current_epoch, max_total_epochs, alpha_0_per_class, num_classes_arg, precomputed_max_Hc=None):
    """
    实现分类熵自适应过滤 (CEAF) 策略。
    输出经过CEAF优化的软伪标签目标 和 原始像素熵图。

    参数:
        prob_map (np.ndarray): 教师模型的概率图，形状为 (H, W, C)。
        current_epoch (int): 当前训练轮次 (k)。
        max_total_epochs (int): 用于alpha退火的最大训练轮次数。
        alpha_0_per_class (np.ndarray): 每个类别的初始过滤比例，形状为 (C,)。
        precomputed_max_Hc (np.ndarray, 可选): 预先计算的被预测为类别c的像素的最大熵
                                                (max(H_c))。形状为 (C,)。
                                                如果为None，则会从当前prob_map计算。
    返回:
        tuple: (ceaf_output_prob_map, pixel_entropy_map)
            - ceaf_output_prob_map (np.ndarray): 过滤并归一化后的概率图，形状为 (H, W, C)。
            - pixel_entropy_map (np.ndarray): 计算得到的每个像素的H(p_i)，形状为 (H,W)。
    """
    H, W, C = prob_map.shape
    if C != num_classes_arg:
        raise ValueError(f"概率图中的类别数 ({C}) 与 num_classes_arg ({num_classes_arg}) 不匹配")
    if len(alpha_0_per_class) != num_classes_arg:
        raise ValueError(f"alpha_0_per_class 的长度 ({len(alpha_0_per_class)}) 必须与 num_classes_arg ({num_classes_arg}) 匹配。")

    # 1. 计算每个像素的像素级熵 H(p_i) (公式 9)
    pixel_entropy_map = calculate_pixel_wise_entropy(prob_map)  # 形状 (H, W)

    # 2. 计算每个类别的动态过滤阈值 T_k^c (公式 10)
    # alpha_k^c 的退火计划: alpha_k^c = alpha_0^c * (1 - k / max_epochs)
    # 确保 k <= max_epochs 以进行有意义的退火
    annealing_factor = 1.0 - (current_epoch / max_total_epochs) if max_total_epochs > 0 else 0.0
    annealing_factor = max(0.0, annealing_factor)  # 裁剪为非负值

    alpha_k_per_class = alpha_0_per_class * annealing_factor  # 形状 (C,)
    alpha_k_per_class = np.clip(alpha_k_per_class, 0.0, 1.0)  # 确保其在 [0,1] 范围内

    # 确定每个类别的 max(H_c)
    if precomputed_max_Hc is None:
        # 从当前 prob_map 估计 max(H_c)。
        # max(H_c) 是被教师模型预测为类别 c 的像素的最大熵。
        predicted_classes_map = np.argmax(prob_map, axis=-1)  # 形状 (H, W)
        max_Hc_values = np.zeros(num_classes_arg)
        for c_idx in range(num_classes_arg):
            pixels_predicted_as_class_c_mask = (predicted_classes_map == c_idx)
            if np.any(pixels_predicted_as_class_c_mask):
                max_Hc_values[c_idx] = np.max(pixel_entropy_map[pixels_predicted_as_class_c_mask])
            else:
                # 如果该图中没有任何像素被预测为类别 c 的回退策略。
                # 使用当前图的全局最大熵，或者如果图没有熵则使用理论最大值。
                current_map_max_entropy = np.max(pixel_entropy_map)
                if current_map_max_entropy > EPS:
                    max_Hc_values[c_idx] = current_map_max_entropy
                else:  # 如果图中所有熵都约为0 (例如，one-hot 输入)
                    max_Hc_values[c_idx] = np.log(num_classes_arg) if num_classes_arg > 1 else EPS
        if np.any(max_Hc_values < EPS):  # 确保 max_Hc_values 不为零，以避免 T_k_c 始终为零
            max_Hc_values[max_Hc_values < EPS] = EPS  # 设置一个小的下限
    else:
        max_Hc_values = precomputed_max_Hc

    if len(max_Hc_values) != num_classes_arg:
        raise ValueError(f"max_Hc_values 的长度 ({len(max_Hc_values)}) 必须与 num_classes_arg ({num_classes_arg}) 匹配。")

    # 计算 T_k^c = alpha_k^c * max(H_c)
    T_k_c_thresholds = alpha_k_per_class * max_Hc_values  # 形状 (C,)

    # 3. 根据阈值更新概率掩码 (实现公式 11 的逻辑)
    # p_bar_i(c) = p_i(c) if H(p_i) <= T_k^c_value (对应类别的值), else 0.
    # H(p_i) 是像素的总熵。T_k^c 是特定于类别的。

    p_bar_intermediate_map = np.zeros_like(prob_map)  # 形状 (H, W, C)
    for r_idx in range(H):
        for c_idx_img in range(W):  # c_idx_img 表示列索引，以避免与类别索引混淆
            current_pixel_total_entropy = pixel_entropy_map[r_idx, c_idx_img]
            for class_label_idx in range(num_classes_arg):
                if current_pixel_total_entropy <= T_k_c_thresholds[class_label_idx]:
                    p_bar_intermediate_map[r_idx, c_idx_img, class_label_idx] = prob_map[
                        r_idx, c_idx_img, class_label_idx]
                # 否则它保持为0 (由于 np.zeros_like 初始化)

    # 4. 归一化更新后的概率掩码 (实现公式 12)
    # 归一化是逐像素的：对 p_bar_i(c) 的 c 求和。
    # normalize_pixel_wise_probabilities 处理总和为零的情况。
    ceaf_output_prob_map = normalize_pixel_wise_probabilities(p_bar_intermediate_map)

    return ceaf_output_prob_map, pixel_entropy_map


# --- PEWO (仅计算权重) 策略 ---
def pewo_calculate_weights(original_pixel_entropy_map):
    """
    实现PEWO策略的核心部分：计算像素权重 w_i (公式 13)。
    这些权重后续将用于加权损失函数。

    参数:
        original_pixel_entropy_map (np.ndarray): 原始像素熵图 H(p_i)，形状为 (H, W)。
                                                 通常由CEAF步骤计算并传递过来。
    返回:
        np.ndarray: PEWO计算的像素置信度权重 w_i，形状为 (H, W)。
    """
    # 计算像素权重 w_i (公式 13)
    # w_i = 1 - H(p_i) / max(H_image)，其中 max(H_image) 是当前图像中的最大熵。
    max_H_in_image = np.max(original_pixel_entropy_map)

    pixel_confidence_weights = np.zeros_like(original_pixel_entropy_map)  # 形状 (H,W)
    if max_H_in_image > EPS:  # 如果图像中所有熵都为零，则避免除以零
        pixel_confidence_weights = 1.0 - (original_pixel_entropy_map / max_H_in_image)
    else:  # 如果最大熵约为0，则所有熵都约为0，意味着高置信度，因此权重应为1。
        pixel_confidence_weights = np.ones_like(original_pixel_entropy_map)

    pixel_confidence_weights = np.clip(pixel_confidence_weights, 0.0, 1.0)  # 确保权重在 [0,1] 范围内

    return pixel_confidence_weights


# --- EDPLO 主函数 ---
def edplo_strategy(teacher_prob_map, current_epoch, max_total_epochs, alpha_0_per_class,
                   num_classes_arg,precomputed_max_Hc_for_ceaf=None):
    """
    实现完整的基于熵的双重伪标签优化 (EDPLO) 策略。该策略输出由CEAF优化的软伪标签目标，以及由PEWO计算的像素级置信度权。这些权重用于在学生模型训练时加权损失。

    参数:
        teacher_prob_map (np.ndarray): 教师模型的概率图，形状为 (H, W, C)。
        current_epoch (int): 当前训练轮次 (k)。
        max_total_epochs (int): 用于 CEAF 中 alpha 退火的最大训练轮次数。
        alpha_0_per_class (np.ndarray or list): 每个类别用于 CEAF 的初始过滤比例因子。形状为 (C,)。
        precomputed_max_Hc_for_ceaf (np.ndarray, 可选): 各类别的预计算最大熵值 (max(H_c))。形状为 (C,)。
    返回:
        tuple: (soft_pseudo_label_targets, pixel_confidence_weights)
            - soft_pseudo_label_targets (np.ndarray): CEAF优化后的软伪标签目标，形状为 (H, W, C)。
            - pixel_confidence_weights (np.ndarray): PEWO计算的像素置信度权重，形状为 (H, W)。
    """
    if teacher_prob_map.shape[-1] != NUM_CLASSES:
        raise ValueError(f"输入 teacher_prob_map 有 {teacher_prob_map.shape[-1]} 个类别, 期望 {NUM_CLASSES} 个。")

    alpha_0_per_class = np.array(alpha_0_per_class)
    if len(alpha_0_per_class) != NUM_CLASSES:
         raise ValueError(f"alpha_0_per_class 长度 {len(alpha_0_per_class)} 与 NUM_CLASSES {NUM_CLASSES} 不匹配。")

    # --- 步骤 1: CEAF 生成软伪标签目标 ---
    # ceaf_filter 返回CEAF优化的软目标 和 原始逐像素熵图
    soft_pseudo_label_targets, original_pixel_entropy_map = ceaf_filter(
        prob_map=teacher_prob_map,
        current_epoch=current_epoch,
        max_total_epochs=max_total_epochs,
        alpha_0_per_class=alpha_0_per_class,
        num_classes_arg=num_classes_arg,
        precomputed_max_Hc=precomputed_max_Hc_for_ceaf
    )

    # --- 步骤 2: PEWO 计算置信度权重 ---
    # 使用在CEAF步骤中计算得到的原始像素熵图
    pixel_confidence_weights = pewo_calculate_weights(
        original_pixel_entropy_map=original_pixel_entropy_map
    )

    return soft_pseudo_label_targets, pixel_confidence_weights