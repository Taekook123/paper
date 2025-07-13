import torch
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries


def masks_to_sdfs(masks: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    将一批离散的分割掩码转换为多通道的符号距离场 (SDF).
    该实现参考了用户提供的 compute_sdf 函数逻辑。

    Args:
        masks (torch.Tensor): 批处理的分割掩码, 形状为 [B, H, W], 在CPU上.
        num_classes (int): 类别的总数 (包括背景).

    Returns:
        torch.Tensor: 生成的SDF, 形状为 [B, C, H, W] (C=num_classes-1), 值在[-1, 1]之间.
    """
    # 将输入的PyTorch Tensor转换为NumPy数组
    masks_np = masks.cpu().numpy()
    batch_size, H, W = masks_np.shape
    # 输出的通道数是前景类别数 (num_classes - 1)
    sdfs = np.zeros((batch_size, num_classes - 1, H, W), dtype=np.float32)

    # 遍历批次中的每一张掩码
    for i in range(batch_size):
        mask = masks_np[i]
        # 遍历每一个前景类别 (通常跳过背景类0)
        for c in range(1, num_classes):
            # 创建当前类别的二值掩码
            posmask = (mask == c)

            # 如果该类别不存在于图像中, 则SDF为全负值 (表示完全在外部)
            if not posmask.any():
                sdfs[i, c - 1, :, :] = -1.0
                continue

            negmask = ~posmask

            # 1. 计算距离变换
            posdis = distance_transform_edt(posmask)
            negdis = distance_transform_edt(negmask)

            # 2. 独立归一化内部和外部距离到 [0, 1]
            # 添加 1e-8 以防止分母为零
            posdis_normalized = (posdis - np.min(posdis)) / (np.max(posdis) - np.min(posdis) + 1e-8)
            negdis_normalized = (negdis - np.min(negdis)) / (np.max(negdis) - np.min(negdis) + 1e-8)

            # 3. 计算SDF (内部为负, 外部为正)
            sdf = negdis_normalized - posdis_normalized

            # 4. 显式地将边界设置为0
            boundary = find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf[boundary == 1] = 0

            sdfs[i, c - 1, :, :] = sdf

    # 将NumPy数组转回PyTorch Tensor
    sdfs_tensor = torch.from_numpy(sdfs)

    return sdfs_tensor