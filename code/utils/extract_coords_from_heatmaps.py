import torch
import torch.nn.functional as F


def extract_coords_from_heatmaps(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    从一批热图中提取每个通道的峰值坐标.

    Args:
        heatmaps (torch.Tensor): 批处理的热图, 形状为 [B, C, H, W].

    Returns:
        torch.Tensor: 提取出的坐标, 形状为 [B, C, 2], 最后一维是 (y, x) 坐标.
    """
    batch_size, num_channels, h, w = heatmaps.shape

    # 将空间维度(H, W)展平
    heatmaps_flat = heatmaps.view(batch_size, num_channels, -1)

    # 寻找每个通道中最大值的索引
    max_indices = torch.argmax(heatmaps_flat, dim=2)

    # 将一维索引转换回二维坐标(y, x)
    coords_y = max_indices // w
    coords_x = max_indices % w

    # 将y和x坐标堆叠起来
    coords = torch.stack([coords_y, coords_x], dim=2)

    return coords.float()