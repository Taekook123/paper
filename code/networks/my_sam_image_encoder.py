from segment_anything import sam_model_registry
from segment_anything.modeling import Sam  # 用于类型提示或直接实例化

import torch
import torch.nn as nn

# import torch.nn.functional as F # 如果 SAM 模型内部需要，会自动导入。如果你的脚本不直接用，可注释。
# from torch import Tensor # 同上

from segment_anything import sam_model_registry
from segment_anything.modeling import Sam, ImageEncoderViT  # 确保 ImageEncoderViT 也被导入


# (你项目中加载 SAM 模型的相关代码，例如 build_sam.py 中的 _build_sam 等)
# 假设你有一个函数来获取配置好的 SAM ImageEncoderViT 实例和其预处理参数
def load_and_freeze_sam_image_encoder(checkpoint_path="/hd_2t/hd_4t/wt/SAM_Hydas/code/networks/sam_vit_b_01ec64.pth",
                                      model_type="vit_b", image_size=1024):
    """
    加载 SAM 模型，提取、冻结其图像编码器，并返回编码器及预处理参数。
    """
    # 使用 sam_model_registry 加载完整的 SAM 模型
    # 注意：sam_model_registry 返回 (model, image_embedding_size)
    # 我们这里需要模型本身来获取预处理参数和图像编码器
    sam_model_full, _ = sam_model_registry[model_type](
        image_size=image_size,
        # num_classes 是为了 MaskDecoder，对于 ImageEncoder 本身不直接使用，但 build 函数需要
        num_classes=4,  # 可以是任意值，因为我们只取 image_encoder
        checkpoint=checkpoint_path
    )
    print(f"原始 SAM 模型 ({model_type}) 已加载。")

    sam_image_encoder = sam_model_full.image_encoder

    # 获取预处理参数 (需要注册为 buffer 才能在不同设备上正确工作)
    # sam_model_full 内部已经将 pixel_mean 和 pixel_std 注册为 buffer
    pixel_mean = sam_model_full.pixel_mean
    pixel_std = sam_model_full.pixel_std

    # 冻结图像编码器的参数
    # for param in sam_image_encoder.parameters():
    #     param.requires_grad = False
    # print("SAM 图像编码器的参数已冻结。")
    for n, value in sam_image_encoder.named_parameters():
        if "hda" not in n and "hybrid_branch_att" not in n and "original_adapter" not in n:
            value.requires_grad = False

    return sam_image_encoder, pixel_mean, pixel_std, sam_model_full.image_encoder.img_size

#
# if __name__ == "__main__":
#     # 1. 加载 SAM 模型
#     # sam_model_registry 会返回一个元组 (model_object, image_embedding_size)
#     # 我们需要解包以获取模型对象。
#     # checkpoint 参数会加载 "sam_vit_b_01ec64.pth" 中的预训练权重。
#     sam_model_object, _ = sam_model_registry["vit_b"](
#         image_size=1024,
#         num_classes=4,  # 此参数主要影响 MaskDecoder，对于仅使用 ImageEncoder 可能不那么重要，但构建模型时需要
#         checkpoint="sam_vit_b_01ec64.pth"  # 确保这个路径是正确的 SAM ViT-B 预训练权重文件
#     )
#     print(f"原始 SAM 模型已加载: {type(sam_model_object)}")
#
#     # 2. 冻结图像编码器的参数
#     # 检查模型是否有 image_encoder 属性
#     if hasattr(sam_model_object, 'image_encoder') and sam_model_object.image_encoder is not None:
#         for param in sam_model_object.image_encoder.parameters():
#             param.requires_grad = False
#         print("SAM 图像编码器的参数已冻结。")
#
#     else:
#         print("错误: SAM 模型对象中没有找到 'image_encoder'。")
#         exit()  # 如果没有图像编码器，后续操作无意义
#
#     # 3. 使用冻结的图像编码器 (示例)
#     # 将模型移至合适的设备 (GPU 或 CPU)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     sam_model_object.to(device)
#
#     # 如果进行推理，通常会设置模型为评估模式
#     sam_model_object.eval()
#
#     # 创建一个随机输入张量 (例如: 1张图片, 3通道, 1024x1024 尺寸)
#     dummy_input = torch.rand(size=(1, 3, 1024, 1024)).to(device)
#
#     print(f"输入张量形状: {dummy_input.shape}, 设备: {dummy_input.device}")
#
#     # 通过图像编码器传递输入
#     # 使用 torch.no_grad() 上下文管理器，确保在前向传播时不会计算梯度
#     with torch.no_grad():
#         try:
#             image_embeddings = sam_model_object.image_encoder(dummy_input)
#             print(f"图像编码器输出的嵌入形状: {image_embeddings.shape}")
#         except Exception as e:
#             print(f"运行图像编码器时出错: {e}")