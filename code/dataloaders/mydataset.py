import torch
import random
import numpy as np
from torch.utils.data import Dataset
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class BaseDataSets(Dataset):
    def __init__(
            self,
            base_dir=None,
            split="train",
            num=None,
            transform=None,
            ops_weak=None,
            ops_strong=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong
        self.trans_512 = transforms.Compose([
            transforms.Resize((512, 512), interpolation=InterpolationMode.NEAREST)
        ])
        self.trans_512_continuous = transforms.Compose([
            transforms.Resize((512, 512), interpolation=InterpolationMode.BILINEAR)
        ])
        self.trans_256 = transforms.Compose([
            transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST)
        ])
        self.dataset = base_dir.split('/')[2]

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        if self.split == "train":
            with open(self._base_dir + "/train.txt", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(self._base_dir + "/val.txt", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "test":
            with open(self._base_dir + "/test.txt", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        # 兼容带后缀和不带后缀的文件名列表
        if "." in case:
            case = case.split(".")[0]

        # --- 路径构建 ---
        if self.dataset == "AIDK":  # 假设AIDK数据集的路径规则不同
            img_path = self._base_dir + "/images/{}.bmp".format(case)
            msk_path = self._base_dir + "/labels/{}.png".format(case)
        else:
            img_path = self._base_dir + "/images/{}.jpg".format(case)
            msk_path = self._base_dir + "/labels/label_{}.png".format(case)

        heatmap_path = self._base_dir + "/heatmaps/{}.npy".format(case)
        sdm_path = self._base_dir + "/SDMs/{}_sdf.npy".format(case)

        # --- 数据加载 ---
        image = Image.open(img_path).convert("L").resize((512, 512))
        label = self.trans_512(Image.open(msk_path).convert("L"))

        heatmap_np = np.load(heatmap_path)
        sdm_np = np.load(sdm_path)

        # # 使用转换后的数组创建 PIL 图像
        # heatmap_pil = Image.fromarray(heatmap_np)
        # sdm_pil = Image.fromarray(sdm_np)
        #
        # # 对heatmap和sdm使用双线性插值
        # heatmap = self.trans_512_continuous(heatmap_pil)
        # sdm = self.trans_512_continuous(sdm_pil)

        # 直接在NumPy数组上进行缩放，而不是转换为PIL
        # 假设heatmap_np的形状是 (C, H, W)，例如 (4, 256, 256)
        target_size = (512, 512)

        # 计算缩放因子 (不缩放通道维度)
        zoom_factors_h = (1, target_size[0] / heatmap_np.shape[1], target_size[1] / heatmap_np.shape[2])
        # 使用双线性插值 (order=1)
        heatmap = ndimage.zoom(heatmap_np, zoom_factors_h, order=1)

        # 对sdm执行同样的操作
        zoom_factors_s = (1, target_size[0] / sdm_np.shape[1], target_size[1] / sdm_np.shape[2])
        sdm = ndimage.zoom(sdm_np, zoom_factors_s, order=1)

        label = np.array(label)
        # 像素值映射关系
        label = np.where(label == 32, 1, label)
        label = np.where(label == 64, 2, label)
        label = np.where(label == 96, 3, label)
        label = np.where(label == 128, 4, label)
        label = np.where(label == 160, 5, label)
        label = np.where(label == 192, 6, label)
        label = np.where(label == 255, 1, label)
        label = Image.fromarray(label)

        sample = {
            "image": image,
            "label": label,
            "heatmap": heatmap,
            "sdm": sdm
        }

        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)
        else:  # val or test
            sample = self.transform(sample)

        sample["idx"] = idx
        sample["filename"] = case
        return sample


def random_rot_flip(image, label=None, heatmap=None, sdm=None):
    # k决定旋转90度的次数
    k = np.random.randint(0, 4)
    # axis决定翻转的轴，1为垂直翻转，2为水平翻转
    axis = np.random.randint(1, 3)

    # 对图像操作
    image = np.rot90(image, k, axes=(0, 1)) # 图像是2D(H,W)，axes=(0,1)
    image = np.flip(image, axis=axis-1).copy() # 图像是2D，轴要对应减1

    outputs = [image]

    # 对标签、热力图和SDM进行完全相同的操作
    if label is not None:
        label = np.rot90(label, k, axes=(0, 1)) # 标签是2D(H,W)，axes=(0,1)
        label = np.flip(label, axis=axis-1).copy()
        outputs.append(label)

    if heatmap is not None:
        # heatmap是3D(C,H,W)，必须指定在空间维度(1,2)上操作
        heatmap = np.rot90(heatmap, k, axes=(1, 2))
        heatmap = np.flip(heatmap, axis=axis).copy()
        outputs.append(heatmap)

    if sdm is not None:
        # sdm是3D(C,H,W)，必须指定在空间维度(1,2)上操作
        sdm = np.rot90(sdm, k, axes=(1, 2))
        sdm = np.flip(sdm, axis=axis).copy()
        outputs.append(sdm)

    return tuple(outputs)


def random_rotate(image, label, heatmap, sdm):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    heatmap = ndimage.rotate(heatmap, angle, order=0, reshape=False)
    sdm = ndimage.rotate(sdm, angle, order=0, reshape=False)
    return image, label, heatmap, sdm


def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):

        image, label, heatmap, sdm = sample["image"], sample["label"], sample["heatmap"], sample["sdm"]

        image = np.array(image)
        label = np.array(label)
        heatmap = np.array(heatmap)
        sdm = np.array(sdm)

        if random.random() > 0.5:
            image, label, heatmap, sdm = random_rot_flip(image, label, heatmap, sdm)
        elif random.random() > 0.5:
            image, label, heatmap, sdm = random_rotate(image, label, heatmap, sdm)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        heatmap = torch.from_numpy(heatmap.astype(np.float32))
        sdm = torch.from_numpy(sdm.astype(np.float32))

        sample = {"image": image, "label": label, "heatmap": heatmap, "sdm": sdm}
        return sample


class RandomGenerator1(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):

        image, label, heatmap, sdm = sample["image"], sample["label"], sample["heatmap"], sample["sdm"]

        image = np.array(image)
        label = np.array(label)
        heatmap = np.array(heatmap)
        sdm = np.array(sdm)


        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        heatmap = torch.from_numpy(heatmap.astype(np.float32))
        sdm = torch.from_numpy(sdm.astype(np.float32))

        sample = {"image": image, "label": label, "heatmap": heatmap, "sdm": sdm}
        return sample


class Transform(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, heatmap, sdm = sample["image"], sample["label"], sample["heatmap"], sample["sdm"]

        image = np.array(image)
        label = np.array(label)
        heatmap = np.array(heatmap)
        sdm = np.array(sdm)

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        heatmap = torch.from_numpy(heatmap.astype(np.float32))
        sdm = torch.from_numpy(sdm.astype(np.float32))

        sample = {"image": image, "label": label, "heatmap": heatmap, "sdm": sdm}
        return sample


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
            grouper(primary_iter, self.primary_batch_size),
            grouper(secondary_iter, self.secondary_batch_size),
        )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)