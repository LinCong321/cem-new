import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader


class BEIRDataset(Dataset):
    def __init__(self, data_dir, mode="train", device=None):
        """
        Args:
            data_dir (str): 预处理后的数据目录，包含 train_samples.pkl, eval_samples.pkl, docid2path.pkl
            mode (str): "train" 或 "eval"
            device (torch.device or None): tensor加载后转移到的设备，None则不转移
        """
        assert mode in ("train", "eval"), "mode必须是 'train' 或 'eval'"
        self.device = device

        samples_file = os.path.join(data_dir, "train_samples.pkl" if mode == "train" else "eval_samples.pkl")
        docid2path_file = os.path.join(data_dir, "docid2path.pkl")

        with open(samples_file, "rb") as f:
            self.samples = pickle.load(f)

        with open(docid2path_file, "rb") as f:
            self.docid2path = pickle.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        query_text, zip_tensor_path, label = self.samples[idx]
        tensor = torch.load(zip_tensor_path)
        if self.device is not None:
            tensor = tensor.to(self.device)
        return query_text, tensor, label


def get_dataloader(data_dir, mode="train", device=None, batch_size=32, shuffle=True, num_workers=0):
    dataset = BEIRDataset(data_dir, mode=mode, device=device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
