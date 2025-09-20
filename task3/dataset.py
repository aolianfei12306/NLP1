import torch
from torch.utils.data import Dataset, DataLoader
import random

class AdditionDataset(Dataset):
    def __init__(self, num_samples=10000, digits_a=3, digits_b=3, max_len=20):
        '''
        num_samples: 样本数量
        digits_a: 第一个加数的位数
        digits_b: 第二个加数的位数
        max_len: 输入/输出最大长度（决定 padding）
        '''
        self.num_samples = num_samples
        self.digits_a = digits_a
        self.digits_b = digits_b
        self.max_len = max_len

        # 构造词表（字符级别）
        self.vocab = {str(i): i for i in range(10)}
        self.vocab.update({"+": 10, "=": 11, "<pad>": 12})
        self.id2token = {v: k for k, v in self.vocab.items()}

        # 生成数据
        self.data = self._generate_data()

    def _generate_data(self):
        data = []
        for _ in range(self.num_samples):
            a = random.randint(10**(self.digits_a-1), 10**self.digits_a - 1)
            b = random.randint(10**(self.digits_b-1), 10**self.digits_b - 1)
            x = f"{a}+{b}="  # 输入序列
            y = str(a + b)   # 输出序列
            data.append((x, y))
        return data

    def _encode(self, text):
        """把字符串转成 ID 序列"""
        return [self.vocab[ch] for ch in text]

    def _pad(self, ids):
        """补齐到 max_len"""
        if len(ids) < self.max_len:
            ids = ids + [self.vocab["<pad>"]] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        return ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        input_ids = self._pad(self._encode(x))
        labels = self._pad(self._encode(y))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


if __name__ == "__main__":
    dataset = AdditionDataset(num_samples=5, digits_a=3, digits_b=3, max_len=10)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch in loader:
        print("input_ids:", batch["input_ids"])
        print("labels:", batch["labels"])
