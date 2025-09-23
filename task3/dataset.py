import json
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import random

class AdditionDataset(Dataset):
    def __init__(self, data_lists=[(3,3,10000),(3,4,20000),(4,3,20000)], max_len=20):
        '''
        data_list: (digit_a,digit_b,num_samples)
        max_len: 输入/输出最大长度（决定 padding）
        '''
        self.max_len = max_len

        # 构造词表（字符级别）
        self.vocab = {str(i): i for i in range(10)}
        self.vocab.update({"+": 10, "=": 11, "<pad>": 12, "<sos>": 13, "<eos>": 14})
        self.id2token = {v: k for k, v in self.vocab.items()}

        # 生成数据
        self.data = []
        for i in data_lists:
            self.data += self._generate_data(i[2],i[0],i[1])

    def _generate_data(self, num, digits_a, digits_b):
        data = []
        for _ in range(num):
            a = random.randint(10**(digits_a-1), 10**digits_a - 1)
            b = random.randint(10**(digits_b-1), 10**digits_b - 1)
            x = f"{a}+{b}="  # 输入序列
            y = str(a + b) + "<eos>"   # 输出序列
            data.append((x, y))
        return data

    def _encode(self, text: str):
        tokens = []
        i = 0
        while i < len(text):
            if text[i] == "<":  # 可能是特殊 token
                j = text.find(">", i)
                if j == -1:
                    raise ValueError(f"不完整的特殊token: {text[i:]}")
                token = text[i:j+1]
                if token not in self.vocab:
                    raise ValueError(f"未知的特殊token: {token}")
                tokens.append(self.vocab[token])
                i = j + 1
            else:
                # 普通字符（数字、+、=）
                ch = text[i]
                if ch not in self.vocab:
                    raise ValueError(f"未知字符: {ch}")
                tokens.append(self.vocab[ch])
                i += 1
        return tokens
    

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


def save_dataset(dataset, file_path):
    """
    保存 AdditionDataset 或 Subset 的数据到本地 JSON 文件，使用统一格式。
    
    Args:
        dataset: AdditionDataset 或 Subset 实例
        file_path: 保存路径（如 'dataset.json'）
    """
    if isinstance(dataset, Subset):
        # For Subset, extract data using indices
        original_dataset = dataset.dataset
        data = [original_dataset.data[i] for i in dataset.indices]
        max_len = original_dataset.max_len
    else:
        # For AdditionDataset, use directly
        data = dataset.data
        max_len = dataset.max_len
    
    data_to_save = {
        'max_len': max_len,
        'data': data  # list of tuples (x, y)
    }
    with open(file_path, 'w') as f:
        json.dump(data_to_save, f)
    print(f"Dataset saved to {file_path}")

def load_dataset(file_path):
    """
    从本地 JSON 文件恢复 AdditionDataset。
    
    Args:
        file_path: 保存路径（如 'dataset.json'）
    
    Returns:
        AdditionDataset 实例
    """
    with open(file_path, 'r') as f:
        data_loaded = json.load(f)
    
    # Create new AdditionDataset instance
    dataset = AdditionDataset(data_lists=[], max_len=data_loaded['max_len'])
    dataset.data = data_loaded['data']
    return dataset


if __name__ == "__main__":
    # dataset = AdditionDataset(num_samples=5, digits_a=3, digits_b=3, max_len=10)
    # loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # for batch in loader:
    #     print("input_ids:", batch["input_ids"])
    #     print("labels:", batch["labels"])


    # 数据
    dataset = AdditionDataset(data_lists=[(3,3,10000),(3,4,20000),(4,3,20000),(5,3,40000),(3,5,40000),(4,4,40000)], max_len=12)
    #loader = DataLoader(dataset, batch_size=32, shuffle=True)
    #划分train val
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [0.7, 0.2, 0.1])
    # 保存数据集到本地
    save_dataset(train_dataset, 'addition_train.json')
    save_dataset(val_dataset, 'addition_validation.json')
    save_dataset(test_dataset, 'addition_test.json')