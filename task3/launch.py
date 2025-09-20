import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os

from dataset import *
from transformer import *

class AdditionModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_layers=2, num_heads=4, d_ff=256, max_len=50, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=vocab_size-1)

        self.positional_encoding = PositionalEncoding(d_model = d_model)
        self.transformer = Transformer(num_layers, d_model, num_heads, d_ff, dropout)

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        # 先加位置编码
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        out, _, _, _ = self.transformer(src, tgt, src_mask, tgt_mask)
        return self.fc_out(out)


def train(epochs = 10, save_every=5):  # 添加save_every参数，默认每5个epoch保存一次
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 创建保存模型的目录
    os.makedirs("checkpoints", exist_ok=True)

    # 数据
    dataset = AdditionDataset(num_samples=20000, digits_a=3, digits_b=3, max_len=12)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    vocab_size = len(dataset.vocab)

    # 模型
    model = AdditionModel(
        vocab_size=vocab_size,
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=256,
        max_len=dataset.max_len
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab["<pad>"])
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # 训练
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in loader:
            src = batch["input_ids"].to(device)
            tgt = batch["labels"].to(device)

            # 教师强制 (Teacher Forcing)：decoder 输入用 <sos> + y[:-1]
            tgt_in = torch.cat([torch.full((tgt.size(0), 1), dataset.vocab["+"], dtype=torch.long, device=device), tgt[:, :-1]], dim=1)

            optimizer.zero_grad()
            output = model(src, tgt_in)

            # output: [batch, seq_len, vocab_size]
            loss = criterion(output.transpose(1, 2), tgt)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
        
        # 每save_every个epoch保存一次模型
        if (epoch + 1) % save_every == 0:
            checkpoint_path = f"checkpoints/model_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss/len(loader),
                'vocab': dataset.vocab  # 保存词汇表以便后续使用
            }, checkpoint_path)
            print(f"模型已保存到 {checkpoint_path}")

    # 训练结束后保存最终模型
    final_checkpoint_path = "checkpoints/model_final.pth"
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss/len(loader),
        'vocab': dataset.vocab
    }, final_checkpoint_path)
    print(f"最终模型已保存到 {final_checkpoint_path}")

    return model, dataset


if __name__ == "__main__":
    model, dataset = train()