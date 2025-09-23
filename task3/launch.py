import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os

from dataset import *
from transformer import *
from test import *


def train(epochs = 10, save_every=5):  # 添加save_every参数，默认每5个epoch保存一次
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 创建保存模型的目录
    os.makedirs("checkpoints", exist_ok=True)

    # # 数据
    # dataset = AdditionDataset(data_lists=[(3,3,10000),(3,4,20000),(4,3,20000),(5,3,40000),(3,5,40000),(4,4,40000)], max_len=12)
    # #loader = DataLoader(dataset, batch_size=32, shuffle=True)
    # #划分train val
    
    # train_dataset, val_dataset, test_dataset = random_split(dataset, [0.7, 0.2, 0.1])
    # # 保存数据集到本地
    # save_dataset(train_dataset, 'addition_train.json')
    # save_dataset(val_dataset, 'addition_validation.json')
    # save_dataset(test_dataset, 'addition_test.json')

    train_dataset = load_dataset("addition_train.json")
    val_dataset = load_dataset("addition_validation.json")
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


    vocab_size = len(train_dataset.vocab)
    pad_idx = train_dataset.vocab["<pad>"]
    sos_idx = train_dataset.vocab["<sos>"]
    eos_idx = train_dataset.vocab["<eos>"]

    # 模型
    model = AdditionModel(
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=256,
        max_len=train_dataset.max_len
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 训练
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            src = batch["input_ids"].to(device)
            tgt = batch["labels"].to(device)
            
            # 教师强制 (Teacher Forcing)：decoder 输入用 <sos> + y[:-1]
            tgt_in = torch.cat([torch.full((tgt.size(0), 1), sos_idx, dtype=torch.long, device=device), tgt[:, :-1]], dim=1)
            # 生成掩码（使用 tgt 而非 tgt_in，因为掩码基于目标序列）
            src_mask = create_src_mask(src, pad_idx, device)
            tgt_mask = create_tgt_mask(tgt_in, pad_idx, device)

            optimizer.zero_grad()
            output = model(src, tgt_in, src_mask, tgt_mask)

            # output: [batch, seq_len, vocab_size]
            loss = criterion(output.transpose(1, 2), tgt)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
        
        print(compute_accuracy(model, val_dataset, val_loader, device))
        
        # 每save_every个epoch保存一次模型
        if (epoch + 1) % save_every == 0:
            checkpoint_path = f"checkpoints/model_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss/len(train_loader),
                'vocab': train_dataset.vocab  # 保存词汇表以便后续使用
            }, checkpoint_path)
            print(f"模型已保存到 {checkpoint_path}")

    # 训练结束后保存最终模型
    final_checkpoint_path = "checkpoints/model_final.pth"
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss/len(train_loader),
        'vocab': train_dataset.vocab
    }, final_checkpoint_path)
    print(f"最终模型已保存到 {final_checkpoint_path}")

    return model


if __name__ == "__main__":
    model = train()