import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os

from tqdm import tqdm

from dataset import *
from transformer import *
from test import *


def train_decoder_only(epochs=100, save_every=5, resume="0.pth"):
    """训练 Decoder-Only 版本的 AdditionModel"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Training Decoder-Only model")

    # 创建保存模型的目录
    os.makedirs("checkpoints", exist_ok=True)

    train_dataset = load_dataset("addition_train.json")
    val_dataset = load_dataset("addition_validation.json")
    
    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = sebset_loader(val_dataset, 3200)

    vocab_size = len(train_dataset.vocab)
    pad_idx = train_dataset.vocab["<pad>"]
    sos_idx = train_dataset.vocab["<sos>"]
    eos_idx = train_dataset.vocab["<eos>"]

    # 模型 - 使用 Decoder-Only 版本
    model = AdditionModel_DecoderOnly(
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=256,
        max_len=train_dataset.max_len
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    start_epoch = 0

    if resume != '0.pth':
        # 加载 checkpoint
        checkpoint = torch.load(resume, map_location=device)

        # 恢复模型和优化器状态
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # 从 epoch+1 开始继续训练
        start_epoch = checkpoint["epoch"]

    # 训练
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f'Training Decoder-Only, epoch={epoch+1}'):
            src = batch["input_ids"].to(device)
            tgt = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            # Decoder-only 模型的训练逻辑
            # 将输入和输出拼接成一个序列
            batch_size = src.size(0)
            
            # 创建拼接后的序列: <sos> + 输入序列 + 输出序列
            combined_seq = torch.cat([
                # torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device),  # 起始符
                src,  # 输入序列
                tgt[:, :]  # 输出序列
            ], dim=1)
            
            # 创建目标序列: 输入序列 + 输出序列 (去掉开头的<sos>)
            # 注意: 我们需要预测的是每个位置的下一个token
            target_seq = torch.cat([
                src[:, 1:],  # 输入序列 (左移一位)
                tgt[:, :],  # 输出序列 (开头本来就没有<sos>)
                torch.full((batch_size, 1), pad_idx, dtype=torch.long, device=device)  # 补齐一个位置
            ], dim=1)
            
            # combined_mask = create_combined_mask(src, tgt, pad_idx, device)
            seq_len = combined_seq.size(1)
            sequence_mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()


            # 前向传播
            output = model(combined_seq, sequence_mask)
            
            seq_len = min(output.size(1), target_seq.size(1))
            output = output[:, :seq_len, :]
            target_seq = target_seq[:, :seq_len]
            
            # 计算损失
            loss_mask = torch.zeros_like(combined_seq, dtype=torch.bool)
            loss_mask[:, src.size(1):] = True  # 仅 tgt 部分参与 loss

            loss = criterion(output.transpose(1, 2), target_seq)
            loss = (loss * loss_mask).sum() / loss_mask.sum()

            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        
        # 每2个epoch进行一次验证
        if (epoch + 1) % 2 == 0:
            # 注意: 当前的测试代码可能只支持encoder-decoder模型
            # 对于decoder-only模型，我们暂时只记录损失
            log1 = compute_accuracy(model, val_dataset, val_loader, device)
            with open("checkpoints/val_log.txt", "a") as f:
                f.write(f"Epoch {epoch+1}, Validation Accuracy: {log1}\n")
            print(f"Epoch {epoch+1}, Validation Accuracy: {log1}")
            with open("checkpoints/val_log_decoder_only.txt", "a") as f:
                f.write(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}\n")
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        
        # 每save_every个epoch保存一次模型
        if (epoch + 1) % save_every == 0:
            checkpoint_path = f"checkpoints/model_decoder_only_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'vocab': train_dataset.vocab,
                'model_type': 'decoder_only'
            }, checkpoint_path)
            print(f"Decoder-Only 模型已保存到 {checkpoint_path}")

    # 训练结束后保存最终模型
    final_checkpoint_path = "checkpoints/model_decoder_only_final.pth"
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'vocab': train_dataset.vocab,
        'model_type': 'decoder_only'
    }, final_checkpoint_path)
    print(f"Decoder-Only 最终模型已保存到 {final_checkpoint_path}")

    return model

if __name__ == "__main__":
    model = train_decoder_only()