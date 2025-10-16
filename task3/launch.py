import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os

from tqdm import tqdm

from dataset import *
from transformer import *
from test import *


def train(epochs = 30, save_every=5, resume = "0.pth", model_type="encoder_decoder"):  # 添加save_every参数，默认每5个epoch保存一次
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using model type: {model_type}")

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
    
    val_loader = sebset_loader(val_dataset, 3200)
    # 只用前100个batch做验证
    val_loader_100 = DataLoader(val_dataset, batch_size=32, shuffle=False)


    vocab_size = len(train_dataset.vocab)
    pad_idx = train_dataset.vocab["<pad>"]
    sos_idx = train_dataset.vocab["<sos>"]
    eos_idx = train_dataset.vocab["<eos>"]

        # 根据模型类型选择不同的模型
    if model_type == "encoder_decoder":
        model = AdditionModel(
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            d_model=128,
            num_layers=2,
            num_heads=4,
            d_ff=256,
            max_len=train_dataset.max_len
        ).to(device)
    elif model_type == "decoder_only":
        model = AdditionModel_DecoderOnly(
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            d_model=128,
            num_layers=2,
            num_heads=4,
            d_ff=256,
            max_len=train_dataset.max_len * 2  # Decoder-only 模型可能需要更长的序列长度
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'encoder_decoder' or 'decoder_only'")

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

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
    for epoch in range(start_epoch,epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc = 'Training,epoch={}'.format(epoch+1)):
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
        
        if (epoch + 1) % 2 == 0:
            log1 = compute_accuracy(model, val_dataset, val_loader, device)
            with open("checkpoints/val_log.txt", "a") as f:
                f.write(f"Epoch {epoch+1}, Validation Accuracy: {log1}\n")
            print(f"Epoch {epoch+1}, Validation Accuracy: {log1}")
        
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