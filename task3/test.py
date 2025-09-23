import torch
from dataset import *
from transformer import *

def decode_tgt(labels, dataset):
    """
    从 labels 张量解码为字符串，忽略 <sos>、<eos>、<pad>
    """
    pad_idx = dataset.vocab["<pad>"]
    eos_idx = dataset.vocab["<eos>"]
    sos_idx = dataset.vocab["<sos>"]
    tokens = []
    for idx in labels:
        if idx.item() == pad_idx or idx.item() == eos_idx or idx.item() == sos_idx:
            if idx.item() == eos_idx:
                break
            continue
        tok = dataset.id2token[idx.item()]
        tokens.append(tok)
    return "".join(tokens)

def greedy_decode(model, dataset, src_str, max_len=None, device=None):
    """
    Greedy decode for your Transformer-based AdditionModel.
    - model: 已加载的 AdditionModel (已 to(device))
    - dataset: AdditionDataset 实例（用于 vocab, id2token, _encode, _pad, max_len）
    - src_str: 输入字符串，例如 "123+456="
    - max_len: 最大生成长度（默认 dataset.max_len）
    - device: torch.device（默认取 model 的参数所在 device）
    返回：生成的字符串（不包含 <sos> 或 <eos>）
    """
    model.eval()
    # 取 device
    if device is None:
        device = next(model.parameters()).device

    pad_idx = dataset.vocab["<pad>"]
    sos_idx = dataset.vocab["<sos>"]
    eos_idx = dataset.vocab["<eos>"]

    if max_len is None:
        max_len = dataset.max_len

    # --- prepare src (encode + pad) ---
    src_ids = dataset._encode(src_str)          # eg [1,2,3,10,4,5,6,11]
    src_ids = dataset._pad(src_ids)             # pad 到 dataset.max_len
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)  # [1, src_len]

    # src mask 与训练时一致
    src_mask = create_src_mask(src_tensor, pad_idx, device)  # 参考你定义的 create_src_mask

    # --- decode loop (greedy) ---
    generated = [sos_idx]

    with torch.no_grad():
        for _ in range(max_len):
            tgt_tensor = torch.tensor([generated], dtype=torch.long, device=device)  # [1, cur_tgt_len]
            tgt_mask = create_tgt_mask(tgt_tensor, pad_idx, device)  # 使用当前 tgt_tensor 生成 mask

            # forward：model 返回 [batch, seq_len, vocab_size]
            logits = model(src_tensor, tgt_tensor, src_mask, tgt_mask)  # logits shape
            # 如果 model 返回 tuple (fc_out out), 上面假定 model 返回 fc_out(out)
            # 取最后一步的 logits
            # logits: [1, cur_tgt_len, vocab_size]
            next_logits = logits[0, -1, :]  # [vocab_size]
            next_id = int(torch.argmax(next_logits, dim=-1).item())

            generated.append(next_id)

            if next_id == eos_idx:
                break

    # 把生成 id 序列转换为字符串（跳过起始符 <sos>，遇到 <eos> 停）
    tokens = []
    for idx in generated[1:]:  # skip <sos>
        tok = dataset.id2token[int(idx)]
        if tok == "<eos>":
            break
        tokens.append(tok)

    return "".join(tokens)


def compute_accuracy(model, dataset, loader, device):
    """
    计算给定 loader 的准确率
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            src = batch["input_ids"].to(device)
            tgt = batch["labels"].to(device)
            batch_size = src.size(0)
            for i in range(batch_size):
                # 解码 src 为字符串
                src_ids = src[i]
                src_str = ""
                for idx in src_ids:
                    if idx.item() == dataset.vocab["<pad>"]:
                        break
                    src_str += dataset.id2token[idx.item()]
                
                # 预测
                pred = greedy_decode(model, dataset, src_str, device=device)
                
                # 真实标签
                true = decode_tgt(tgt[i], dataset)
                
                if pred == true:
                    correct += 1
                total += 1
    accuracy = correct / total if total > 0 else 0
    return accuracy


if __name__  == "__main__":
    dataset = AdditionDataset()

    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = AdditionModel(vocab_size=len(dataset.vocab)).to(device)
    checkpoint = torch.load("checkpoints/model_final.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_dataset = load_dataset("addition_test.json")
    # 创建 DataLoader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    model.eval()

    print(compute_accuracy(model, test_dataset, test_loader, device))

    while(1):
        in_src = input("input:")
        print(greedy_decode(model, dataset ,src_str = in_src))