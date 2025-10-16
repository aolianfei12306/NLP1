import torch
from tqdm import tqdm
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

def generate(model, dataset, src_str, device='cuda', max_len=50):
    arch = "decoder-only" if isinstance(model, AdditionModel_DecoderOnly) else "encoder-decoder"
    pad_idx = dataset.vocab["<pad>"]
    sos_idx = dataset.vocab["<sos>"]
    eos_idx = dataset.vocab["<eos>"]
    generated = [sos_idx] + dataset._encode(src_str)

    # === 准备输入 ===
    if arch == "encoder-decoder":
        src_ids = dataset._pad(dataset._encode(src_str))
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = create_src_mask(src_tensor, pad_idx, device)
        generated = [sos_idx]
        decode_start = 1
    else:  # decoder-only
        src_tensor, src_mask = None, None
        generated = dataset._encode(src_str)
        decode_start = len(generated)

    # === 循环生成 ===
    with torch.no_grad():
        for _ in range(max_len):
            tgt_tensor = torch.tensor([generated], dtype=torch.long, device=device)
            tgt_mask = create_tgt_mask(tgt_tensor, pad_idx, device)

            # 统一 forward，区别只在参数
            #if arch == "encoder-decoder":
            logits = model(src=src_tensor,tgt=tgt_tensor,src_mask=src_mask,tgt_mask=tgt_mask)
            # else:  # decoder-only
            #     logits = model(tgt_tensor, tgt_mask)

            next_id = int(torch.argmax(logits[0, -1, :]).item())
            generated.append(next_id)

            if next_id == eos_idx:
                break

    # === 解码输出 ===
    tokens = []
    for idx in generated[decode_start:]:
        tok = dataset.id2token[int(idx)]
        if tok == "<eos>":
            break
        tokens.append(tok)

    return "".join(tokens)




from collections import defaultdict

def sebset_loader(dataset, x):
    if(len(dataset) >= x):
        subset = Subset(dataset, indices=range(x))
        loader = DataLoader(subset, batch_size=32, shuffle=False)
    else:
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    return loader

def compute_accuracy(model, dataset, loader, device):
    """
    计算给定 loader 的准确率，同时输出各类别的统计结果。
    arch: "encoder-decoder" 或 "decoder-only"
    """
    arch = "decoder-only" if isinstance(model, AdditionModel_DecoderOnly) else "encoder-decoder"
    model.eval()
    correct = 0
    total = 0
    stats = defaultdict(lambda: {"correct": 0, "total": 0})

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            src = batch["input_ids"].to(device)
            tgt = batch["labels"].to(device)
            batch_size = src.size(0)
            for i in range(batch_size):
                # ground truth
                true = decode_tgt(tgt[i], dataset)

                # 构造输入
                src_str = ""
                for idx in src[i]:
                    if idx.item() == dataset.vocab["<pad>"] and arch == "encoder-decoder":
                        break
                    src_str += dataset.id2token[idx.item()]

                # 类别统计
                if "+" in src_str and "=" in src_str:
                    a, b = src_str.split("+")
                    b = b.replace("=", "")
                    key = f"{len(a)}+{len(b)}"
                else:
                    key = "unknown"
                
                # decoderonly 模型需要补齐
                if arch == "decoder-only":
                    for _ in range(len(src_str), dataset.max_len):
                        src_str += "<pad>"

                # 推理
                pred = generate(model, dataset, src_str, device=device)

                stats[key]["total"] += 1
                if pred == true:
                    correct += 1
                    stats[key]["correct"] += 1
                total += 1

    accuracy = correct / total if total > 0 else 0

    detailed_stats = {}
    for key, v in stats.items():
        total_k = v["total"]
        correct_k = v["correct"]
        acc_k = correct_k / total_k if total_k > 0 else 0
        detailed_stats[key] = {
            "samples": total_k,
            "correct": correct_k,
            "accuracy": acc_k
        }

    return accuracy, detailed_stats



if __name__  == "__main__":
    dataset = AdditionDataset()

    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    #model = AdditionModel(vocab_size=len(dataset.vocab)).to(device)
    model = AdditionModel_DecoderOnly(vocab_size=len(dataset.vocab)).to(device)
    checkpoint = torch.load("checkpoints/model_decoder_only_final.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])


    max_len = dataset.max_len
    test_dataset = load_dataset("addition_test.json")
    # 创建 DataLoader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
    model.eval()

    print(compute_accuracy(model, test_dataset,test_loader, device))

    while(1):
        in_src = input("input:")
        if(in_src == "exit"):
            break
        if(in_src == ""):
            continue
        if(in_src[-1] != '='):
            in_src += '='
        if(len(in_src) > max_len):
            print(f"Input too long! Max length is {max_len}.")
            continue
        if isinstance(model, AdditionModel_DecoderOnly):
            for _ in range(len(in_src), dataset.max_len):
               in_src += "<pad>"
        print(generate(model, dataset ,src_str = in_src))