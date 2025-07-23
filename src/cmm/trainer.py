import time
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from beir_data_loader import get_dataloader
from cross_modal_model import CrossModalModel
from beir import util
from beir.retrieval.evaluation import EvaluateRetrieval


def nt_xent_loss(z1, z2, labels, temperature=0.07, standardize='batch'):
    """
    对比学习的NT-Xent损失
    z1, z2: [B, D] tensor
    labels: [B] tensor，1表示正样本，0表示负样本
    """
    if standardize == 'sample':
        z1 = (z1 - z1.mean(dim=1, keepdim=True)) / z1.std(dim=1, keepdim=True).clamp(min=1e-6)
        z2 = (z2 - z2.mean(dim=1, keepdim=True)) / z2.std(dim=1, keepdim=True).clamp(min=1e-6)
    elif standardize == 'batch':
        z1 = (z1 - z1.mean(dim=0)) / z1.std(dim=0).clamp(min=1e-6)
        z2 = (z2 - z2.mean(dim=0)) / z2.std(dim=0).clamp(min=1e-6)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    cosine_sim = torch.matmul(z1, z2.T) / temperature
    pos_mask = torch.eye(len(labels), device=z1.device) * labels.view(-1, 1)
    exp_sim = torch.exp(cosine_sim)
    exp_sim_sum = exp_sim.sum(dim=1, keepdim=True)
    pos_loss = -torch.log((exp_sim * pos_mask).sum(dim=1) / exp_sim_sum.clamp(min=1e-8))
    return pos_loss.mean()


def train_one_epoch(model, optimizer, data_loader, device, scaler):
    model.train()
    total_loss = 0.0
    for batch in data_loader:
        descriptions, zip_tensors, labels = batch
        zip_tensors = zip_tensors.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            text_embeds, zip_embeds = model(descriptions, zip_tensors)
            loss = nt_xent_loss(text_embeds, zip_embeds, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(data_loader)


def evaluate_ndcg(model, data_loader, device):
    """
    这里用 BEIR 的 EvaluateRetrieval 工具，前提是你提前准备了query-doc映射和相关性label文件
    我们手动构造retrieval结果和qrels供评估
    """
    model.eval()
    queries = []
    corpus_embeddings = {}
    qrels = {}
    with torch.no_grad():
        for batch in data_loader:
            descriptions, zip_tensors, labels = batch
            zip_tensors = zip_tensors.to(device)
            text_embeds, zip_embeds = model(descriptions, zip_tensors)
            text_embeds = text_embeds.cpu()
            zip_embeds = zip_embeds.cpu()
            for i, desc in enumerate(descriptions):
                qid = f"q{i}"
                queries.append((qid, text_embeds[i]))
                docid = f"d{i}"
                corpus_embeddings[docid] = zip_embeds[i]
                # qrels里标注相关文档，1为相关，0为不相关
                qrels[qid] = {docid: int(labels[i].item())}
    
    # 构造检索结果 dict: {query_id: {doc_id: score}}
    retrieval_results = {}
    for qid, q_embed in queries:
        scores = {}
        for docid, d_embed in corpus_embeddings.items():
            sim = torch.cosine_similarity(q_embed.unsqueeze(0), d_embed.unsqueeze(0)).item()
            scores[docid] = sim
        # 按分数排序
        sorted_docs = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
        retrieval_results[qid] = sorted_docs

    evaluator = EvaluateRetrieval()
    results = evaluator.evaluate(retrieval_results, qrels, retriever_name="CrossModalModel")
    ndcg_10 = results.get("ndcg@10", 0.0)
    print(f"nDCG@10: {ndcg_10:.4f}")
    return ndcg_10


def launch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossModalModel(device=device).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    data_dir = "./beir_prepared"  # 你预处理BEIR数据集后放的目录

    train_loader = get_dataloader(data_dir, mode="train", device=device, batch_size=64, shuffle=True)
    eval_loader = get_dataloader(data_dir, mode="eval", device=device, batch_size=64, shuffle=False)

    epochs = 10
    best_ndcg = 0.0

    for epoch in range(epochs):
        start = time.time()
        train_loss = train_one_epoch(model, optimizer, train_loader, device, scaler)
        print(f"Epoch {epoch+1} Training Loss: {train_loss:.4f}")

        ndcg = evaluate_ndcg(model, eval_loader, device)
        if ndcg > best_ndcg:
            best_ndcg = ndcg
            torch.save(model.state_dict(), "best_cross_modal_model.pt")
            print("Best model saved.")

        print(f"Epoch {epoch+1} finished in {time.time() - start:.2f}s\n")
