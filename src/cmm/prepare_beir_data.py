import os
import json
import random
import pickle
import torch
import zipfile


def make_zip_tensor(txt_path, zip_dir):
    """
    将文本文件打包成zip后再转成tensor并保存，避免重复制作。
    输入是txt路径，输出zip_tensor文件路径。
    """
    filename = os.path.basename(txt_path)
    zip_filename = filename.replace(".txt", ".zip")
    zip_path = os.path.join(zip_dir, zip_filename)
    tensor_path = zip_path + ".pt"

    if os.path.exists(tensor_path):
        return tensor_path

    # 打包成zip文件
    os.makedirs(zip_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(txt_path, arcname=filename)

    # 读取zip二进制内容转tensor保存
    with open(zip_path, "rb") as f:
        data = f.read()
    tensor = torch.tensor(list(data), dtype=torch.uint8)
    torch.save(tensor, tensor_path)

    return tensor_path


def prepare_beir_data(data_dir, out_dir, train_ratio=0.9, negative_ratio=1):
    """
    data_dir: BEIR数据根目录，假设已经解压并包含：
        - corpus.json
        - queries.json
        - qrels/train.tsv
    out_dir: 输出目录，保存 train_samples.pkl eval_samples.pkl docid2path.pkl
    train_ratio: 训练集占比
    negative_ratio: 每个正样本负采样数量

    适配BEIR msmarco-passage格式。
    """

    # 1. 读语料
    corpus_path = os.path.join(data_dir, "corpus.json")
    queries_path = os.path.join(data_dir, "queries.json")
    qrels_path = os.path.join(data_dir, "qrels", "train.tsv")

    print("加载语料中...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    print(f"语料条数: {len(corpus)}")

    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    print(f"查询条数: {len(queries)}")

    qrels = {}
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            qid, docid, rel = line.strip().split("\t")
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][docid] = int(rel)

    print(f"qrels条数: {sum(len(v) for v in qrels.values())}")

    # 2. 把语料写成txt文件
    txt_dir = os.path.join(out_dir, "txts")
    os.makedirs(txt_dir, exist_ok=True)
    docid2txt = {}
    for docid, doc in corpus.items():
        txt_path = os.path.join(txt_dir, f"{docid}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(doc.get("text", ""))
        docid2txt[docid] = txt_path

    # 3. 做zip tensor文件，避免重复做
    zip_dir = os.path.join(out_dir, "zips")
    os.makedirs(zip_dir, exist_ok=True)

    docid2ziptensor = {}
    for docid, txt_path in docid2txt.items():
        tensor_path = make_zip_tensor(txt_path, zip_dir)
        docid2ziptensor[docid] = tensor_path

    # 4. 构造训练样本: (query_text, zip_tensor_path, label)
    samples = []
    all_docids = set(docid2ziptensor.keys())

    for qid, query_text in queries.items():
        pos_docids = [docid for docid, rel in qrels.get(qid, {}).items() if rel > 0]
        if not pos_docids:
            continue

        neg_docids = list(all_docids - set(pos_docids))

        for pos_docid in pos_docids:
            pos_tensor = docid2ziptensor[pos_docid]
            samples.append((query_text, pos_tensor, 1))
            for _ in range(negative_ratio):
                neg_docid = random.choice(neg_docids)
                neg_tensor = docid2ziptensor[neg_docid]
                samples.append((query_text, neg_tensor, 0))

    # 5. 打乱样本，划分训练集和评估集
    random.shuffle(samples)
    train_size = int(len(samples) * train_ratio)
    train_samples = samples[:train_size]
    eval_samples = samples[train_size:]

    # 6. 保存结果
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "train_samples.pkl"), "wb") as f:
        pickle.dump(train_samples, f)
    with open(os.path.join(out_dir, "eval_samples.pkl"), "wb") as f:
        pickle.dump(eval_samples, f)
    with open(os.path.join(out_dir, "docid2path.pkl"), "wb") as f:
        pickle.dump(docid2ziptensor, f)

    print(f"数据准备完成！")
    print(f"训练集样本数: {len(train_samples)}")
    print(f"评估集样本数: {len(eval_samples)}")
    print(f"输出路径: {out_dir}")
