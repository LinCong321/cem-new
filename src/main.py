import torch
# from cmm import trainer


def test_e5_encoder():
    from encoders.e5_encoder import E5Encoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = E5Encoder().to(device)

    sample_texts = [
        "这是一个测试句子。",
        "这是第二个测试。"
    ]

    with torch.no_grad():
        embeddings = encoder(sample_texts)

    print("输出向量形状:", embeddings.shape)
    print("示例向量（第一条）:", embeddings[0][:5])


def test_blt_encoder():
    from encoders.blt_encoder import BLTEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = BLTEncoder().to(device)

    sample_tokens = torch.randint(0, 256, (2, 128), dtype=torch.long).to(device)

    with torch.no_grad():
        embeddings = encoder(sample_tokens)

    print("输出向量形状:", embeddings.shape)
    print("示例向量（第一条）:", embeddings[0][:5])


if __name__ == "__main__":
    test_e5_encoder()
    test_blt_encoder()
    # trainer.launch()
