import torch
from transformer import Transformer
from preprocessing import tokenize_code, preprocess_code

def load_model(model_path, vocab_size, embed_dim, num_heads, num_layers, ff_dim):
    """
    Carga un modelo Transformer con configuraciones correctas.
    """
    assert embed_dim % num_heads == 0, "embed_dim debe ser divisible por num_heads"
    model = Transformer(
        input_dim=vocab_size,
        output_dim=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_dim=ff_dim
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

def generate_code(model, src, max_len, vocab):
    """
    Genera código a partir de una instrucción.
    """
    model.eval()
    sos_token = vocab["<sos>"]
    eos_token = vocab["<eos>"]

    # Tokenizar y convertir a índices
    src_indices = [vocab.get("<unk>")] * 10  # Simula una entrada válida
    src = torch.tensor([src_indices], dtype=torch.long)
    print("Dimensiones de src:", src.shape)

    # Pasar por el encoder
    memory = model.encoder(model.encoder_embedding(src))
    print("Dimensiones de memory antes de permute:", memory.shape)

    # Ajustar dimensiones de memory
    memory = memory.permute(1, 0, 2)  # Cambiar a (batch_size, seq_len, embed_dim)
    print("Dimensiones de memory después de permute:", memory.shape)

    # Inicializar la generación con el token <sos>
    tgt = torch.tensor([[sos_token]], dtype=torch.long)
    for _ in range(max_len):
        # Paso por el decoder
        tgt_emb = model.decoder_embedding(tgt)
        print("Dimensiones de tgt_emb:", tgt_emb.shape)

        # Asegurar que tgt_emb y memory sean compatibles
        memory = memory.contiguous()  # Asegura la contigüidad
        tgt_emb = tgt_emb.contiguous()  # Asegura la contigüidad

        # Pasar por el decoder
        decoder_output = model.decoder(tgt_emb, memory)
        print("Dimensiones de decoder_output:", decoder_output.shape)

        # Proyectar la salida al tamaño del vocabulario
        output = model.fc_out(decoder_output)
        print("Dimensiones de output:", output.shape)

        next_token = output.argmax(-1)[:, -1].item()
        tgt = torch.cat([tgt, torch.tensor([[next_token]])], dim=1)
        if next_token == eos_token:
            break

    # Convertir índices a tokens
    generated_tokens = [key for key, idx in vocab.items() if idx in tgt.squeeze().tolist()]
    return " ".join(generated_tokens)

# Configuraciones
model_path = "transformer_model.pth"
embed_dim = 128  # Debe ser divisible por num_heads
num_heads = 8
num_layers = 4
ff_dim = 512

# Inferir el tamaño del vocabulario
checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
vocab_size = checkpoint["encoder_embedding.weight"].shape[0]
print(f"Tamaño del vocabulario: {vocab_size}")

# Reconstruir vocabulario
vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
for i in range(4, vocab_size):
    vocab[f"token_{i}"] = i

# Cargar el modelo
model = load_model(model_path, vocab_size, embed_dim, num_heads, num_layers, ff_dim)

# Generar código
instruction = "Create a quick-sort algorithm in Python."
response = generate_code(model, instruction, max_len=50, vocab=vocab)
print("Generated Code:", response)
