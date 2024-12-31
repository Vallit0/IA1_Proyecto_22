import torch
from torch.utils.data import DataLoader
from transformer import Transformer
from dataset import CodeDataset, split_dataset
from preprocessing import preprocess_code, tokenize_code, build_vocab, tokens_to_indices
import json

def load_training_data(file_path):
    """
    Carga los datos de entrenamiento desde un archivo JSON.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    instructions = [item['instruction'] for item in data]
    responses = [item['response'] for item in data]
    return instructions, responses


# Validar índices
def validate_indices(data, vocab_size):
    """
    Valida que los índices en los datos estén dentro del rango del vocabulario.
    """
    if data.max() >= vocab_size or data.min() < 0:
        print("Datos problemáticos:", data)
        raise ValueError(f"Índice fuera de rango detectado en los datos.")


# Collate function para manejar longitudes variables
def collate_fn(batch):
    """
    Rellena cada tensor en el lote con el índice <pad> para igualar las longitudes.
    """
    max_len = max(len(item) for item in batch)
    padded_batch = [torch.cat([item, torch.full((max_len - len(item),), vocab["<pad>"], dtype=torch.long)]) for item in
                    batch]
    batch_tensor = torch.stack(padded_batch)

    # Validar índices en el lote
    validate_indices(batch_tensor, len(vocab))
    return batch_tensor


# Cargar y procesar los datos
file_path = "dataset.json"
instructions, responses = load_training_data(file_path)

# Tokenización y creación del vocabulario
codes = instructions + responses
tokenized_codes = [tokenize_code(preprocess_code(code)) for code in codes]
vocab = build_vocab(tokenized_codes)

# Revisar tokens especiales
assert "<pad>" in vocab, "Falta el token <pad> en el vocabulario."
assert "<sos>" in vocab, "Falta el token <sos> en el vocabulario."
assert "<eos>" in vocab, "Falta el token <eos> en el vocabulario."
assert "<unk>" in vocab, "Falta el token <unk> en el vocabulario."

# Validar vocabulario y tokens
print("Vocabulario:", vocab)
print("Tamaño del vocabulario:", len(vocab))

# Revisar datos procesados
for code in codes[:5]:
    tokens = tokenize_code(preprocess_code(code))
    indices = tokens_to_indices(tokens, vocab)
    print(f"Tokens: {tokens}")
    print(f"Índices: {indices}")
    assert max(indices) < len(vocab), f"Índice fuera de rango detectado: {indices}"

# Configurar DataLoader y modelo
train_dataset, val_dataset = split_dataset(CodeDataset(codes, vocab))
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, collate_fn=collate_fn)

# Modelo y configuración
model = Transformer(input_dim=len(vocab), output_dim=len(vocab), embed_dim=128, num_heads=8, num_layers=4, ff_dim=512)
criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento del modelo
for epoch in range(10):
    model.train()
    for batch in train_loader:
        src = batch[:, :-1]  # Entrada para el encoder
        tgt = batch[:, :-1]  # Entrada para el decoder
        tgt_y = batch[:, 1:]  # Salida esperada

        optimizer.zero_grad()

        # Salida del modelo
        output = model(src, tgt)

        # Asegurar contigüidad
        output = output.contiguous()
        tgt_y = tgt_y.contiguous()

        # Calcular pérdida
        loss = criterion(output.reshape(-1, len(vocab)), tgt_y.view(-1))

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")



# Guardar el modelo entrenado
torch.save(model.state_dict(), "transformer_model.pth")
