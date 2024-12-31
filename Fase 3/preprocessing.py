import re
from collections import Counter
import torch

# Limpieza de código fuente
def preprocess_code(code):
    code = re.sub(r"#.*", "", code)  # Eliminar comentarios en Python
    code = re.sub(r"//.*", "", code)  # Eliminar comentarios en JavaScript
    code = re.sub(r"/\*[\s\S]*?\*/", "", code)  # Eliminar bloques de comentarios en JavaScript
    code = re.sub(r"\s+", " ", code).strip()  # Espacios innecesarios
    return code

# Tokenización del código
def tokenize_code(code):
    token_pattern = r"[a-zA-Z_][a-zA-Z0-9_]*|==|!=|<=|>=|&&|\|\||[{}\[\]()<>+=\-*/%,;.:\"']"
    tokens = re.findall(token_pattern, code)
    return tokens

# Construcción de vocabulario
def build_vocab(tokens_list, max_vocab_size=5000):
    counter = Counter([token for tokens in tokens_list for token in tokens])
    most_common = counter.most_common(max_vocab_size - 4)  # Reservar 4 espacios para tokens especiales
    vocab = {
        "<pad>": 0,
        "<sos>": 1,
        "<eos>": 2,
        "<unk>": 3
    }
    vocab.update({token: idx + 4 for idx, (token, _) in enumerate(most_common)})
    return vocab

# Conversión de tokens a índices
def tokens_to_indices(tokens, vocab):
    return [vocab["<sos>"]] + [vocab.get(token, vocab["<unk>"]) for token in tokens] + [vocab["<eos>"]]

# Validar datos procesados
def validate_indices(indices, vocab_size):
    assert all(0 <= idx < vocab_size for idx in indices), f"Índice fuera de rango detectado: {indices}"
