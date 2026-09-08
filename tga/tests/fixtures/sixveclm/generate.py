import ast
import copy
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

source = Path(sys.argv[1])
classes = {
    "EncoderDecoder", "Generator", "Encoder", "LayerNorm", "SublayerConnection",
    "EncoderLayer", "Decoder", "DecoderLayer", "MultiHeadedAttention",
    "PositionwiseFeedForward", "Embeddings", "PositionalEncoding",
}
functions = {"clones", "attention", "subsequent_mask", "sample", "next_generation", "word2id"}
tree = ast.parse(source.read_text())
selected = [node for node in tree.body if
            isinstance(node, ast.ClassDef) and node.name in classes or
            isinstance(node, ast.FunctionDef) and node.name in functions]
exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"))

torch.set_num_threads(1)
bank = torch.tensor([
    [0.0, 0.0, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4],
    [-0.4, 0.3, -0.2, 0.1], [0.2, -0.1, 0.4, -0.3],
    [0.3, 0.1, -0.4, 0.2], [-0.1, 0.4, 0.2, 0.3],
])
model = EncoderDecoder(
    Encoder(EncoderLayer(4, MultiHeadedAttention(2, 4, 0.0), PositionwiseFeedForward(4, 8, 0.0), 0.0), 2),
    Decoder(DecoderLayer(4, MultiHeadedAttention(2, 4, 0.0), MultiHeadedAttention(2, 4, 0.0),
                         PositionwiseFeedForward(4, 8, 0.0), 0.0), 2),
    nn.Sequential(Embeddings(4, 6, bank.clone()), PositionalEncoding(4, 0.0)),
    nn.Sequential(Embeddings(4, 6, bank.clone()), PositionalEncoding(4, 0.0)),
    Generator(4, 6),
)
with torch.no_grad():
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            for row in range(layer.out_features):
                for col in range(layer.in_features):
                    layer.weight[row, col] = math.sin((row * layer.in_features + col + 1) * 0.17) * 0.2
                layer.bias[row] = (row - 0.5) * 0.01

src = torch.tensor([[1, 2, 3], [3, 2, 1]])
tgt = torch.tensor([[4, 5, 1], [1, 5, 4]])
labels = torch.tensor([[5, 1, 2], [5, 4, 3]])
mask = subsequent_mask(3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
memory = model.encode(src, None)
projected = model.generator(model.decode(memory, None, tgt, mask))
loss = (1.0 - F.cosine_similarity(projected, bank[labels], dim=2, eps=1e-8)).mean()
loss.backward()

def values(tensor):
    return tensor.detach().flatten().tolist()

fixture = {
    "source": "https://github.com/CuiTianyu961030/6VecLM",
    "revision": "e754093bdd8c3526e190bf231bcd2e658b699acb",
    "torch": torch.__version__,
    "bank": values(bank),
    "memory": values(memory),
    "projected": values(projected),
    "loss": loss.item(),
    "projection_gradient": values(model.generator.proj.weight.grad.transpose(0, 1)),
    "encoder_query_gradient": values(model.encoder.layers[0].self_attn.linears[0].weight.grad.transpose(0, 1)),
}
assert model.src_embed[0].lut.weight.grad is None
assert model.tgt_embed[0].lut.weight.grad is None
optimizer.step()
fixture["after_step"] = values(model.generator(model(src, tgt, None, mask)))
class WordVectors(dict):
    @property
    def vocab(self):
        return self

class WordModel:
    wv = WordVectors({"0h": bank[1].tolist(), "1h": bank[2].tolist(), "2h": bank[3].tolist()})

prediction = torch.tensor([[0.11, -0.2, 0.3, 0.7]])
fixture["sampling_prediction"] = values(prediction)
fixture["sampling_vectors"] = [values(bank[index]) for index in (1, 2, 3)]
fixture["sampling"] = []

def capture_multinomial(count, probabilities, trials):
    fixture["sampling"].append(probabilities.tolist())
    result = np.zeros((trials, len(probabilities)), dtype=int)
    result[0, np.argmax(probabilities)] = count
    return result

np.random.multinomial = capture_multinomial
fixture["temperatures"] = [0.01, 0.5, 1.0]
for temperature in fixture["temperatures"]:
    next_generation(WordModel(), prediction, temperature, 17)

Path(__file__).with_name("transformer.json").write_text(json.dumps(fixture, indent=2) + "\n")
