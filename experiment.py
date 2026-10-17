"""Word-level autoregressive comparison for ICT4403 Assignment 2.
Run in Google Colab or locally with PyTorch, NumPy and Matplotlib.
"""
import json
import copy
import math
import os
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
CONFIG = dict(chars=320_000, seq_len=16, stride=4, min_freq=2,
              embedding=48, hidden=64, batch_size=64, epochs=8,
              learning_rate=0.002, seed=42, temperatures=[0.6, 1.0, 1.4])
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|[.,!?;:]")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokenize(s):
    return TOKEN_RE.findall(s.lower())


def prepare():
    path = ROOT / 'data' / 'tinyshakespeare.txt'
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        import urllib.request
        urllib.request.urlretrieve(DATA_URL, path)
    raw = path.read_text(encoding='utf-8')[:CONFIG['chars']]
    # Contiguous chronological splits; vocabulary is learned from training only.
    cut1, cut2 = int(len(raw)*.7), int(len(raw)*.85)
    train, val, test = [tokenize(s) for s in (raw[:cut1], raw[cut1:cut2], raw[cut2:])]
    counts = Counter(train)
    vocab = ['<unk>'] + sorted(w for w, n in counts.items() if n >= CONFIG['min_freq'])
    stoi = {w: i for i, w in enumerate(vocab)}
    ids = [[stoi.get(w, 0) for w in part] for part in (train, val, test)]
    stats = dict(source=DATA_URL, total_characters=len(raw), raw_file_characters=len(path.read_text(encoding='utf-8')),
                 train_tokens=len(train), validation_tokens=len(val), test_tokens=len(test),
                 vocabulary=len(vocab), validation_oov=sum(w not in stoi for w in val),
                 test_oov=sum(w not in stoi for w in test), config=CONFIG)
    return ids, vocab, stoi, stats


class Windows(Dataset):
    def __init__(self, ids):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.starts = range(0, len(ids) - CONFIG['seq_len'], CONFIG['stride'])
    def __len__(self):
        return len(self.starts)
    def __getitem__(self, i):
        s = self.starts[i]
        return self.ids[s:s+CONFIG['seq_len']], self.ids[s+CONFIG['seq_len']]


class NextWord(nn.Module):
    def __init__(self, kind, vocab_size):
        super().__init__()
        self.kind = kind
        self.emb = nn.Embedding(vocab_size, CONFIG['embedding'])
        if kind == 'Bidirectional RNN':
            self.rnn = nn.RNN(CONFIG['embedding'], CONFIG['hidden'], batch_first=True, bidirectional=True)
            out_size = 2 * CONFIG['hidden']
        elif kind == 'Simple RNN':
            self.rnn = nn.RNN(CONFIG['embedding'], CONFIG['hidden'], batch_first=True)
            out_size = CONFIG['hidden']
        elif kind == 'LSTM':
            self.rnn = nn.LSTM(CONFIG['embedding'], CONFIG['hidden'], batch_first=True)
            out_size = CONFIG['hidden']
        elif kind == 'GRU':
            self.rnn = nn.GRU(CONFIG['embedding'], CONFIG['hidden'], batch_first=True)
            out_size = CONFIG['hidden']
        else:
            raise ValueError(kind)
        self.head = nn.Linear(out_size, vocab_size)

    def forward(self, x):
        _, state = self.rnn(self.emb(x))
        if self.kind == 'LSTM':
            state = state[0]
        # For the bidirectional model, use both final directional hidden states;
        # every input token precedes the target word. No future target is visible.
        h = torch.cat((state[-2], state[-1]), dim=1) if self.kind == 'Bidirectional RNN' else state[-1]
        return self.head(h)


def evaluate(model, loader, device):
    model.eval()
    summed_loss, hits, n = 0., 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            summed_loss += nn.functional.cross_entropy(logits, y, reduction='sum').item()
            hits += (logits.argmax(1) == y).sum().item()
            n += len(y)
    loss = summed_loss / n
    return dict(loss=loss, perplexity=math.exp(loss), accuracy=hits/n, samples=n)


def generate(model, seed, stoi, vocab, temp, device, words=30):
    # Independent RNG for each model/seed/temperature, so sampling is reproducible.
    ids = [stoi.get(w, 0) for w in tokenize(seed)]
    if not ids:
        raise ValueError('Seed contains no tokens')
    model.eval()
    output = ids.copy()
    with torch.no_grad():
        for _ in range(words):
            context = ([0] * max(0, CONFIG['seq_len']-len(output)) + output)[-CONFIG['seq_len']:]
            logits = model(torch.tensor([context], device=device))[0] / temp
            logits[0] = -float('inf')  # do not emit the unknown token
            sampled = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
            output.append(sampled)
    generated = [vocab[i] for i in output[len(ids):]]
    return dict(seed=seed, temperature=temp, generated=' '.join(generated),
                unique_ratio=len(set(generated))/len(generated),
                repeated_bigram_ratio=1-len(set(zip(generated, generated[1:])))/max(1, len(generated)-1))


def main():
    set_seed(CONFIG['seed'])
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ids, vocab, stoi, stats = prepare()
    loaders = [DataLoader(Windows(part), batch_size=CONFIG['batch_size'], shuffle=i==0)
               for i, part in enumerate(ids)]
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    kinds = ['Simple RNN', 'Bidirectional RNN', 'LSTM', 'GRU']
    results = {}
    all_samples = []
    for kind in kinds:
        set_seed(CONFIG['seed'])
        model = NextWord(kind, len(vocab)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
        history = []
        best_loss = float("inf")
        best_state = None
        best_epoch = None
        start = time.perf_counter()
        for epoch in range(CONFIG['epochs']):
            model.train()
            for x, y in loaders[0]:
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(model(x.to(device)), y.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            val = evaluate(model, loaders[1], device)
            history.append(val)
            if val["loss"] < best_loss:
                best_loss = val["loss"]
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
            print(f'{kind}: epoch {epoch+1}/{CONFIG["epochs"]} validation loss {val["loss"]:.3f}', flush=True)
        elapsed = time.perf_counter() - start
        model.load_state_dict(best_state)
        test = evaluate(model, loaders[2], device)
        results[kind] = dict(parameters=sum(p.numel() for p in model.parameters()),
                             training_seconds=elapsed, validation_history=history, best_epoch=best_epoch, test=test)
        for seed in ['my lord', 'i pray you', 'the king is']:
            for temp in CONFIG['temperatures']:
                # Reset before each draw; same random stream for all models.
                set_seed(CONFIG['seed'] + sum(map(ord, seed)) + round(temp*100))
                item = generate(model, seed, stoi, vocab, temp, device)
                all_samples.append(dict(model=kind, **item))
        del model
    stats['device'] = str(device)
    report = dict(dataset=stats, models=results, samples=all_samples)
    (output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (output / 'generated_samples.txt').open('w', encoding='utf-8') as f:
        for s in all_samples:
            f.write(f'{s["model"]} | seed: {s["seed"]} | temperature: {s["temperature"]}\n{s["generated"]}\n\n')
    plt.figure(figsize=(8, 4.8))
    for kind, vals in results.items():
        plt.plot(range(1, CONFIG['epochs']+1), [v['loss'] for v in vals['validation_history']], marker='o', label=kind)
    plt.xlabel('Epoch'); plt.ylabel('Validation cross-entropy'); plt.title('Next-word prediction')
    plt.xticks(range(1, CONFIG['epochs']+1)); plt.grid(alpha=.2); plt.legend()
    plt.tight_layout(); plt.savefig(output / 'validation_loss.png', dpi=170); plt.close()
    print('\nHeld-out test results')
    for kind, vals in results.items():
        print(kind, vals['test'], 'training seconds:', round(vals['training_seconds'], 1))
    print('Saved results in', output)
    return report

if __name__ == '__main__':
    report = main()
