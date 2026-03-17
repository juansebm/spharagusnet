#!/usr/bin/env python3
"""
Entrenar modelo de clasificación de bloques OCR.

Cambios vs. versión anterior:
- **--resume**: carga checkpoint anterior y continúa entrenando (no parte de cero).
- **Carga desde cache**: lee .labels.json (etiquetados) en vez de re-correr OCR.
- **Vocab merge**: al resumir, expande el vocabulario con palabras nuevas sin
  perder los embeddings aprendidos.
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Paquete local
sys.path.insert(0, str(Path(__file__).parent.parent))
from spharagusnet.model import DocumentTextExtractor, DEVICE
from spharagusnet.tokenizer import PAD_TOKEN, UNK_TOKEN, MAX_SEQ_LEN

# ── Configuración ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "data" / "dataset_entrenamiento"
MODEL_DIR = BASE_DIR / "models" / "texto_extractor"
IMAGENES_DIR = DATASET_DIR / "imagenes"

print(f"🖥️  Dispositivo: {DEVICE}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.2f} GB"
          if hasattr(torch.cuda.get_device_properties(0), 'total_mem')
          else f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")


# ── Vocabulario ──────────────────────────────────────────────────────────────

def construir_vocabulario(datos: List[Dict], max_vocab: int = 10000) -> Dict[str, int]:
    counter: Counter = Counter()
    for ej in datos:
        for bloque in ej["bloques"]:
            counter.update(bloque["texto"].lower().split())
    vocab: Dict[str, int] = {"<PAD>": PAD_TOKEN, "<UNK>": UNK_TOKEN}
    for palabra, _ in counter.most_common(max_vocab - 2):
        vocab[palabra] = len(vocab)
    print(f"   Vocabulario: {len(vocab)} tokens (de {len(counter)} únicas)")
    return vocab


def merge_vocab(old_vocab: Dict[str, int], new_vocab: Dict[str, int]) -> Dict[str, int]:
    """Expande old_vocab con tokens nuevos de new_vocab, sin reasignar IDs existentes."""
    merged = dict(old_vocab)
    next_id = max(merged.values()) + 1
    for token in new_vocab:
        if token not in merged:
            merged[token] = next_id
            next_id += 1
    return merged


# ── Tokenización ─────────────────────────────────────────────────────────────

def tokenizar(texto: str, vocab: Dict[str, int],
              max_len: int = MAX_SEQ_LEN) -> List[int]:
    palabras = texto.lower().split()
    tokens = [vocab.get(p, UNK_TOKEN) for p in palabras[:max_len]]
    tokens += [PAD_TOKEN] * (max_len - len(tokens))
    return tokens


# ── Dataset ──────────────────────────────────────────────────────────────────

class BloqueDataset(Dataset):
    def __init__(self, datos: List[Dict], vocab: Dict[str, int]):
        self.samples: List[Tuple[List[int], int]] = []
        self.n_pos = 0
        self.n_neg = 0
        for ej in datos:
            for bloque in ej["bloques"]:
                tokens = tokenizar(bloque["texto"], vocab)
                label = 1 if bloque.get("relevante", False) else 0
                if label == 1:
                    self.n_pos += 1
                else:
                    self.n_neg += 1
                self.samples.append((tokens, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        tokens, label = self.samples[idx]
        return (torch.tensor(tokens, dtype=torch.long),
                torch.tensor(label, dtype=torch.long))


# ── Métricas ─────────────────────────────────────────────────────────────────

def calcular_metricas(preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


@torch.no_grad()
def evaluar(model, dl, criterion):
    model.eval()
    total_loss = 0.0
    all_p, all_l = [], []
    n = 0
    for tokens, labels in dl:
        tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)
        pos = torch.zeros(tokens.size(0), tokens.size(1), 2, device=DEVICE)
        _, scores = model(tokens, pos)
        total_loss += criterion(scores, labels).item()
        n += 1
        all_p.append(scores.argmax(1).cpu())
        all_l.append(labels.cpu())
    m = calcular_metricas(torch.cat(all_p), torch.cat(all_l))
    m["loss"] = total_loss / max(n, 1)
    return m


# ── Preparación de datos (desde cache) ──────────────────────────────────────

def cargar_datos_desde_cache() -> Tuple[List[Dict], List[Dict]]:
    """
    Lee .labels.json cacheados (generados por preparar_dataset_entrenamiento.py).
    Es instantáneo vs. re-correr OCR.
    """
    t0 = time.time()
    indice_path = DATASET_DIR / "indice.json"
    if not indice_path.exists():
        print(f"❌ No se encontró {indice_path}")
        print("   Ejecuta primero: python scripts/preparar_dataset_entrenamiento.py")
        return [], []

    with open(indice_path, "r", encoding="utf-8") as f:
        indice = json.load(f)

    datos: List[Dict] = []
    total = len(indice["registros"])

    for i, reg in enumerate(indice["registros"], 1):
        for img_rel in reg["imagenes"]:
            img_path = DATASET_DIR / img_rel
            labels_path = img_path.with_suffix(".labels.json")

            if not labels_path.exists():
                # Intentar el cache OCR sin labels
                ocr_path = img_path.with_suffix(".ocr.json")
                if not ocr_path.exists():
                    continue
                # Si hay OCR pero no labels, marcar todo como "sin label"
                # (no debería pasar si corriste preparar_dataset primero)
                print(f"   ⚠️  {labels_path.name} no existe, saltando")
                continue

            with open(labels_path, "r", encoding="utf-8") as f:
                bloques = json.load(f)

            if bloques:
                datos.append({
                    "imagen": str(img_path),
                    "bloques": bloques,
                })

        if i % 10 == 0 or i == total:
            print(f"   [{i}/{total}] registros cargados — {len(datos)} ejemplos")

    dt = time.time() - t0
    print(f"✅ {len(datos)} ejemplos cargados en {dt:.1f}s (desde cache)")

    # Split 80/20
    split = int(len(datos) * 0.8)
    return datos[:split], datos[split:]


# ── Entrenamiento ────────────────────────────────────────────────────────────

def formato_tiempo(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        m, s = divmod(s, 60)
        return f"{int(m)}m {s:.0f}s"
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{int(h)}h {int(m)}m"


def entrenar(datos_train: List[Dict], datos_val: List[Dict],
             resume: bool = False, epochs: int = 20, patience: int = 5):
    t0 = time.time()

    # ── Vocabulario ──────────────────────────────────────────────────────
    print("\n📖 Vocabulario...")
    new_vocab = construir_vocabulario(datos_train + datos_val)

    old_vocab = None
    checkpoint = None
    start_epoch = 0
    best_val_f1 = 0.0

    if resume:
        model_path = MODEL_DIR / "modelo_entrenado.pth"
        vocab_path = MODEL_DIR / "vocab.json"
        if model_path.exists() and vocab_path.exists():
            print("   📦 Cargando checkpoint anterior...")
            with open(vocab_path, "r", encoding="utf-8") as f:
                old_vocab = json.load(f)
            checkpoint = torch.load(model_path, map_location=DEVICE,
                                    weights_only=False)
            start_epoch = checkpoint.get("epoch", 0)
            best_val_f1 = checkpoint.get("val_f1", 0.0)
            print(f"   Resumiendo desde epoch {start_epoch}, "
                  f"best val-F1={best_val_f1:.4f}")
        else:
            print("   ⚠️  No hay checkpoint, entrenando desde cero")

    # Merge vocabularios si estamos resumiendo
    if old_vocab is not None:
        vocab = merge_vocab(old_vocab, new_vocab)
        nuevas = len(vocab) - len(old_vocab)
        print(f"   Vocab merged: {len(old_vocab)} → {len(vocab)} (+{nuevas} nuevas)")
    else:
        vocab = new_vocab

    vocab_size = len(vocab)

    # ── Datasets ─────────────────────────────────────────────────────────
    print("📦 Datasets...")
    ds_train = BloqueDataset(datos_train, vocab)
    ds_val = BloqueDataset(datos_val, vocab)
    print(f"   Train: {len(ds_train)} bloques (pos={ds_train.n_pos}, neg={ds_train.n_neg})")
    print(f"   Val  : {len(ds_val)} bloques (pos={ds_val.n_pos}, neg={ds_val.n_neg})")

    if not ds_train:
        print("⚠️  Dataset vacío")
        return

    batch_size = 32
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    # ── Modelo ───────────────────────────────────────────────────────────
    hidden_dim = 512
    model = DocumentTextExtractor(vocab_size=vocab_size, hidden_dim=hidden_dim,
                                  device=DEVICE)

    if checkpoint is not None:
        old_state = checkpoint["model_state_dict"]
        old_vocab_size = checkpoint.get("vocab_size", len(old_vocab))

        if vocab_size > old_vocab_size:
            # Expandir embedding y output layer para nuevos tokens
            print(f"   🔧 Expandiendo modelo: {old_vocab_size} → {vocab_size} tokens")
            model.load_state_dict(old_state, strict=False)
            # Copiar pesos antiguos al nuevo embedding
            with torch.no_grad():
                old_emb = old_state["text_embedding.weight"]
                model.text_embedding.weight[:old_vocab_size] = old_emb
                old_out = old_state["output_layer.weight"]
                old_bias = old_state["output_layer.bias"]
                model.output_layer.weight[:old_vocab_size] = old_out
                model.output_layer.bias[:old_vocab_size] = old_bias
        else:
            model.load_state_dict(old_state)

    # ── Pesos de clase ───────────────────────────────────────────────────
    if ds_train.n_pos > 0 and ds_train.n_neg > 0:
        w = ds_train.n_neg / ds_train.n_pos
        weights = torch.tensor([1.0, w], dtype=torch.float32, device=DEVICE)
        print(f"   Class weights: neg=1.00, pos={w:.2f}")
    else:
        weights = None

    criterion = nn.CrossEntropyLoss(weight=weights).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    if checkpoint is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("   ✓ Optimizer state restaurado")
        except Exception:
            print("   ⚠️  No se pudo restaurar optimizer (cambió vocab), usando nuevo")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Guardar vocabulario
    with open(MODEL_DIR / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    # ── Loop de entrenamiento ────────────────────────────────────────────
    epochs_sin_mejora = 0
    best_epoch = start_epoch
    historial: List[Dict] = []

    # Cargar historial previo si existe
    hist_path = MODEL_DIR / "historial_entrenamiento.json"
    if resume and hist_path.exists():
        with open(hist_path, "r", encoding="utf-8") as f:
            historial = json.load(f)

    col = (f"{'Epoch':>7} │ {'Loss':>7} {'Acc':>6} {'Prec':>6} {'Rec':>6} "
           f"{'F1':>6} │ {'vLoss':>7} {'vAcc':>6} {'vPrec':>6} {'vRec':>6} "
           f"{'vF1':>6} │ {'t':>6} {'ETA':>8}")
    sep = "─" * len(col)

    total_epochs = start_epoch + epochs
    print(f"\n🔄 Epochs {start_epoch + 1}→{total_epochs}, batch={batch_size}, "
          f"patience={patience}")
    print(sep)
    print(col)
    print(sep)

    t_train = time.time()

    for ep_i in range(epochs):
        ep = start_epoch + ep_i + 1
        t_ep = time.time()

        # Train
        model.train()
        run_loss = 0.0
        ep_p, ep_l = [], []
        nb = 0
        for tokens, labels in dl_train:
            tokens, labels = tokens.to(DEVICE), labels.to(DEVICE)
            pos = torch.zeros(tokens.size(0), tokens.size(1), 2, device=DEVICE)
            _, scores = model(tokens, pos)
            loss = criterion(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            nb += 1
            ep_p.append(scores.argmax(1).cpu())
            ep_l.append(labels.cpu())

        tr_loss = run_loss / max(nb, 1)
        tr = calcular_metricas(torch.cat(ep_p), torch.cat(ep_l))

        # Val
        vl = evaluar(model, dl_val, criterion) if len(ds_val) else {
            "loss": 0, "accuracy": 0, "precision": 0, "recall": 0, "f1": 0,
        }

        dt = time.time() - t_ep
        elapsed = time.time() - t_train
        eta = (elapsed / (ep_i + 1)) * (epochs - ep_i - 1)

        print(f" {ep:>3}/{total_epochs:<3} │ "
              f"{tr_loss:7.4f} {tr['accuracy']:6.3f} {tr['precision']:6.3f} "
              f"{tr['recall']:6.3f} {tr['f1']:6.3f} │ "
              f"{vl['loss']:7.4f} {vl['accuracy']:6.3f} {vl['precision']:6.3f} "
              f"{vl['recall']:6.3f} {vl['f1']:6.3f} │ "
              f"{dt:5.1f}s {formato_tiempo(eta):>8}")

        historial.append({
            "epoch": ep,
            "train_loss": tr_loss,
            "train_acc": tr["accuracy"], "train_prec": tr["precision"],
            "train_rec": tr["recall"], "train_f1": tr["f1"],
            "val_loss": vl["loss"],
            "val_acc": vl["accuracy"], "val_prec": vl["precision"],
            "val_rec": vl["recall"], "val_f1": vl["f1"],
        })

        # Early stopping / save best
        if vl["f1"] > best_val_f1:
            best_val_f1 = vl["f1"]
            best_epoch = ep
            epochs_sin_mejora = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": ep,
                "val_f1": best_val_f1,
                "vocab_size": vocab_size,
                "hidden_dim": hidden_dim,
                "device": str(DEVICE),
            }, MODEL_DIR / "modelo_entrenado.pth")
        else:
            epochs_sin_mejora += 1
            if epochs_sin_mejora >= patience:
                print(f"\n⏹️  Early stopping (sin mejora en {patience} epochs)")
                break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(sep)

    # Guardar historial
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(historial, f, ensure_ascii=False, indent=2)

    print(f"\n📊 Resumen")
    print(f"   Mejor epoch : {best_epoch}")
    print(f"   Mejor val-F1: {best_val_f1:.4f}")
    print(f"   Modelo      : {MODEL_DIR / 'modelo_entrenado.pth'}")
    print(f"   ⏱️  {formato_tiempo(time.time() - t0)}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Entrena clasificador de bloques")
    parser.add_argument("--resume", action="store_true",
                        help="Continuar desde el último checkpoint")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Número de epochs (default: 20)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience (default: 5)")
    args = parser.parse_args()

    print("🚀 Entrenamiento de Modelo")
    print("=" * 80)

    datos_train, datos_val = cargar_datos_desde_cache()
    if not datos_train:
        print("⚠️  Sin datos de entrenamiento")
        return

    entrenar(datos_train, datos_val,
             resume=args.resume, epochs=args.epochs, patience=args.patience)

    print("\n" + "=" * 80)
    print("✅ Completado")


if __name__ == "__main__":
    main()
