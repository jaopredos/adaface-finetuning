"""
Plota curvas de treino/validação (loss e acurácia) a partir dos logs do
train.py, comparando múltiplos runs no mesmo gráfico.

Cada run é especificado como "<arquivo_de_log>:<rótulo>".

Uso:
    python plot_training.py \
        --runs logs/train_z03.log:z_max=0.3 logs/train_z05.log:z_max=0.5 logs/train_z07.log:z_max=0.7 \
        --output results/comparacao_z.png
"""
import argparse
import os
import re

import matplotlib.pyplot as plt


EPOCH_RE = re.compile(
    r"== epoca (\d+) concluida: loss=([\d.]+) acc=([\d.]+) "
    r"\| val_loss=([\d.]+) val_acc=([\d.]+) \| tempo=([\d.]+)s =="
)


def parse_log(log_path):
    epochs, loss, acc, val_loss, val_acc = [], [], [], [], []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = EPOCH_RE.search(line)
            if match:
                epochs.append(int(match.group(1)))
                loss.append(float(match.group(2)))
                acc.append(float(match.group(3)))
                val_loss.append(float(match.group(4)))
                val_acc.append(float(match.group(5)))
    return {"epoch": epochs, "loss": loss, "acc": acc, "val_loss": val_loss, "val_acc": val_acc}


def main():
    parser = argparse.ArgumentParser(description="Compara curvas de treino entre runs")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Lista de '<log>:<rótulo>', ex.: logs/train_z03.log:z_max=0.3")
    parser.add_argument("--output", default="results/comparacao_treino.png")
    args = parser.parse_args()

    runs = []
    for item in args.runs:
        log_path, _, label = item.partition(":")
        label = label or log_path
        runs.append((label, parse_log(log_path)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for label, data in runs:
        axes[0, 0].plot(data["epoch"], data["loss"], label=label)
        axes[0, 1].plot(data["epoch"], data["acc"], label=label)
        axes[1, 0].plot(data["epoch"], data["val_loss"], label=label)
        axes[1, 1].plot(data["epoch"], data["val_acc"], label=label)

    axes[0, 0].set_title("Loss (treino)")
    axes[0, 1].set_title("Acurácia (treino)")
    axes[1, 0].set_title("Loss (validação)")
    axes[1, 1].set_title("Acurácia (validação)")

    for ax in axes.flat:
        ax.set_xlabel("Época")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Gráfico salvo em {args.output}")


if __name__ == "__main__":
    main()
