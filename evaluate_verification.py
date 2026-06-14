"""
Avaliação do backbone treinado no protocolo oficial de Face Verification
do QMUL-SurvFace.

Usa os pares definidos em:
    data/QMUL-SurvFace/Face_Verification_Test_Set/positive_pairs_names.mat
    data/QMUL-SurvFace/Face_Verification_Test_Set/negative_pairs_names.mat

Para cada par, calcula a similaridade de cosseno entre os embeddings
(normalizados) das duas imagens. Reporta a AUC, a acurácia de verificação
no melhor threshold e TAR@FAR (taxa de aceites verdadeiros para taxas de
falsos positivos fixas) -- métrica padrão em benchmarks de verificação
facial em vigilância, pois fixa o ponto de operação pelo lado dos
negativos em vez de escolher o threshold olhando para o teste todo.

As identidades desse conjunto NÃO aparecem no training_set (cenário
open-set) — por isso a avaliação é feita por similaridade de pares, e não
pela AdaFaceHead de classificação usada no treino.

Uso:
    python evaluate_verification.py --checkpoint checkpoints/epoch_019.pt
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.io import loadmat
from torchvision import transforms

from backbone import build_model


def _to_str(value):
    """Extrai uma string de um elemento de cell array do MATLAB carregado pelo scipy."""
    while isinstance(value, np.ndarray):
        value = value[0]
    return str(value)


def load_pairs(mat_path):
    """Lê um arquivo .mat com um cell array Nx2 de nomes de imagens."""
    mat = loadmat(mat_path)
    key = next(k for k in mat.keys() if not k.startswith("__"))
    cells = mat[key]
    return [(_to_str(row[0]), _to_str(row[1])) for row in cells]


@torch.no_grad()
def extract_embeddings(backbone, image_names, images_dir, transform, device, batch_size):
    backbone.eval()
    embeddings = {}

    unique_names = sorted(set(image_names))
    for i in range(0, len(unique_names), batch_size):
        batch_names = unique_names[i:i + batch_size]
        batch_imgs = [
            transform(Image.open(images_dir / name).convert("RGB"))
            for name in batch_names
        ]
        batch_tensor = torch.stack(batch_imgs).to(device)
        feats = F.normalize(backbone(batch_tensor), dim=1).cpu()
        for name, feat in zip(batch_names, feats):
            embeddings[name] = feat

    return embeddings


def roc_auc(scores, labels):
    """AUC via estatística de Mann-Whitney (sem dependência de sklearn)."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    ranks = np.concatenate([pos, neg]).argsort().argsort()
    pos_ranks = ranks[: len(pos)]
    auc = (pos_ranks.sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def best_accuracy(scores, labels):
    """Busca o threshold (sobre os próprios scores) que maximiza a acurácia."""
    thresholds = np.unique(scores)
    best_acc, best_thr = 0.0, 0.0
    for thr in thresholds:
        acc = ((scores >= thr).astype(int) == labels).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, thr
    return best_acc, best_thr


def tar_at_far(scores, labels, far_levels=(0.1, 0.01, 0.001)):
    """TAR (True Accept Rate) para cada FAR (False Accept Rate) alvo.

    O threshold é escolhido pelos scores dos pares negativos: para
    FAR = far_level, usa-se o score que deixa aproximadamente
    `far_level * n_neg` negativos acima do threshold. TAR é a fração de
    pares positivos com score >= threshold.

    Returns:
        lista de tuplas (far_alvo, far_real, tar, threshold)
    """
    neg_scores = np.sort(scores[labels == 0])[::-1]  # descendente
    pos_scores = scores[labels == 1]
    n_neg = len(neg_scores)

    results = []
    for far in far_levels:
        idx = min(max(int(round(far * n_neg)), 1), n_neg)
        threshold = neg_scores[idx - 1]
        far_real = (neg_scores >= threshold).mean()
        tar = (pos_scores >= threshold).mean()
        results.append((far, float(far_real), float(tar), float(threshold)))
    return results


def main():
    parser = argparse.ArgumentParser(description="Avaliação de verificação facial - QMUL-SurvFace")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint de treino (epoch_XXX.pt)")
    parser.add_argument("--arch", default="ir_50", choices=["ir_18", "ir_50", "ir_100"])
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--data-root", default="data/QMUL-SurvFace/Face_Verification_Test_Set")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default=None,
                        help="Caminho do relatório .md (padrão: results/<nome_do_checkpoint>_verification.md)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    data_root = Path(args.data_root)
    images_dir = data_root / "verification_images"

    positive_pairs = load_pairs(data_root / "positive_pairs_names.mat")
    negative_pairs = load_pairs(data_root / "negative_pairs_names.mat")
    print(f"Pares positivos: {len(positive_pairs)} | Pares negativos: {len(negative_pairs)}")

    backbone = build_model(args.arch, embedding_dim=args.embedding_dim).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    backbone.load_state_dict(ckpt["backbone"])
    print(f"Checkpoint '{args.checkpoint}' (época {ckpt.get('epoch', '?')}) carregado")

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    all_pairs = positive_pairs + negative_pairs
    all_names = [name for pair in all_pairs for name in pair]
    embeddings = extract_embeddings(backbone, all_names, images_dir, transform, device, args.batch_size)

    scores, labels = [], []
    for a, b in positive_pairs:
        scores.append(torch.dot(embeddings[a], embeddings[b]).item())
        labels.append(1)
    for a, b in negative_pairs:
        scores.append(torch.dot(embeddings[a], embeddings[b]).item())
        labels.append(0)

    scores = np.array(scores)
    labels = np.array(labels)

    auc = roc_auc(scores, labels)
    acc, thr = best_accuracy(scores, labels)
    far_results = tar_at_far(scores, labels)

    print(f"AUC: {auc:.4f}")
    print(f"Acurácia de verificação (melhor threshold={thr:.4f}): {acc:.4f}")

    print("TAR@FAR:")
    for far_target, far_real, tar, thr_far in far_results:
        print(f"  FAR={far_target:.3f} (real={far_real:.4f}) -> "
              f"TAR={tar:.4f} (threshold={thr_far:.4f})")

    output_path = args.output
    if output_path is None:
        output_path = os.path.join("results", f"{Path(args.checkpoint).stem}_verification.md")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Avaliação de verificação facial - QMUL-SurvFace\n\n")
        f.write(f"- Checkpoint: `{args.checkpoint}` (época {ckpt.get('epoch', '?')})\n")
        f.write(f"- Pares positivos: {len(positive_pairs)} | Pares negativos: {len(negative_pairs)}\n")
        f.write(f"- AUC: {auc:.4f}\n")
        f.write(f"- Acurácia de verificação (melhor threshold={thr:.4f}): {acc:.4f}\n\n")
        f.write("## TAR@FAR\n\n")
        f.write("| FAR alvo | FAR real | TAR | threshold |\n")
        f.write("|---|---|---|---|\n")
        for far_target, far_real, tar, thr_far in far_results:
            f.write(f"| {far_target:.3f} | {far_real:.4f} | {tar:.4f} | {thr_far:.4f} |\n")

    print(f"\nRelatório salvo em {output_path}")


if __name__ == "__main__":
    main()
