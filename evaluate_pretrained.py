"""
Avalia o checkpoint oficial pré-treinado do AdaFace (sem fine-tuning),
calculando as mesmas métricas usadas para acompanhar o fine-tuning, para
servir de baseline ("modelo puro").

Métricas:
  1. Verificação facial (AUC + acurácia no melhor threshold), no protocolo
     oficial do QMUL-SurvFace (Face_Verification_Test_Set). Reaproveita
     evaluate_verification.py -- métrica diretamente comparável antes/depois
     do fine-tuning.

  2. Acurácia closed-set por centróide no split treino/validação do
     training_set (mesmo split de train.py): para cada identidade, calcula o
     centróide (média normalizada) dos embeddings das imagens de treino, e
     classifica cada imagem de validação pelo centróide mais próximo
     (similaridade de cosseno). O checkpoint pré-treinado não tem uma
     AdaFaceHead treinada para estas 5319 identidades, então esta é a
     métrica mais próxima de "val_acc" sem treinar uma head -- e pode ser
     recalculada da mesma forma após o fine-tuning para comparação direta.

Uso:
    python evaluate_pretrained.py --checkpoint adaface_ir50_ms1mv2.ckpt
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from backbone import build_model
from dataset import QMULSurvFaceDataset, split_dataset
from train import load_pretrained_backbone
from evaluate_verification import load_pairs, extract_embeddings, roc_auc, best_accuracy, tar_at_far


@torch.no_grad()
def embed_loader(backbone, loader, device):
    backbone.eval()
    all_embeddings = []
    all_labels = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        feats = F.normalize(backbone(images), dim=1).cpu()
        all_embeddings.append(feats)
        all_labels.append(labels)
    return torch.cat(all_embeddings), torch.cat(all_labels)


def closed_set_accuracy(train_emb, train_labels, val_emb, val_labels, num_classes):
    embedding_dim = train_emb.size(1)
    centroids = torch.zeros(num_classes, embedding_dim)
    for c in range(num_classes):
        centroids[c] = train_emb[train_labels == c].mean(dim=0)
    centroids = F.normalize(centroids, dim=1)

    sims = val_emb @ centroids.T  # [N_val, num_classes]
    preds = sims.argmax(dim=1)
    return (preds == val_labels).float().mean().item()


def main():
    parser = argparse.ArgumentParser(description="Avaliação do checkpoint pré-treinado (baseline sem fine-tuning)")
    parser.add_argument("--checkpoint", default="adaface_ir50_ms1mv2.ckpt",
                        help="Checkpoint oficial do AdaFace (PyTorch Lightning)")
    parser.add_argument("--arch", default="ir_50", choices=["ir_18", "ir_50", "ir_100"])
    parser.add_argument("--embedding-dim", type=int, default=512)

    parser.add_argument("--data-root", default="data/QMUL-SurvFace/training_set")
    parser.add_argument("--val-per-identity", type=int, default=1,
                        help="Mesmo valor usado em train.py, para usar o mesmo split")
    parser.add_argument("--val-seed", type=int, default=42)

    parser.add_argument("--verification-root", default="data/QMUL-SurvFace/Face_Verification_Test_Set")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", default=None,
                        help="Caminho do relatório .md (padrão: results/<nome_do_checkpoint>_pretrained.md)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    backbone = build_model(args.arch, embedding_dim=args.embedding_dim).to(device)
    load_pretrained_backbone(backbone, args.checkpoint, device)
    backbone.eval()

    # ── 1. Verificação facial (AUC + acurácia) ─────────────────────────────
    print("\n== Verificação facial (Face_Verification_Test_Set) ==")
    verification_root = Path(args.verification_root)
    images_dir = verification_root / "verification_images"

    positive_pairs = load_pairs(verification_root / "positive_pairs_names.mat")
    negative_pairs = load_pairs(verification_root / "negative_pairs_names.mat")
    print(f"Pares positivos: {len(positive_pairs)} | Pares negativos: {len(negative_pairs)}")

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

    # ── 2. Acurácia closed-set por centróide (training_set, mesmo split) ──
    print("\n== Acurácia closed-set por centróide (training_set) ==")
    dataset_full = QMULSurvFaceDataset(args.data_root, transform=transform)
    train_indices, val_indices = split_dataset(
        dataset_full, val_per_identity=args.val_per_identity, seed=args.val_seed
    )
    train_dataset = Subset(dataset_full, train_indices)
    val_dataset = Subset(dataset_full, val_indices)
    print(f"Dataset: {len(dataset_full)} imagens, {dataset_full.num_classes} identidades "
          f"({len(train_dataset)} treino / {len(val_dataset)} validação)")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    train_emb, train_labels = embed_loader(backbone, train_loader, device)
    val_emb, val_labels = embed_loader(backbone, val_loader, device)

    val_acc = closed_set_accuracy(train_emb, train_labels, val_emb, val_labels, dataset_full.num_classes)
    print(f"val_acc (centróide, closed-set): {val_acc:.4f}")

    # ── Relatório ───────────────────────────────────────────────────────────
    output_path = args.output
    if output_path is None:
        output_path = os.path.join("results", f"{Path(args.checkpoint).stem}_pretrained.md")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Avaliação do checkpoint pré-treinado (baseline) - QMUL-SurvFace\n\n")
        f.write(f"- Checkpoint: `{args.checkpoint}`\n\n")

        f.write("## Verificação facial\n\n")
        f.write(f"- Pares positivos: {len(positive_pairs)} | Pares negativos: {len(negative_pairs)}\n")
        f.write(f"- AUC: {auc:.4f}\n")
        f.write(f"- Acurácia de verificação (melhor threshold={thr:.4f}): {acc:.4f}\n\n")
        f.write("### TAR@FAR\n\n")
        f.write("| FAR alvo | FAR real | TAR | threshold |\n")
        f.write("|---|---|---|---|\n")
        for far_target, far_real, tar, thr_far in far_results:
            f.write(f"| {far_target:.3f} | {far_real:.4f} | {tar:.4f} | {thr_far:.4f} |\n")

        f.write("\n## Acurácia closed-set por centróide (training_set)\n\n")
        f.write(f"- Dataset: {len(dataset_full)} imagens, {dataset_full.num_classes} identidades "
                f"({len(train_dataset)} treino / {len(val_dataset)} validação)\n")
        f.write(f"- val_acc (centróide, closed-set): {val_acc:.4f}\n")

    print(f"\nRelatório salvo em {output_path}")


if __name__ == "__main__":
    main()
