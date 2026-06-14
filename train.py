"""
Script de treinamento do backbone IResNet-50 + AdaFaceHead no QMUL-SurvFace.

Fluxo:
    1. Carrega o dataset de treino (data/QMUL-SurvFace/training_set).
    2. Constrói o backbone IResNet-50 e carrega os pesos pré-treinados
       (adaface_ir50_ms1mv2.ckpt), permitindo fine-tuning.
    3. Constrói a AdaFaceHead com num_classes = número de identidades do dataset.
    4. Treina com SGD + cosine annealing + mixed precision (AMP) quando há GPU.
    5. Salva checkpoints (backbone, head, otimizador, scheduler) por época.

Uso:
    python train.py --epochs 20 --batch-size 64
    python train.py --resume checkpoints/epoch_005.pt
"""
import argparse
import os
import time

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from backbone import build_model
from head import AdaFaceHead
from dataset import QMULSurvFaceDataset, split_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Treino AdaFace (IResNet-50) no QMUL-SurvFace")

    parser.add_argument("--data-root", default="data/QMUL-SurvFace/training_set",
                        help="Pasta com uma subpasta por identidade")
    parser.add_argument("--checkpoint", default="adaface_ir50_ms1mv2.ckpt",
                        help="Checkpoint pré-treinado do backbone (carregado apenas na 1a execução)")
    parser.add_argument("--arch", default="ir_50", choices=["ir_18", "ir_50", "ir_100"])
    parser.add_argument("--embedding-dim", type=int, default=512)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-3, help="LR do backbone")
    parser.add_argument("--head-lr", type=float, default=1e-2, help="LR da AdaFaceHead")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)

    parser.add_argument("--m", type=float, default=0.4, help="Margem base do AdaFace")
    parser.add_argument("--h", type=float, default=0.333, help="Força da adaptação por qualidade")
    parser.add_argument("--s", type=float, default=64.0, help="Escala dos logits")
    parser.add_argument("--z-min", type=float, default=-1.0, help="Limite inferior do clamp de ẑ")
    parser.add_argument("--z-max", type=float, default=0.0,
                        help="Limite superior do clamp de ẑ (0.0 = clamp assimétrico, sem bônus de margem)")

    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--resume", default=None, help="Checkpoint de treino para retomar")
    parser.add_argument("--log-interval", type=int, default=50)

    parser.add_argument("--val-per-identity", type=int, default=1,
                        help="Quantidade de imagens por identidade reservadas para validação")
    parser.add_argument("--val-seed", type=int, default=42)

    return parser.parse_args()


def load_pretrained_backbone(model, ckpt_path, device):
    """Carrega os pesos do checkpoint oficial do AdaFace no backbone.

    O checkpoint oficial é salvo via PyTorch Lightning: as chaves do
    state_dict ficam sob ckpt["state_dict"] e prefixadas com "model.".
    """
    # weights_only=False: o checkpoint oficial do AdaFace é salvo via PyTorch
    # Lightning e contém objetos (ex.: ModelCheckpoint) além de tensores.
    # Seguro pois o arquivo vem de fonte confiável (mk-minchul/AdaFace).
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    model_keys = set(model.state_dict().keys())
    filtered = {}
    for key, value in state_dict.items():
        clean_key = key
        for prefix in ("model.", "module."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        if clean_key in model_keys:
            filtered[clean_key] = value

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"[checkpoint] {len(filtered)}/{len(model_keys)} chaves carregadas "
          f"(faltando={len(missing)}, inesperadas={len(unexpected)})")


@torch.no_grad()
def evaluate(backbone, head, loader, device):
    backbone.eval()
    head.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        embeddings = backbone(images)
        loss, logits = head(embeddings, labels)

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Usando dispositivo: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Dados ────────────────────────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dataset_full = QMULSurvFaceDataset(args.data_root, transform=train_transform)
    val_dataset_full = QMULSurvFaceDataset(args.data_root, transform=val_transform)
    train_indices, val_indices = split_dataset(
        train_dataset_full, val_per_identity=args.val_per_identity, seed=args.val_seed
    )
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    print(f"Dataset: {len(train_dataset_full)} imagens, {train_dataset_full.num_classes} identidades "
          f"({len(train_dataset)} treino / {len(val_dataset)} validação)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_amp,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_amp,
    )

    # ── Modelo ───────────────────────────────────────────────────────────────
    backbone = build_model(args.arch, embedding_dim=args.embedding_dim).to(device)

    head = AdaFaceHead(
        embedding_dim=args.embedding_dim,
        num_classes=train_dataset_full.num_classes,
        m=args.m,
        h=args.h,
        s=args.s,
        z_min=args.z_min,
        z_max=args.z_max,
    ).to(device)

    optimizer = torch.optim.SGD(
        [
            {"params": backbone.parameters(), "lr": args.lr},
            {"params": head.parameters(), "lr": args.head_lr},
        ],
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        backbone.load_state_dict(ckpt["backbone"])
        head.load_state_dict(ckpt["head"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"Retomando treino de '{args.resume}' a partir da época {start_epoch}")
    else:
        load_pretrained_backbone(backbone, args.checkpoint, device)

    # ── Loop de treino ───────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        backbone.train()
        head.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0
        epoch_start = time.time()

        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, enabled=use_amp):
                embeddings = backbone(images)

            loss, logits = head(embeddings, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            running_correct += (logits.argmax(dim=1) == labels).sum().item()
            running_total += images.size(0)

            if (step + 1) % args.log_interval == 0:
                avg_loss = running_loss / running_total
                acc = running_correct / running_total
                print(f"epoca {epoch} [{step + 1}/{len(train_loader)}] "
                      f"loss={avg_loss:.4f} acc={acc:.4f}")

        scheduler.step()

        epoch_loss = running_loss / running_total
        epoch_acc = running_correct / running_total
        elapsed = time.time() - epoch_start

        val_loss, val_acc = evaluate(backbone, head, val_loader, device)

        print(f"== epoca {epoch} concluida: loss={epoch_loss:.4f} acc={epoch_acc:.4f} "
              f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f} | tempo={elapsed:.1f}s ==")

        checkpoint_data = {
            "epoch": epoch,
            "backbone": backbone.state_dict(),
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "class_to_idx": train_dataset_full.class_to_idx,
            "args": vars(args),
            "val_acc": val_acc,
            "best_val_acc": max(best_val_acc, val_acc),
        }

        ckpt_path = os.path.join(args.out_dir, f"epoch_{epoch:03d}.pt")
        torch.save(checkpoint_data, ckpt_path)
        print(f"checkpoint salvo em {ckpt_path}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(args.out_dir, "best.pt")
            torch.save(checkpoint_data, best_path)
            print(f"novo melhor val_acc={val_acc:.4f} -> checkpoint salvo em {best_path}")


if __name__ == "__main__":
    main()
