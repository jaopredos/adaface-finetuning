"""
Dataset para o conjunto de treino do QMUL-SurvFace.

Estrutura esperada em disco:
    data/QMUL-SurvFace/training_set/<identidade>/<identidade>_camX_Y.jpg

Cada subpasta é uma identidade (classe). O mapeamento classe -> índice
é construído ordenando os nomes das pastas, garantindo reprodutibilidade
entre execuções (desde que o conteúdo da pasta não mude).
"""
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class QMULSurvFaceDataset(Dataset):
    """
    Args:
        root: caminho para data/QMUL-SurvFace/training_set
        transform: transformações aplicadas em cada imagem (PIL -> Tensor)
    """

    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform

        identities = sorted(
            d.name for d in self.root.iterdir() if d.is_dir()
        )
        self.class_to_idx = {name: idx for idx, name in enumerate(identities)}

        self.samples = []
        for name in identities:
            label = self.class_to_idx[name]
            for img_path in sorted((self.root / name).iterdir()):
                if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((img_path, label))

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def split_dataset(dataset, val_per_identity=1, seed=42):
    """Split estratificado treino/validação.

    Reserva `val_per_identity` imagens de cada identidade (escolhidas
    aleatoriamente com seed fixa) para validação. As demais ficam no treino.
    Como o split é por identidade, treino e validação compartilham o mesmo
    conjunto de classes (head de classificação fechada).

    Returns:
        (train_indices, val_indices)
    """
    by_class = {}
    for idx, (_, label) in enumerate(dataset.samples):
        by_class.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    val_indices = []
    for indices in by_class.values():
        indices = indices.copy()
        rng.shuffle(indices)
        val_indices.extend(indices[:val_per_identity])

    val_set = set(val_indices)
    train_indices = [i for i in range(len(dataset)) if i not in val_set]
    return train_indices, val_indices
