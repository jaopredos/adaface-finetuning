"""
AdaFace Loss Head.

A ideia central do AdaFace (Kim et al., CVPR 2022):
  → A norma do embedding é um proxy para a qualidade da imagem.
  → Imagens de alta qualidade (norma alta) recebem margem MAIOR → treinamento mais difícil.
  → Imagens de baixa qualidade (norma baixa, e.g. vigilância desfocada) recebem margem MENOR
    → evitamos punir o modelo por imagens irreconhecíveis.

Isso é especialmente relevante para QMUL-SurvFace onde há grande variação de qualidade.

Fórmula da margem adaptativa:
    norm_hat   = clip(norm, 0.001, 100)
    scaled     = (norm_hat - mean_norm) / (std_norm + eps)   # normaliza para média 0
    scaled     = clip(scaled, z_min, z_max)                  # clamp assimétrico (padrão: [-1, 0])
    margin     = m + h * scaled * m                           # margem adaptativa
    logit_adj  = cos(θ + margin)                             # aplicada ao logit correto

Referência: https://arxiv.org/abs/2204.09416
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaFaceHead(nn.Module):
    """
    Head classificador com perda AdaFace.

    Args:
        embedding_dim: dimensão do embedding do backbone (512).
        num_classes:   número de identidades no training set (5319).
        m:             margem base (0.4). Aumentar → mais discriminativo, mas instável.
        h:             força da adaptação pela qualidade (0.333 = padrão do paper).
        s:             escala do logit / temperatura (64.0).
        t_alpha:       taxa de atualização da média exponencial da norma (0.01).
        z_min:         limite inferior do clamp de ẑ (-1.0 = padrão do paper).
        z_max:         limite superior do clamp de ẑ (0.0 por padrão, assimétrico).
                       Em datasets de vigilância (ex.: QMUL-SurvFace), a imagem
                       "menos pior" de um batch todo ruim pode ter ẑ ≈ +1 e
                       receber a margem máxima, como se fosse alta qualidade —
                       o que é conceitualmente errado. Com z_max=0, a margem
                       nunca excede `m` (sem bônus por qualidade), mas ainda
                       pode cair até m*(1-h) para imagens de baixa qualidade.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        num_classes: int = 5319,
        m: float = 0.4,
        h: float = 0.333,
        s: float = 64.0,
        t_alpha: float = 0.01,
        z_min: float = -1.0,
        z_max: float = 0.0,
    ):
        super().__init__()
        self.m = m
        self.h = h
        self.s = s
        self.t_alpha = t_alpha
        self.z_min = z_min
        self.z_max = z_max

        # Pesos do classificador: cada linha = protótipo de uma identidade no espaço unitário
        self.weight = nn.Parameter(torch.normal(0, 0.01, (num_classes, embedding_dim)))

        # Estatísticas de norma: média móvel exponencial para estabilidade
        # Não são parâmetros treináveis — são buffers atualizados manualmente
        self.register_buffer("batch_mean_norm", torch.ones(1) * 20)
        self.register_buffer("batch_std_norm",  torch.ones(1))

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            embeddings: [B, embedding_dim] — saída do backbone (não normalizado)
            labels:     [B] — índice da identidade (0-indexed)

        Returns:
            loss: escalar
            logits: [B, num_classes] — para calcular acurácia
        """
        # ── 0. Garante float32 em todo o head ────────────────────────────────
        # O backbone roda em float16 (AMP), mas arccos/cos da margem adaptativa
        # são numericamente instáveis em float16. O head inteiro fica em float32.
        # O custo é mínimo: o head é leve comparado ao backbone.
        embeddings = embeddings.float()

        # ── 1. Norma do embedding (qualidade proxy) ───────────────────────────
        norms = torch.norm(embeddings, dim=1, keepdim=True).clamp(min=1e-4)

        # ── 2. Normaliza embeddings e pesos para a hiperesfera unitária ───────
        embeddings_norm = F.normalize(embeddings, dim=1)
        weight_norm = F.normalize(self.weight.float(), dim=1)

        # ── 3. Similaridade cosseno: cos(θ_j) para cada classe j ─────────────
        # [B, num_classes]
        cosine = F.linear(embeddings_norm, weight_norm).clamp(-1 + 1e-7, 1 - 1e-7)

        # ── 4. Atualiza estatísticas de norma (média móvel exponencial) ───────
        safe_norms = norms.detach().squeeze(1)
        batch_mean = safe_norms.mean()
        batch_std  = safe_norms.std().clamp(min=1e-4)

        self.batch_mean_norm = (
            (1 - self.t_alpha) * self.batch_mean_norm + self.t_alpha * batch_mean
        )
        self.batch_std_norm = (
            (1 - self.t_alpha) * self.batch_std_norm + self.t_alpha * batch_std
        )

        # ── 5. Margem adaptativa ──────────────────────────────────────────────
        # Normaliza a norma do batch atual em relação à média histórica
        norm_hat = (safe_norms - self.batch_mean_norm) / (self.batch_std_norm + 1e-3)
        # Clamp assimétrico: por padrão [-1, 0] em vez do [-1, 1] do paper original.
        # Evita que a imagem "menos pior" de um batch todo ruim (ẑ -> +1)
        # receba a margem máxima como se fosse de alta qualidade.
        norm_hat = norm_hat.clamp(self.z_min, self.z_max)

        # Margem: maior para imagens "boas" (norma alta), menor para imagens "ruins"
        # [B, 1] → será broadcast para [B, num_classes]
        margin = (self.m + self.h * norm_hat * self.m).unsqueeze(1)

        # ── 6. Aplica margem apenas no logit da classe correta ────────────────
        # θ_y = arccos(cos(θ_y))
        # logit_adj = cos(θ_y + margin)
        theta = torch.acos(cosine.gather(1, labels.unsqueeze(1)))  # [B, 1]
        theta_m = (theta + margin).clamp(0, math.pi)               # [B, 1]
        cos_theta_m = torch.cos(theta_m)                           # [B, 1]

        # Substitui o logit da classe correta pelo ajustado
        logits = cosine.clone()
        logits.scatter_(1, labels.unsqueeze(1), cos_theta_m)

        # ── 7. Escala e cross-entropy ─────────────────────────────────────────
        logits = logits * self.s
        loss = F.cross_entropy(logits, labels)

        return loss, logits