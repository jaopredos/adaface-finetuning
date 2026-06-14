# Fine-tuning AdaFace (IResNet-50) no QMUL-SurvFace

Projeto de fine-tuning do modelo de reconhecimento facial **AdaFace**
(checkpoint oficial `adaface_ir50_ms1mv2.ckpt`, treinado no MS1MV2) para o
cenário de **vigilância** representado pelo dataset **QMUL-SurvFace**.

A ideia central do AdaFace é usar a **norma do embedding** como proxy de
qualidade da imagem: imagens de alta qualidade recebem margem maior na loss
(treino mais discriminativo), e imagens de baixa qualidade recebem margem
menor (evita punir o modelo por imagens irreconhecíveis). Este projeto
fine-tuna esse modelo no QMUL-SurvFace e experimenta uma variação
**assimétrica** dessa margem adaptativa, mais adequada a cenários onde
*todas* as imagens são de baixa qualidade.

## 1. Organização do dataset (`data/QMUL-SurvFace/`)

```
data/QMUL-SurvFace/
├── readme.txt
├── training_set/                          ← usado para o fine-tuning
│   ├── <identidade_1>/
│   │   ├── <id>_camX_Y.jpg
│   │   └── ...
│   ├── <identidade_2>/
│   └── ...                                ← 5.319 identidades, 220.888 imagens
│
├── Face_Verification_Test_Set/            ← avaliação (open-set)
│   ├── verification_images/               ← 10.051 imagens
│   ├── positive_pairs_names.mat           ← 5.320 pares da mesma identidade
│   └── negative_pairs_names.mat           ← 5.320 pares de identidades diferentes
│
├── Face_Identification_Test_Set/          ← não usado neste projeto
│   ├── gallery/, mated_probe/, unmated_probe/, ...
│
└── Face_Verification_Evaluation/, Face_Identification_Evaluation/
    └── scripts MATLAB do benchmark oficial (não usados)
```

Pontos importantes:

- **`training_set`**: cada subpasta é uma identidade (classe), nomeada pelo
  ID da pessoa. Arquivo `<id>_camX_Y.jpg`. As 5.319 pastas = 5.319 classes
  da `AdaFaceHead`.
- **`Face_Verification_Test_Set`**: identidades **disjuntas** do
  `training_set` (cenário *open-set*). Avaliação por **similaridade de
  pares** (cosseno entre embeddings), não por classificação — por isso
  funciona tanto para o modelo pré-treinado quanto para o fine-tuned, e é a
  métrica **comparável entre os dois**.
- Os demais diretórios (`Face_Identification_*`) seguem o protocolo oficial
  do paper, mas não são usados pelos scripts deste projeto.

## 2. Arquivos do projeto

### Modelo

- **`backbone.py`** — Arquitetura **IResNet-50** (`build_model("ir_50")`),
  compatível com o checkpoint oficial do AdaFace. Recebe uma imagem
  112×112 e devolve um embedding de 512 dimensões.

- **`head.py`** — **`AdaFaceHead`**: loss de classificação com margem
  adaptativa baseada na norma do embedding.
  - `m`, `h`, `s`: margem base, força da adaptação e escala dos logits
    (padrões do paper).
  - `z_min` / `z_max`: limites do clamp de `ẑ` (norma normalizada). Padrão
    do paper era `[-1, +1]` (simétrico). Neste projeto, o padrão foi
    alterado para `[-1, 0]` (**assimétrico**): a margem nunca excede `m`
    (sem "bônus" por parecer relativamente melhor dentro do batch), mas
    ainda pode cair até `m*(1-h)` para imagens ruins — evita que a "menos
    pior" imagem de um batch todo ruim receba margem máxima como se fosse
    de alta qualidade.

### Dados

- **`dataset.py`** — `QMULSurvFaceDataset`: lê `training_set/<id>/*.jpg`,
  mapeia cada pasta para uma classe (0..5318).
  `split_dataset()`: split estratificado treino/validação, reservando
  `val_per_identity` imagens de cada identidade (seed fixa) — treino e
  validação compartilham as mesmas 5.319 classes.

### Treino

- **`train.py`** — script principal de fine-tuning:
  - Carrega `training_set` com o split treino/validação acima.
  - Constrói `IResNet-50` + `AdaFaceHead`, carrega pesos do
    `adaface_ir50_ms1mv2.ckpt` (checkpoint PyTorch Lightning).
  - Treina com SGD (LRs separados para backbone/head) + cosine annealing +
    AMP (mixed precision em GPU).
  - A cada época: loga `loss`/`acc` (treino) e `val_loss`/`val_acc`
    (validação), salva `checkpoints/epoch_NNN.pt` e, se `val_acc` melhorar,
    também `checkpoints/best.pt`.
  - Principais flags: `--epochs`, `--batch-size`, `--lr`, `--head-lr`,
    `--m`, `--h`, `--s`, `--z-min`, `--z-max`, `--out-dir`, `--resume`.

### Avaliação

- **`evaluate_verification.py`** — avalia um checkpoint (pré-treinado ou
  fine-tuned) no protocolo oficial de **verificação facial** do
  QMUL-SurvFace: extrai embeddings das `verification_images`, calcula
  similaridade de cosseno para os pares positivos/negativos e reporta:
  - **AUC**
  - **acurácia no melhor threshold**
  - **TAR@FAR** (taxa de aceites verdadeiros em pontos fixos de falsos
    positivos — 10%, 1%, 0.1% — métrica padrão em verificação facial de
    vigilância, mais robusta que "melhor threshold" porque não escolhe o
    limiar olhando para o teste todo).
  - Salva um relatório `.md` em `results/`.

- **`evaluate_pretrained.py`** — mesma avaliação de verificação, mas para o
  **checkpoint oficial sem fine-tuning** (baseline). Adicionalmente calcula
  uma **acurácia closed-set por centróide**: centróide de cada identidade
  calculado com o split de treino, classificação do split de validação pelo
  centróide mais próximo (cosseno). Serve como substituto de `val_acc` para
  o modelo puro (que não tem uma `AdaFaceHead` treinada nessas classes).
  ⚠️ Essa métrica de centróide **não é comparável** com o `val_acc` do
  modelo fine-tuned, pois para este o `training_set` é dado de treino
  (ver seção 4). A métrica comparável é a de verificação.

### Visualização

- **`plot_training.py`** — lê os logs do `train.py` (linhas
  `== epoca N concluida: loss=... val_acc=... ==`) de múltiplos runs e
  plota um painel comparando loss/acurácia de treino e validação entre
  eles.

## 3. Fluxo de trabalho

```
1. python evaluate_pretrained.py --checkpoint adaface_ir50_ms1mv2.ckpt
   → baseline do modelo "puro" (AUC, TAR@FAR, acurácia closed-set)

2. python train.py --epochs 20 --batch-size 64 --z-min -1 --z-max <valor>
   → fine-tuning no QMUL-SurvFace, salva checkpoints/best.pt

3. python evaluate_verification.py --checkpoint checkpoints/best.pt
   → métricas de verificação do modelo fine-tuned (comparáveis ao passo 1)

4. python plot_training.py --runs logs/train_*.log:<rótulo> ...
   → gráficos comparando curvas de treino entre diferentes configurações
```

## 4. Experimento atual: clipping assimétrico de `ẑ`

Comparação do `z_max` (mantendo `z_min = -1`) em `{0.3, 0.5, 0.7}` —
posições intermediárias entre o novo padrão assimétrico (`0`) e o padrão
simétrico original do paper (`+1`). Cada configuração é treinada
separadamente (`--out-dir checkpoints_z0X`, `logs/train_z0X.log`) e
comparada via:
- curvas de treino/validação (`plot_training.py`);
- métricas de verificação (AUC/TAR@FAR) de cada `best.pt` via
  `evaluate_verification.py`, comparáveis entre si e com o baseline
  pré-treinado do passo 1.
