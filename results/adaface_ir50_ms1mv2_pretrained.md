# Avaliação do checkpoint pré-treinado (baseline) - QMUL-SurvFace

- Checkpoint: `adaface_ir50_ms1mv2.ckpt`

## Verificação facial

- Pares positivos: 5320 | Pares negativos: 5320
- AUC: 0.6592
- Acurácia de verificação (melhor threshold=0.2751): 0.6197

### TAR@FAR

| FAR alvo | FAR real | TAR | threshold |
|---|---|---|---|
| 0.100 | 0.1000 | 0.2600 | 0.3990 |
| 0.010 | 0.0100 | 0.0660 | 0.5323 |
| 0.001 | 0.0009 | 0.0211 | 0.6142 |

## Acurácia closed-set por centróide (training_set)

- Dataset: 220888 imagens, 5319 identidades (215569 treino / 5319 validação)
- val_acc (centróide, closed-set): 0.0865
