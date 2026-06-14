# Avaliação de verificação facial - QMUL-SurvFace

- Checkpoint: `checkpoints/epoch_019.pt` (época 19)
- Pares positivos: 5320 | Pares negativos: 5320
- AUC: 0.8834
- Acurácia de verificação (melhor threshold=0.1060): 0.8092

## TAR@FAR

| FAR alvo | FAR real | TAR | threshold |
|---|---|---|---|
| 0.100 | 0.1000 | 0.7164 | 0.1222 |
| 0.010 | 0.0100 | 0.4791 | 0.2267 |
| 0.001 | 0.0009 | 0.2306 | 0.3835 |
