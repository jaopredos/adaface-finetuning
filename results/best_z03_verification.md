# Avaliação de verificação facial - QMUL-SurvFace

- Checkpoint: `checkpoints_z03/best.pt` (época 18)
- Pares positivos: 5320 | Pares negativos: 5320
- AUC: 0.8926
- Acurácia de verificação (melhor threshold=0.1029): 0.8203

## TAR@FAR

| FAR alvo | FAR real | TAR | threshold |
|---|---|---|---|
| 0.100 | 0.1000 | 0.7357 | 0.1114 |
| 0.010 | 0.0100 | 0.5002 | 0.2196 |
| 0.001 | 0.0009 | 0.2848 | 0.3389 |
