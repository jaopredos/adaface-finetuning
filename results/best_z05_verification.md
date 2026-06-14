# Avaliação de verificação facial - QMUL-SurvFace

- Checkpoint: `checkpoints_z05/best.pt` (época 19)
- Pares positivos: 5320 | Pares negativos: 5320
- AUC: 0.8862
- Acurácia de verificação (melhor threshold=0.1134): 0.8148

## TAR@FAR

| FAR alvo | FAR real | TAR | threshold |
|---|---|---|---|
| 0.100 | 0.1000 | 0.7289 | 0.1140 |
| 0.010 | 0.0100 | 0.4919 | 0.2216 |
| 0.001 | 0.0009 | 0.3312 | 0.3096 |
