# Avaliação de verificação facial - QMUL-SurvFace

- Checkpoint: `checkpoints_z07/best.pt` (época 18)
- Pares positivos: 5320 | Pares negativos: 5320
- AUC: 0.8873
- Acurácia de verificação (melhor threshold=0.1065): 0.8162

## TAR@FAR

| FAR alvo | FAR real | TAR | threshold |
|---|---|---|---|
| 0.100 | 0.1000 | 0.7305 | 0.1156 |
| 0.010 | 0.0100 | 0.5045 | 0.2158 |
| 0.001 | 0.0009 | 0.3164 | 0.3214 |
