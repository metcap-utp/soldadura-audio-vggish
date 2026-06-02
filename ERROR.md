# Error: NaN en StatsPooling con segmentos de 1 segundo

## Problema

Los modelos entrenados con segmentos de **1 segundo** producían predicciones inválidas en inferencia. Los 3 modelos (XVector, ECAPA-TDNN, FeedForward) predecían siempre la clase mayoritaria con métricas ~29%/12%/35% en el conjunto test.

## Causa raíz

El cálculo de **desviación estándar** en las capas de pooling produce `NaN` cuando la dimensión temporal es 1:

```python
# En modelo_xvector.py, modelo_ecapa.py, modelo_feedforward.py
std = x.std(dim=2)  # Si time_steps=1 → NaN
```

Esto ocurre porque `torch.std()` usa por defecto `unbiased=True`, que calcula:
```
std = sqrt(Σ(x - mean)² / (N - 1))
```

Cuando N=1, el denominador es 0 → **NaN**.

## Solución aplicada

Cambiar a **desviación estándar poblacional** con `correction=0`:

```python
std = x.std(dim=2, correction=0)  # Denominador = N (no N-1)
```

Esto evita NaN cuando hay un solo frame temporal.

### Archivos modificados

- `modelo_xvector.py`: Línea 43 - `StatsPooling.forward()`
- `modelo_ecapa.py`: Línea 138 - `AttentiveStatisticsPooling.forward()`
- `modelo_feedforward.py`: Líneas 33, 39 - `VGGishAggregator.forward()`

## Impacto en otras duraciones

| Duración | Frames VGGish | Effecto del bug |
|----------|---------------|-----------------|
| 1s | 1 frame | **NaN** (parche necesario) |
| 2s | 3 frames | Funciona (N>1) |
| 5s | 9 frames | Funciona |
| 10s+ | 19+ frames | Funciona |

Los modelos entrenados con duraciones ≥2 segundos **no requieren reentrenamiento** ya que el cálculo de std() nunca produce NaN con más de 1 frame temporal.

Solo fue necesario reentrenar los modelos de **1 segundo** porque los pesos anteriores fueron aprendidos con comportamiento indefinido (NaN en el forward pass).
