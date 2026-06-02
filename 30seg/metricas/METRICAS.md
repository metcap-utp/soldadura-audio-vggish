# Métricas de Clasificación SMAW - 30 seg

**Fecha de evaluación:** 2026-02-21T21:48:53

**Configuración:**
- Duración de segmento: 30.0s
- Número de muestras (test): 113
- Número de modelos (ensemble): 10
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 0.7080 | 0.7145 |
| Electrode Type | 0.8584 | 0.8435 |
| Current Type | 0.9558 | 0.9524 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 0.7080
- **Macro F1-Score:** 0.7145

### Confusion Matrix (Cantidad)

| Pred \ Real | 12 mm | 3 mm | 6 mm |
|---|---|---|---|
| **12 mm** | 24 | 1 | 5 |
| **3 mm** | 1 | 26 | 2 |
| **6 mm** | 8 | 16 | 30 |

### Confusion Matrix (Fracción)

| Pred \ Real | 12 mm | 3 mm | 6 mm |
|---|---|---|---|
| **12 mm** | 80.0% | 3.3% | 16.7% |
| **3 mm** | 3.4% | 89.7% | 6.9% |
| **6 mm** | 14.8% | 29.6% | 55.6% |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.7273 | 0.8000 | 0.7619 | 30 |
| Placa_3mm | 0.6047 | 0.8966 | 0.7222 | 29 |
| Placa_6mm | 0.8108 | 0.5556 | 0.6593 | 54 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 0.8584
- **Macro F1-Score:** 0.8435

### Confusion Matrix (Cantidad)

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 13 | 0 | 0 | 1 |
| **E6011** | 1 | 35 | 1 | 0 |
| **E6013** | 1 | 1 | 21 | 1 |
| **E7018** | 5 | 1 | 4 | 28 |

### Confusion Matrix (Fracción)

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 92.9% | 0.0% | 0.0% | 7.1% |
| **E6011** | 2.7% | 94.6% | 2.7% | 0.0% |
| **E6013** | 4.2% | 4.2% | 87.5% | 4.2% |
| **E7018** | 13.2% | 2.6% | 10.5% | 73.7% |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.6500 | 0.9286 | 0.7647 | 14 |
| E6011 | 0.9459 | 0.9459 | 0.9459 | 37 |
| E6013 | 0.8077 | 0.8750 | 0.8400 | 24 |
| E7018 | 0.9333 | 0.7368 | 0.8235 | 38 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 0.9558
- **Macro F1-Score:** 0.9524

### Confusion Matrix (Cantidad)

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 39 | 1 |
| **DC** | 4 | 69 |

### Confusion Matrix (Fracción)

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 97.5% | 2.5% |
| **DC** | 5.5% | 94.5% |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.9070 | 0.9750 | 0.9398 | 40 |
| DC | 0.9857 | 0.9452 | 0.9650 | 73 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **test** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
