# Métricas de Clasificación SMAW - 20 seg

**Fecha de evaluación:** 2026-02-21T21:48:00

**Configuración:**
- Duración de segmento: 20.0s
- Número de muestras (blind): 199
- Número de modelos (ensemble): 10
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 0.7688 | 0.7742 |
| Electrode Type | 0.8342 | 0.8227 |
| Current Type | 0.9698 | 0.9674 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 0.7688
- **Macro F1-Score:** 0.7742

### Confusion Matrix (Cantidad)

| Pred \ Real | 12 mm | 3 mm | 6 mm |
|---|---|---|---|
| **12 mm** | 46 | 1 | 9 |
| **3 mm** | 1 | 48 | 0 |
| **6 mm** | 17 | 18 | 59 |

### Confusion Matrix (Fracción)

| Pred \ Real | 12 mm | 3 mm | 6 mm |
|---|---|---|---|
| **12 mm** | 82.1% | 1.8% | 16.1% |
| **3 mm** | 2.0% | 98.0% | 0.0% |
| **6 mm** | 18.1% | 19.1% | 62.8% |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.7188 | 0.8214 | 0.7667 | 56 |
| Placa_3mm | 0.7164 | 0.9796 | 0.8276 | 49 |
| Placa_6mm | 0.8676 | 0.6277 | 0.7284 | 94 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 0.8342
- **Macro F1-Score:** 0.8227

### Confusion Matrix (Cantidad)

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 24 | 1 | 0 | 0 |
| **E6011** | 2 | 61 | 3 | 0 |
| **E6013** | 0 | 1 | 42 | 2 |
| **E7018** | 12 | 6 | 6 | 39 |

### Confusion Matrix (Fracción)

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 96.0% | 4.0% | 0.0% | 0.0% |
| **E6011** | 3.0% | 92.4% | 4.5% | 0.0% |
| **E6013** | 0.0% | 2.2% | 93.3% | 4.4% |
| **E7018** | 19.0% | 9.5% | 9.5% | 61.9% |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.6316 | 0.9600 | 0.7619 | 25 |
| E6011 | 0.8841 | 0.9242 | 0.9037 | 66 |
| E6013 | 0.8235 | 0.9333 | 0.8750 | 45 |
| E7018 | 0.9512 | 0.6190 | 0.7500 | 63 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 0.9698
- **Macro F1-Score:** 0.9674

### Confusion Matrix (Cantidad)

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 69 | 1 |
| **DC** | 5 | 124 |

### Confusion Matrix (Fracción)

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 98.6% | 1.4% |
| **DC** | 3.9% | 96.1% |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.9324 | 0.9857 | 0.9583 | 70 |
| DC | 0.9920 | 0.9612 | 0.9764 | 129 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **blind** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
