# Métricas de Clasificación SMAW - 50 seg

**Fecha de evaluación:** 2026-02-21T21:49:37

**Configuración:**
- Duración de segmento: 50.0s
- Número de muestras (blind): 59
- Número de modelos (ensemble): 10
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 0.7119 | 0.7338 |
| Electrode Type | 0.8983 | 0.8933 |
| Current Type | 0.9661 | 0.9612 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 0.7119
- **Macro F1-Score:** 0.7338

### Confusion Matrix (Cantidad)

| Pred \ Real | 12 mm | 3 mm | 6 mm |
|---|---|---|---|
| **12 mm** | 13 | 0 | 3 |
| **3 mm** | 0 | 14 | 1 |
| **6 mm** | 0 | 13 | 15 |

### Confusion Matrix (Fracción)

| Pred \ Real | 12 mm | 3 mm | 6 mm |
|---|---|---|---|
| **12 mm** | 81.2% | 0.0% | 18.8% |
| **3 mm** | 0.0% | 93.3% | 6.7% |
| **6 mm** | 0.0% | 46.4% | 53.6% |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 1.0000 | 0.8125 | 0.8966 | 16 |
| Placa_3mm | 0.5185 | 0.9333 | 0.6667 | 15 |
| Placa_6mm | 0.7895 | 0.5357 | 0.6383 | 28 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 0.8983
- **Macro F1-Score:** 0.8933

### Confusion Matrix (Cantidad)

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 6 | 0 | 0 | 1 |
| **E6011** | 1 | 16 | 0 | 0 |
| **E6013** | 0 | 0 | 14 | 0 |
| **E7018** | 0 | 1 | 3 | 17 |

### Confusion Matrix (Fracción)

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 85.7% | 0.0% | 0.0% | 14.3% |
| **E6011** | 5.9% | 94.1% | 0.0% | 0.0% |
| **E6013** | 0.0% | 0.0% | 100.0% | 0.0% |
| **E7018** | 0.0% | 4.8% | 14.3% | 81.0% |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.8571 | 0.8571 | 0.8571 | 7 |
| E6011 | 0.9412 | 0.9412 | 0.9412 | 17 |
| E6013 | 0.8235 | 1.0000 | 0.9032 | 14 |
| E7018 | 0.9444 | 0.8095 | 0.8718 | 21 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 0.9661
- **Macro F1-Score:** 0.9612

### Confusion Matrix (Cantidad)

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 18 | 0 |
| **DC** | 2 | 39 |

### Confusion Matrix (Fracción)

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 100.0% | 0.0% |
| **DC** | 4.9% | 95.1% |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.9000 | 1.0000 | 0.9474 | 18 |
| DC | 1.0000 | 0.9512 | 0.9750 | 41 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **blind** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
