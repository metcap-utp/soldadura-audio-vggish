# Métricas de Clasificación SMAW - 5seg

**Fecha de evaluación:** 2026-03-03 17:52:03

**Configuración:**
- Duración de segmento: 5.0s
- Número de muestras (test): 951
- Número de modelos (ensemble): 1
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 0.7708 | 0.7766 |
| Electrode Type | 0.8465 | 0.8367 |
| Current Type | 0.9432 | 0.9387 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 0.7708
- **Macro F1-Score:** 0.7766

### Confusion Matrix

| Pred \ Real | Placa_12mm | Placa_3mm | Placa_6mm |
|---|---|---|---|
| **Placa_12mm** | 201 | 9 | 65 |
| **Placa_3mm** | 5 | 233 | 9 |
| **Placa_6mm** | 88 | 42 | 299 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.6837 | 0.7309 | 0.7065 | 275 |
| Placa_3mm | 0.8204 | 0.9433 | 0.8776 | 247 |
| Placa_6mm | 0.8016 | 0.6970 | 0.7456 | 429 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 0.8465
- **Macro F1-Score:** 0.8367

### Confusion Matrix

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 106 | 5 | 5 | 2 |
| **E6011** | 8 | 294 | 6 | 2 |
| **E6013** | 2 | 7 | 220 | 12 |
| **E7018** | 33 | 47 | 17 | 185 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.7114 | 0.8983 | 0.7940 | 118 |
| E6011 | 0.8329 | 0.9484 | 0.8869 | 310 |
| E6013 | 0.8871 | 0.9129 | 0.8998 | 241 |
| E7018 | 0.9204 | 0.6560 | 0.7660 | 282 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 0.9432
- **Macro F1-Score:** 0.9387

### Confusion Matrix

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 320 | 10 |
| **DC** | 44 | 577 |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.8791 | 0.9697 | 0.9222 | 330 |
| DC | 0.9830 | 0.9291 | 0.9553 | 621 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **test** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
