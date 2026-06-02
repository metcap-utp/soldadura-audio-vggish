# Métricas de Clasificación SMAW - 02 seg

**Fecha de evaluación:** 2026-02-21T21:44:45

**Configuración:**
- Duración de segmento: 02.0s
- Número de muestras (test): 2465
- Número de modelos (ensemble): 10
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 0.6840 | 0.6905 |
| Electrode Type | 0.7639 | 0.7499 |
| Current Type | 0.8738 | 0.8687 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 0.6840
- **Macro F1-Score:** 0.6905

### Confusion Matrix (Cantidad)

| Pred \ Real | 12 mm | 3 mm | 6 mm |
|---|---|---|---|
| **12 mm** | 527 | 38 | 150 |
| **3 mm** | 45 | 544 | 58 |
| **6 mm** | 308 | 180 | 615 |

### Confusion Matrix (Fracción)

| Pred \ Real | 12 mm | 3 mm | 6 mm |
|---|---|---|---|
| **12 mm** | 73.7% | 5.3% | 21.0% |
| **3 mm** | 7.0% | 84.1% | 9.0% |
| **6 mm** | 27.9% | 16.3% | 55.8% |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.5989 | 0.7371 | 0.6608 | 715 |
| Placa_3mm | 0.7139 | 0.8408 | 0.7722 | 647 |
| Placa_6mm | 0.7473 | 0.5576 | 0.6386 | 1103 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 0.7639
- **Macro F1-Score:** 0.7499

### Confusion Matrix (Cantidad)

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 257 | 25 | 21 | 4 |
| **E6011** | 51 | 699 | 38 | 14 |
| **E6013** | 25 | 48 | 515 | 45 |
| **E7018** | 100 | 107 | 104 | 412 |

### Confusion Matrix (Fracción)

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 83.7% | 8.1% | 6.8% | 1.3% |
| **E6011** | 6.4% | 87.2% | 4.7% | 1.7% |
| **E6013** | 3.9% | 7.6% | 81.4% | 7.1% |
| **E7018** | 13.8% | 14.8% | 14.4% | 57.0% |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.5935 | 0.8371 | 0.6946 | 307 |
| E6011 | 0.7952 | 0.8716 | 0.8316 | 802 |
| E6013 | 0.7596 | 0.8136 | 0.7857 | 633 |
| E7018 | 0.8674 | 0.5698 | 0.6878 | 723 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 0.8738
- **Macro F1-Score:** 0.8687

### Confusion Matrix (Cantidad)

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 834 | 21 |
| **DC** | 290 | 1320 |

### Confusion Matrix (Fracción)

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 97.5% | 2.5% |
| **DC** | 18.0% | 82.0% |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.7420 | 0.9754 | 0.8428 | 855 |
| DC | 0.9843 | 0.8199 | 0.8946 | 1610 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **test** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
