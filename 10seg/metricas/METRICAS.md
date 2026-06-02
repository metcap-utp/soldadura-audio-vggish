# Métricas de Clasificación SMAW - 10 seg

**Fecha de evaluación:** 2026-02-21T21:47:08

**Configuración:**
- Duración de segmento: 10.0s
- Número de muestras (test): 447
- Número de modelos (ensemble): 10
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 0.7584 | 0.7635 |
| Electrode Type | 0.8345 | 0.8203 |
| Current Type | 0.9709 | 0.9686 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 0.7584
- **Macro F1-Score:** 0.7635

### Confusion Matrix (Cantidad)

| Pred \ Real | 12 mm | 3 mm | 6 mm |
|---|---|---|---|
| **12 mm** | 100 | 6 | 23 |
| **3 mm** | 5 | 108 | 1 |
| **6 mm** | 45 | 28 | 131 |

### Confusion Matrix (Fracción)

| Pred \ Real | 12 mm | 3 mm | 6 mm |
|---|---|---|---|
| **12 mm** | 77.5% | 4.7% | 17.8% |
| **3 mm** | 4.4% | 94.7% | 0.9% |
| **6 mm** | 22.1% | 13.7% | 64.2% |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.6667 | 0.7752 | 0.7168 | 129 |
| Placa_3mm | 0.7606 | 0.9474 | 0.8438 | 114 |
| Placa_6mm | 0.8452 | 0.6422 | 0.7298 | 204 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 0.8345
- **Macro F1-Score:** 0.8203

### Confusion Matrix (Cantidad)

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 55 | 0 | 0 | 0 |
| **E6011** | 6 | 133 | 5 | 2 |
| **E6013** | 0 | 3 | 105 | 3 |
| **E7018** | 33 | 14 | 8 | 80 |

### Confusion Matrix (Fracción)

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 100.0% | 0.0% | 0.0% | 0.0% |
| **E6011** | 4.1% | 91.1% | 3.4% | 1.4% |
| **E6013** | 0.0% | 2.7% | 94.6% | 2.7% |
| **E7018** | 24.4% | 10.4% | 5.9% | 59.3% |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.5851 | 1.0000 | 0.7383 | 55 |
| E6011 | 0.8867 | 0.9110 | 0.8986 | 146 |
| E6013 | 0.8898 | 0.9459 | 0.9170 | 111 |
| E7018 | 0.9412 | 0.5926 | 0.7273 | 135 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 0.9709
- **Macro F1-Score:** 0.9686

### Confusion Matrix (Cantidad)

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 156 | 2 |
| **DC** | 11 | 278 |

### Confusion Matrix (Fracción)

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 98.7% | 1.3% |
| **DC** | 3.8% | 96.2% |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.9341 | 0.9873 | 0.9600 | 158 |
| DC | 0.9929 | 0.9619 | 0.9772 | 289 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **test** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
