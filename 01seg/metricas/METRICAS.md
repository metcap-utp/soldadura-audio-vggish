# Métricas de Clasificación SMAW - 01 seg

**Fecha de evaluación:** 2026-02-23T11:42:28

**Configuración:**
- Duración de segmento: 01.0s
- Número de muestras (test): 4988
- Número de modelos (ensemble): 10
- Método de votación: soft

---

## Resumen de Métricas

| Tarea | Accuracy | Macro F1 |
|-------|----------|----------|
| Plate Thickness | 0.6468 | 0.6510 |
| Electrode Type | 0.7037 | 0.6895 |
| Current Type | 0.8380 | 0.8326 |

---

## Plate Thickness (Espesor de Placa)

### Métricas
- **Accuracy:** 0.6468
- **Macro F1-Score:** 0.6510

### Confusion Matrix (Cantidad)

| Pred \ Real | 12 mm | 3 mm | 6 mm |
|---|---|---|---|
| **12 mm** | 966 | 112 | 371 |
| **3 mm** | 130 | 1009 | 174 |
| **6 mm** | 570 | 405 | 1251 |

### Confusion Matrix (Fracción)

| Pred \ Real | 12 mm | 3 mm | 6 mm |
|---|---|---|---|
| **12 mm** | 66.7% | 7.7% | 25.6% |
| **3 mm** | 9.9% | 76.8% | 13.3% |
| **6 mm** | 25.6% | 18.2% | 56.2% |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| Placa_12mm | 0.5798 | 0.6667 | 0.6202 | 1449 |
| Placa_3mm | 0.6612 | 0.7685 | 0.7108 | 1313 |
| Placa_6mm | 0.6965 | 0.5620 | 0.6221 | 2226 |

---

## Electrode Type (Tipo de Electrodo)

### Métricas
- **Accuracy:** 0.7037
- **Macro F1-Score:** 0.6895

### Confusion Matrix (Cantidad)

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 465 | 64 | 65 | 29 |
| **E6011** | 155 | 1249 | 164 | 56 |
| **E6013** | 79 | 121 | 965 | 120 |
| **E7018** | 205 | 207 | 213 | 831 |

### Confusion Matrix (Fracción)

| Pred \ Real | E6010 | E6011 | E6013 | E7018 |
|---|---|---|---|---|
| **E6010** | 74.6% | 10.3% | 10.4% | 4.7% |
| **E6011** | 9.5% | 76.9% | 10.1% | 3.4% |
| **E6013** | 6.1% | 9.4% | 75.1% | 9.3% |
| **E7018** | 14.1% | 14.2% | 14.6% | 57.1% |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| E6010 | 0.5144 | 0.7464 | 0.6090 | 623 |
| E6011 | 0.7611 | 0.7691 | 0.7651 | 1624 |
| E6013 | 0.6859 | 0.7510 | 0.7169 | 1285 |
| E7018 | 0.8021 | 0.5707 | 0.6669 | 1456 |

---

## Current Type (Tipo de Corriente)

### Métricas
- **Accuracy:** 0.8380
- **Macro F1-Score:** 0.8326

### Confusion Matrix (Cantidad)

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 1641 | 87 |
| **DC** | 721 | 2539 |

### Confusion Matrix (Fracción)

| Pred \ Real | AC | DC |
|---|---|---|
| **AC** | 95.0% | 5.0% |
| **DC** | 22.1% | 77.9% |

### Classification Report

| Clase | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|--------|
| AC | 0.6948 | 0.9497 | 0.8024 | 1728 |
| DC | 0.9669 | 0.7788 | 0.8627 | 3260 |

---

## Notas

- Las métricas se calcularon sobre el conjunto **test** (datos nunca vistos durante entrenamiento).
- El ensemble usa **Soft Voting**: promedia logits de todos los modelos antes de aplicar argmax.
- Los modelos fueron entrenados con **StratifiedGroupKFold** para evitar data leakage por sesión.
