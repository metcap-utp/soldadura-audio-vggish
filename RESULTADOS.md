# Resultados del Modelo de Clasificación SMAW

## I. Validación Cruzada (5-Fold Cross-Validation)

### Audio de 5 segundos

**Fecha de ejecución:** 2026-01-16 17:07:23

| Parámetro     | Accuracy | F1-Score | Precision | Recall |
| ------------- | -------- | -------- | --------- | ------ |
| **Placa**     | 88.40%   | 88.41%   | 88.45%    | 88.40% |
| **Electrodo** | 90.85%   | 90.87%   | 91.08%    | 90.85% |
| **Corriente** | 98.31%   | 98.31%   | 98.33%    | 98.31% |

**Mejora vs modelo individual:** +14.12% (Placa), +13.09% (Electrodo), +3.46% (Corriente)

---

### Audio de 10 segundos

**Fecha de ejecución:** 2026-01-16 17:22:45

| Parámetro     | Accuracy | F1-Score | Precision | Recall |
| ------------- | -------- | -------- | --------- | ------ |
| **Placa**     | 94.41%   | 94.40%   | 94.39%    | 94.41% |
| **Electrodo** | 94.66%   | 94.68%   | 94.80%    | 94.66% |
| **Corriente** | 99.08%   | 99.08%   | 99.10%    | 99.08% |

**Mejora vs modelo individual:** +16.12% (Placa), +10.83% (Electrodo), +3.35% (Corriente)

---

### Audio de 30 segundos

**Fecha de ejecución:** 2026-01-16 17:38:32

| Parámetro     | Accuracy | F1-Score | Precision | Recall |
| ------------- | -------- | -------- | --------- | ------ |
| **Placa**     | 99.74%   | 99.74%   | 99.75%    | 99.74% |
| **Electrodo** | 98.59%   | 98.59%   | 98.65%    | 98.59% |
| **Corriente** | 99.74%   | 99.74%   | 99.75%    | 99.74% |

**Mejora vs modelo individual:** +16.54% (Placa), +10.13% (Electrodo), +1.79% (Corriente)

---

## II. Evaluación en Conjunto Holdout (Datos No Vistos)

Los datos de holdout representan condiciones de uso real, nunca vistos durante el entrenamiento.

### Audio de 5 segundos

**Fecha de ejecución:** 2026-01-16 18:10:29  
**Tamaño del conjunto:** 430 muestras

| Parámetro     | Accuracy | F1-Score | Precision | Recall |
| ------------- | -------- | -------- | --------- | ------ |
| **Placa**     | 72.79%   | 73.00%   | 73.00%    | 73.00% |
| **Electrodo** | 82.33%   | 82.00%   | 83.00%    | 82.00% |
| **Corriente** | 96.74%   | 97.00%   | 97.00%    | 97.00% |

---

### Audio de 10 segundos

**Fecha de ejecución:** 2026-01-16 18:11:08  
**Tamaño del conjunto:** 224 muestras

| Parámetro     | Accuracy | F1-Score | Precision | Recall |
| ------------- | -------- | -------- | --------- | ------ |
| **Placa**     | 74.55%   | 75.00%   | 75.00%    | 75.00% |
| **Electrodo** | 87.05%   | 87.00%   | 88.00%    | 87.00% |
| **Corriente** | 98.21%   | 98.00%   | 98.00%    | 98.00% |

---

### Audio de 30 segundos

**Fecha de ejecución:** 2026-01-16 18:11:45  
**Tamaño del conjunto:** 87 muestras

| Parámetro     | Accuracy | F1-Score | Precision | Recall |
| ------------- | -------- | -------- | --------- | ------ |
| **Placa**     | 80.46%   | 81.00%   | 81.00%    | 80.00% |
| **Electrodo** | 89.66%   | 90.00%   | 92.00%    | 90.00% |
| **Corriente** | 97.70%   | 98.00%   | 98.00%    | 98.00% |

---

## III. Comparación General

| Longitud               | Métricas del Modelo | Holdout Test | Diferencia |
| ---------------------- | ------------------- | ------------ | ---------- |
| **5 seg - Placa**      | 88.40%              | 72.79%       | -15.61%    |
| **5 seg - Electrodo**  | 90.85%              | 82.33%       | -8.52%     |
| **5 seg - Corriente**  | 98.31%              | 96.74%       | -1.57%     |
| **10 seg - Placa**     | 94.41%              | 74.55%       | -19.86%    |
| **10 seg - Electrodo** | 94.66%              | 87.05%       | -7.61%     |
| **10 seg - Corriente** | 99.08%              | 98.21%       | -0.87%     |
| **30 seg - Placa**     | 99.74%              | 80.46%       | -19.28%    |
| **30 seg - Electrodo** | 98.59%              | 89.66%       | -8.93%     |
| **30 seg - Corriente** | 99.74%              | 97.70%       | -2.04%     |

---

## IV. Metodología

### Ensemble de Modelos

El sistema utiliza un **ensemble de 5 modelos** entrenados mediante validación cruzada K-Fold. Cada modelo se entrena con una partición diferente de los datos, lo que permite:

- Aprovechar toda la información disponible para entrenamiento
- Reducir la varianza y mejorar la robustez
- Obtener predicciones más confiables mediante votación

### Soft Voting

Las predicciones finales se obtienen mediante **soft voting**, que:

1. Cada modelo genera probabilidades para cada clase (no solo la predicción final)
2. Se promedian las probabilidades de todos los modelos
3. Se selecciona la clase con mayor probabilidad promedio

**Ventaja:** Aprovecha la confianza de cada modelo en sus predicciones, no solo su elección, resultando en decisiones más informadas y precisas que el hard voting (voto por mayoría simple).

---

## V. Fórmulas de Evaluación

Las métricas se calculan utilizando **scikit-learn**, biblioteca estándar validada en investigación científica.

### Métricas por Clase

Para cada clase individual:

$$\text{Precision} = \frac{TP}{TP + FP}$$

$$\text{Recall} = \frac{TP}{TP + FN}$$

$$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Donde:

- **TP** (True Positives): Predicciones correctas de la clase
- **FP** (False Positives): Predicciones incorrectas como esa clase
- **FN** (False Negatives): Casos de la clase no detectados
- **TN** (True Negatives): Correctamente no clasificados como esa clase

---

_Configuración: 5-fold cross-validation, soft voting, 100 epochs, batch size 32_
