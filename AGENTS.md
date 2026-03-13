# AGENTS.md - Guía para Agentes de Código

## Información del Proyecto

**Proyecto**: Clasificación de Audio SMAW (Soldadura por Arco Eléctrico)  
**Lenguaje**: Python 3.10+  
**Framework**: PyTorch + Librosa + Scikit-learn + TensorFlow Hub (VGGish)  
**Idioma**: Español (código y documentación)

## Restricción Crítica: Ejecución Secuencial

**IMPORTANTE**: Las tareas de entrenamiento e inferencia consumen recursos significativos de GPU y CPU.
Solo se puede ejecutar **una tarea a la vez** - deben ser **secuenciales**, no paralelas.

Esto incluye:

- `entrenar_xvector.py`, `entrenar_ecapa.py`, `entrenar_feedforward.py`
- `inferir.py --evaluar`
- `generar_splits.py`

No lanzar múltiples procesos de entrenamiento/inferencia simultáneamente.

## Comandos de Desarrollo

### Generación de Datos

```bash
python generar_splits.py --duration 5 --overlap 0.5
python generar_splits.py --duration 10 --overlap 0.0
```

### Entrenamiento

```bash
# X-Vector
python entrenar_xvector.py --duration 5 --overlap 0.5 --k-folds 10

# ECAPA-TDNN
python entrenar_ecapa.py --duration 5 --overlap 0.5 --k-folds 10

# FeedForward
python entrenar_feedforward.py --duration 5 --overlap 0.5 --k-folds 10
```

### Inferencia/Evaluación

```bash
# Evaluar ensemble en conjunto blind
python inferir.py --duration 5 --overlap 0.5 --k-folds 10 --model xvector --evaluar

# Predicción de un archivo específico
python inferir.py --duration 5 --overlap 0.5 --audio ruta/archivo.wav

# Predicciones aleatorias
python inferir.py --duration 5 --overlap 0.5 --n 10
```

### Scripts Batch

```bash
# Entrenar y evaluar todos los modelos
./entrenar_todos.sh                              # Todo (k=10, overlap=0.5)
./entrenar_todos.sh --duration 5 --model xvector # Solo xvector, 5seg
./entrenar_todos.sh --dry-run                    # Solo mostrar qué se haría
./entrenar_todos.sh --skip-train                 # Solo evaluación
./entrenar_todos.sh --skip-eval                  # Solo entrenamiento
```

### Visualización

```bash
python scripts/graficar_folds.py 05seg
python scripts/graficar_duraciones.py
python scripts/graficar_overlap.py --save
```

### Linting y Formato

```bash
# Instalar herramientas
pip install black ruff mypy

# Formatear código (línea: 100 caracteres)
black --line-length 100 *.py scripts/ utils/

# Linting
ruff check *.py scripts/ utils/

# Type checking
mypy modelo_ecapa.py modelo_xvector.py utils/
```

### Testing

```bash
# No hay suite de tests configurada. Usar pytest si se añaden:
pip install pytest pytest-cov
pytest -v
pytest test_modelo.py::test_funcion -v
pytest --cov=. --cov-report=html
```

## Archivos de Log

Todos los scripts de entrenamiento e inferencia generan archivos de log automáticamente en la carpeta `logs/`.

### Localización de Logs

- **Entrenamiento**: `logs/entrenar_[arquitectura]_[duracion]seg_[timestamp].log`
  - Ejemplo: `logs/entrenar_ecapa_05seg_20250228_143000.log`
- **Inferencia**: `logs/inferir_[duracion]seg_[modelo]_[timestamp].log`
  - Ejemplo: `logs/inferir_05seg_xvector_20250228_150000.log`

### Formato de Timestamp

Los archivos de log usan el formato `YYYYMMDD_HHMMSS`:

- `YYYY`: Año (2025)
- `MM`: Mes (01-12)
- `DD`: Día (01-31)
- `HH`: Hora (00-23)
- `MM`: Minuto (00-59)
- `SS`: Segundo (00-59)

### Contenido de Logs

Los archivos de log contienen:

- Todos los prints de la ejecución del script
- Métricas de entrenamiento (loss, accuracy por fold)
- Tiempos de ejecución (total, por fold, extracción de VGGish)
- Información de enfoque utilizado (vggish, yamnet, spectral-mfcc)
- Cualquier error o warning durante la ejecución

### Gestión de Logs

Los archivos .log están excluidos de control de versión (`.gitignore`). Para limpiar logs antiguos:

```bash
# Eliminar todos los logs
rm logs/*.log

# Eliminar logs más viejos que 30 días
find logs/ -name "*.log" -mtime +30 -delete
```

### Ejemplo de Lectura de Logs

```bash
# Ver último log de entrenamiento (últimas 50 líneas)
tail -n 50 logs/entrenar_ecapa_05seg_*.log

# Buscar errores en los logs
grep -i "error\|exception" logs/*.log

# Ver el log de una ejecución específica
cat logs/entrenar_xvector_10seg_20250228_143000.log
```

## Guías de Estilo de Código

### Formato General

- **Longitud de línea**: 100 caracteres máximo
- **Indentación**: 4 espacios (PEP 8)
- **Comillas**: Dobles `""` para strings, triple comilla doble para docstrings
- **Encoding**: UTF-8
- **Fin de línea**: Unix (LF)

### Imports (Orden estricto)

```python
# 1. Librerías estándar (alfabético)
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# 2. Librerías de terceros (alfabético)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 3. Imports internos (usar sys.path.insert)
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))
from modelo_xvector import SMAWXVectorModel
from utils.audio_utils import PROJECT_ROOT
```

### Convenciones de Nombres

| Elemento           | Convención       | Ejemplo                          |
| ------------------ | ---------------- | -------------------------------- |
| Clases             | PascalCase       | `ECAPAMultiTask`                 |
| Funciones          | snake_case       | `extract_vggish_embeddings()`    |
| Variables          | snake_case       | `segment_duration`, `num_epochs` |
| Constantes         | UPPER_SNAKE_CASE | `RANDOM_SEED = 42`               |
| Atributos privados | \_snake_case     | `_internal_state`                |

### Type Hints (Obligatorios en funciones públicas)

```python
def load_audio_segment(
    audio_path: Path,
    segment_duration: float,
    segment_index: int,
    sr: int = 16000,
) -> Optional[np.ndarray]:
    """Carga un segmento específico de audio."""
    ...
```

### Docstrings (Google Style)

```python
def extract_labels_from_session_path(session_path: Path) -> Optional[Dict]:
    """
    Extrae etiquetas del path de una carpeta de sesión.

    Args:
        session_path: Path de la sesión de audio.

    Returns:
        dict con Plate Thickness, Electrode, Type of Current, Session.
        None si el path no tiene la estructura esperada.
    """
```

### Manejo de Errores

```python
# Try-except con mensaje claro
try:
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
except Exception as e:
    print(f"Error loading {audio_path}: {e}")
    return None

# Validaciones con raise
if not audio_path.exists():
    raise ValueError(f"Audio file not found: {audio_path}")
```

### Estructura de Archivos Python

```python
"""
Descripción corta del módulo.
"""

# === Imports ===
import sys
from pathlib import Path

import torch
import torch.nn as nn

# === Path setup ===
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# === Constantes ===
RANDOM_SEED = 42
BATCH_SIZE = 32

# === Clases ===
class MiClase:
    """Docstring de la clase."""
    pass

# === Funciones ===
def funcion_principal():
    """Docstring de la función."""
    pass

# === Entry Point ===
if __name__ == "__main__":
    main()
```

## Reglas Específicas del Proyecto

### Path Handling (Siempre usar Pathlib)

```python
from pathlib import Path

ROOT_DIR = Path(__file__).parent
audio_path = ROOT_DIR / "audio" / "file.wav"
```

### Imports de Proyecto

Siempre usar `sys.path.insert` para imports internos:

```python
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))
from models.modelo_xvector import SMAWXVectorModel
from utils.audio_utils import load_audio_segment
```

### Estructura de Carpetas

- `models/` - Definiciones de modelos (modelo_xvector.py, modelo_ecapa.py, modelo_feedforward.py)
- `logs/` - Archivos de log de entrenamiento e inferencia
- `utils/` - Utilidades (audio_utils.py, timing.py, logging_utils.py)
- `{N}seg/` - Datos y resultados por duración de segmento
  - `modelos/{arquitectura}/k{K}_overlap_{ratio}/` - Modelos entrenados
  - `resultados.json` - Métricas de entrenamiento (acumulativo)
  - `inferencia.json` - Métricas de evaluación (acumulativo)

### Warning Filters

Suprimir warnings en scripts de producción:

```python
import warnings
warnings.filterwarnings("ignore")
```

### Seed para Reproducibilidad

```python
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
```

## Hiperparámetros de Entrenamiento

| Parámetro       | Valor     |
| --------------- | --------- |
| Batch Size      | 32        |
| Epochs          | 100       |
| Learning Rate   | 1e-3      |
| Early Stopping  | 15 epochs |
| Optimizer       | AdamW     |
| Label Smoothing | 0.1       |
| Weight Decay    | 1e-4      |

## Estructura de Salida

- `{N}seg/modelos/{arquitectura}/k{K}_overlap_{ratio}/` - Modelos `.pth`
- `{N}seg/resultados.json` - Métricas de entrenamiento (acumulativo)
- `{N}seg/inferencia.json` - Métricas de evaluación (acumulativo)
- `{N}seg/metricas/METRICAS.md` - Documento Markdown con matrices de confusión

## Notas Importantes

1. **No versionar**: Archivos `.pth`, `.keras`, `.pkl`, `.wav`, `.mp3`
2. **VGGish**: Modelo pre-entrenado de Google (~275MB) se descarga automáticamente desde TensorFlow Hub
3. **Argumentos CLI**: `--duration` (1,2,5,10,20,30,50), `--overlap` (0.0-0.75), `--k-folds` (3-20)
4. **Arquitecturas**: xvector, ecapa_tdnn, feedforward
