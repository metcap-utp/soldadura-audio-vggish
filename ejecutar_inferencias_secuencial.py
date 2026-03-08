#!/usr/bin/env python3
"""
Ejecuta las inferencias de forma secuencial para los modelos que tuvieron
overlaps de timestamps.

Todos están en 05seg con diferentes configuraciones de k y overlap.
El modelo xvector no se repite porque sus tiempos no tenían overlaps significativos.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Lista de inferencias a ejecutar (secuencialmente)
# (duration, overlap, k_folds, model_type)
INFERENCIAS = [
    # VGGish ECAPA
    (5, 0.5, 1, "ecapa"),
    (5, 0.5, 3, "ecapa"),
    (5, 0.5, 5, "ecapa"),
    (5, 0.5, 7, "ecapa"),
    (5, 0.0, 10, "ecapa"),
    (5, 0.25, 10, "ecapa"),
    (5, 0.75, 10, "ecapa"),
    (5, 0.5, 15, "ecapa"),
    
    # VGGish FEEDFORWARD
    (5, 0.5, 3, "feedforward"),
    (5, 0.5, 5, "feedforward"),
    (5, 0.5, 15, "feedforward"),
    (5, 0.5, 20, "feedforward"),
]

def run_inference(duration, overlap, k_folds, model_type):
    """Ejecuta una inferencia con los parámetros especificados."""
    cmd = [
        "python3", "inferir.py",
        "--duration", str(duration),
        "--overlap", str(overlap),
        "--k-folds", str(k_folds),
        "--model", model_type,
        "--evaluar"
    ]
    
    print(f"\n{'='*80}")
    print(f"Ejecutando: {' '.join(cmd)}")
    print(f"{'='*80}")
    print(f"Timestamp: {datetime.now().isoformat()}\n")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode == 0


def main():
    print("\n" + "="*80)
    print("EJECUCIÓN SECUENCIAL DE INFERENCIAS (VGGish 05seg)")
    print("="*80)
    print(f"Total de inferencias a ejecutar: {len(INFERENCIAS)}")
    print(f"Inicio: {datetime.now().isoformat()}\n")
    
    successful = 0
    failed = 0
    failed_list = []
    
    for i, (duration, overlap, k_folds, model_type) in enumerate(INFERENCIAS, 1):
        print(f"\n[{i}/{len(INFERENCIAS)}] {model_type:12s} k={k_folds:2d} overlap={overlap:.2f}")
        
        try:
            if run_inference(duration, overlap, k_folds, model_type):
                successful += 1
                print(f"✓ Completada exitosamente")
            else:
                failed += 1
                failed_list.append((model_type, k_folds, overlap))
                print(f"✗ Falló")
        except Exception as e:
            failed += 1
            failed_list.append((model_type, k_folds, overlap))
            print(f"✗ Error: {e}")
    
    # Resumen final
    print("\n" + "="*80)
    print("RESUMEN FINAL")
    print("="*80)
    print(f"Exitosas: {successful}/{len(INFERENCIAS)}")
    print(f"Fallidas: {failed}/{len(INFERENCIAS)}")
    
    if failed_list:
        print("\nInferencias fallidas:")
        for model_type, k_folds, overlap in failed_list:
            print(f"  - {model_type} k={k_folds} overlap={overlap:.2f}")
    
    print(f"\nFinal: {datetime.now().isoformat()}\n")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
