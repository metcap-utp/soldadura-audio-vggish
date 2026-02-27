#!/usr/bin/env python3
"""
Genera heatmap de métricas (accuracy) vs duración para la tesis.

Usa datos de inferencia (blind evaluation) para mostrar el rendimiento real.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plot_styles import DPI, TASKS


SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent


DURATION_DIRS = ["01seg", "02seg", "05seg", "10seg", "20seg", "30seg", "50seg"]


def get_duration_value(dir_name: str) -> int:
    """Extrae el valor numérico de la duración."""
    return int(dir_name.replace("seg", ""))


def load_inference_results(duration_dir: str) -> dict:
    """Carga los mejores resultados de inferencia (blind) para una duración."""
    infer_path = ROOT_DIR / duration_dir / "inferencia.json"

    if not infer_path.exists():
        return {}

    with open(infer_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]

    blind_entries = [e for e in data if e.get("mode") == "blind_evaluation"]

    if not blind_entries:
        return {}

    # Tomar la entrada más reciente
    entry = sorted(blind_entries, key=lambda e: e.get("timestamp", ""), reverse=True)[0]

    acc = entry.get("accuracy", {})

    return {
        "plate": acc.get("plate_thickness", 0),
        "electrode": acc.get("electrode", 0),
        "current": acc.get("current_type", 0),
    }


def create_accuracy_heatmap(save: bool = True):
    """Genera heatmap de accuracy vs duración usando datos de inferencia."""
    durations = []
    data = {task: [] for task in TASKS}

    for duration_dir in sorted(DURATION_DIRS):
        results = load_inference_results(duration_dir)

        if not results:
            print(f"  Sin datos de inferencia para {duration_dir}")
            continue

        durations.append(get_duration_value(duration_dir))

        for task in TASKS:
            value = results.get(task, 0)
            data[task].append(value)

    if not durations:
        print("No se encontraron datos de inferencia")
        return

    durations = sorted(durations)

    task_labels = {
        "plate": "Espesor de Placa",
        "electrode": "Tipo de Electrodo",
        "current": "Tipo de Corriente",
    }

    matrix = np.array([data[task] for task in TASKS])

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.grid(False)

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(durations)))
    ax.set_xticklabels([f"{d}s" for d in durations])

    ax.set_yticks(range(len(TASKS)))
    ax.set_yticklabels([task_labels[t] for t in TASKS])

    ax.set_xlabel("Duración del segmento", fontsize=12)
    ax.set_ylabel("Tarea", fontsize=12)
    ax.set_title("Accuracy por Duración (Evaluación Ciega)", fontsize=14)

    for i in range(len(TASKS)):
        for j in range(len(durations)):
            value = matrix[i, j]
            color = "white" if value < 0.5 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy", fontsize=11)

    plt.tight_layout()

    if save:
        output_path = ROOT_DIR / "img" / "heatmap_accuracy_blind.png"
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
        print(f"Guardado: {output_path}")

    plt.close()


def create_summary_heatmap(save: bool = True):
    """Genera heatmap con accuracy por tarea y promedio."""
    durations = []
    acc_matrix = {task: [] for task in TASKS}
    avg_list = []

    for duration_dir in sorted(DURATION_DIRS):
        results = load_inference_results(duration_dir)

        if not results:
            print(f"  Sin datos de inferencia para {duration_dir}")
            continue

        durations.append(get_duration_value(duration_dir))

        task_accs = []
        for task in TASKS:
            value = results.get(task, 0)
            acc_matrix[task].append(value)
            task_accs.append(value)

        avg_list.append(np.mean(task_accs))

    if not durations:
        print("No se encontraron datos de inferencia")
        return

    durations = sorted(durations)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.grid(False)

    task_labels = {
        "plate": "Placa",
        "electrode": "Electrodo",
        "current": "Corriente",
    }

    matrix = np.array([acc_matrix[task] for task in TASKS])

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(durations)))
    ax.set_xticklabels([f"{d}s" for d in durations])

    ax.set_yticks(range(len(TASKS)))
    ax.set_yticklabels([task_labels[t] for t in TASKS])

    ax.set_xlabel("Duración del segmento", fontsize=12)
    ax.set_ylabel("Tarea", fontsize=12)
    ax.set_title("Accuracy por Duración y Tarea (Evaluación Ciega)", fontsize=14)

    for i in range(len(TASKS)):
        for j in range(len(durations)):
            value = matrix[i, j]
            color = "white" if value < 0.5 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy", fontsize=11)

    plt.tight_layout()

    if save:
        output_path = ROOT_DIR / "img" / "heatmap_accuracy_blind_resumen.png"
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
        print(f"Guardado: {output_path}")

    plt.close()


if __name__ == "__main__":
    print("Generando heatmaps para la tesis (datos de inferencia)...")
    print()

    print("1. Heatmap de Accuracy (blind)...")
    create_accuracy_heatmap(save=True)

    print("2. Heatmap resumen (blind)...")
    create_summary_heatmap(save=True)

    print()
    print("Heatmaps generados en: img/")
