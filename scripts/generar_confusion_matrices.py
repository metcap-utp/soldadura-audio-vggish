#!/usr/bin/env python3
"""
Genera gráficas de matrices de confusión a partir de los resultados de inferencia.

Uso:
    python scripts/generar_confusion_matrices.py              # Todas las duraciones
    python scripts/generar_confusion_matrices.py --duracion 10seg  # Solo 10seg
    python scripts/generar_confusion_matrices.py --ultimo     # Solo el último resultado de cada duración
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plot_styles import COLORS, DPI, TASK_LABELS, TASKS

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent

# Duraciones disponibles
DURACIONES = ["01seg", "02seg", "05seg", "10seg", "20seg", "30seg", "50seg"]

# Nombres legibles para las tareas
TASK_NAMES = {
    "plate_thickness": "Grosor de Placa",
    "electrode": "Tipo de Electrodo",
    "current_type": "Tipo de Corriente",
}

TASK_COLORS = {
    "plate_thickness": "Blues",
    "electrode": "Greens",
    "current_type": "Oranges",
}

TASK_CMAP = {
    "plate_thickness": COLORS["plate"],
    "electrode": COLORS["electrode"],
    "current_type": COLORS["current"],
}


def cargar_resultados(duracion: str) -> list:
    """Carga los resultados de inferencia de una duración específica."""
    infer_json = ROOT_DIR / duracion / "inferencia.json"

    if not infer_json.exists():
        print(f"  No se encontró {infer_json}")
        return []

    with open(infer_json, "r") as f:
        data = json.load(f)

    # Filtrar solo evaluaciones blind con matrices de confusión
    resultados = [
        r
        for r in data
        if r.get("mode") == "blind_evaluation" and "confusion_matrices" in r
    ]

    return resultados


def generar_grafica_confusion(
    cm: np.ndarray,
    clases: list,
    titulo: str,
    output_path: Path,
    cmap: str = "Blues",
    accuracy: float = None,
    f1_macro: float = None,
    normalize: bool = False,
    color_hex: str = None,
):
    """Genera y guarda una gráfica de matriz de confusión."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    display_cm = cm_normalized if normalize else cm

    annot_format = ".2%" if normalize else "d"
    cbar_label = "Fracción" if normalize else "Cantidad"

    if color_hex:
        colors = [color_hex, "white"]
        custom_cmap = LinearSegmentedColormap.from_list("custom", colors)
        cmap_to_use = custom_cmap
    else:
        cmap_to_use = cmap

    sns.heatmap(
        display_cm,
        annot=True,
        fmt=annot_format,
        cmap=cmap_to_use,
        xticklabels=clases,
        yticklabels=clases,
        ax=ax,
        cbar_kws={"label": cbar_label},
        linewidths=0.5,
        linecolor="white",
    )

    ax.set_xlabel("Predicción", fontsize=12)
    ax.set_ylabel("Real", fontsize=12)
    ax.set_title(titulo, fontsize=14, fontweight="bold")

    if accuracy is not None or f1_macro is not None:
        metrics_text = []
        if accuracy is not None:
            metrics_text.append(f"Accuracy: {accuracy:.4f}")
        if f1_macro is not None:
            metrics_text.append(f"F1 Macro: {f1_macro:.4f}")

        ax.text(
            0.5,
            -0.12,
            " | ".join(metrics_text),
            transform=ax.transAxes,
            ha="center",
            fontsize=10,
            style="italic",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def generar_grafica_combinada(
    resultado: dict,
    duracion: str,
    output_path: Path,
    normalize: bool = False,
):
    """Genera una gráfica combinada con las 3 matrices de confusión."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.set_style("whitegrid")

    tasks = ["plate_thickness", "electrode", "current_type"]

    for ax, task in zip(axes, tasks):
        cm = np.array(resultado["confusion_matrices"][task])
        clases = resultado["classes"][task]

        clases_cortas = [c.replace("Placa_", "").replace("mm", " mm") for c in clases]

        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        display_cm = cm_normalized if normalize else cm

        annot_format = ".2%" if normalize else "d"

        color_hex = TASK_CMAP[task]
        colors = [color_hex, "white"]
        custom_cmap = LinearSegmentedColormap.from_list("custom", colors)

        sns.heatmap(
            display_cm,
            annot=True,
            fmt=annot_format,
            cmap=custom_cmap,
            xticklabels=clases_cortas,
            yticklabels=clases_cortas,
            ax=ax,
            cbar=False,
            linewidths=0.5,
            linecolor="white",
        )

        ax.set_xlabel("Predicción", fontsize=10)
        ax.set_ylabel("Real", fontsize=10)

        acc = resultado["accuracy"].get(task, None)
        f1 = resultado.get("macro_f1", {}).get(task, None)

        if acc is not None and f1 is not None:
            titulo = f"{TASK_NAMES[task]}\nAcc: {acc:.3f} | F1: {f1:.3f}"
        elif acc is not None:
            titulo = f"{TASK_NAMES[task]}\nAcc: {acc:.3f}"
        else:
            titulo = TASK_NAMES[task]
        ax.set_title(titulo, fontsize=11, fontweight="bold")

    config = resultado.get("config", {})
    segment_dur = resultado.get("segment_duration", duracion)
    n_samples = resultado.get("n_samples", "?")
    k_folds = config.get("k_folds", resultado.get("n_models", "?"))
    test_seconds = config.get("test_seconds", segment_dur)
    overlap_seconds = config.get("overlap_seconds", None)

    overlap_text = ""
    if overlap_seconds is not None and test_seconds not in (None, "?"):
        try:
            overlap_ratio = float(overlap_seconds) / float(test_seconds)
            overlap_text = f" | Solapamiento: {overlap_seconds}s ({overlap_ratio * 100:.0f}%)"
        except (ValueError, ZeroDivisionError):
            overlap_text = f" | Solapamiento: {overlap_seconds}s"

    global_metrics = resultado.get("global_metrics", {})
    exact_match = global_metrics.get("exact_match_accuracy", None)
    hamming = global_metrics.get("hamming_accuracy", None)

    title_parts = [
        f"Matrices de Confusión - Audio de {test_seconds} segundos",
        f"K={k_folds} ({n_samples} muestras)",
    ]
    if exact_match is not None and hamming is not None:
        title_parts.append(f"Exact Match: {exact_match:.3f} | Hamming: {hamming:.3f}")

    suffix = " (Fracción)" if normalize else " (Cantidad)"
    fig.suptitle("\n".join(title_parts) + suffix, fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def timestamp_to_filename(timestamp: str) -> str:
    """Convierte un timestamp ISO a un nombre de archivo legible."""
    # "2026-01-21T22:17:34.283909" -> "2026-01-21_22-17-34"
    try:
        dt = datetime.fromisoformat(timestamp)
        return dt.strftime("%Y-%m-%d_%H-%M-%S")
    except (ValueError, TypeError):
        return "unknown"


def procesar_duracion(duracion: str, solo_ultimo: bool = False):
    """Procesa una duración y genera sus gráficas."""
    print(f"\nProcesando {duracion}...")

    resultados = cargar_resultados(duracion)

    if not resultados:
        print(f"  No hay resultados de blind para {duracion}")
        return

    output_dir = ROOT_DIR / duracion / "matrices_confusion"
    output_dir.mkdir(exist_ok=True)

    if solo_ultimo:
        resultados = [resultados[-1]]

    for i, resultado in enumerate(resultados):
        timestamp = resultado.get("timestamp", f"result_{i}")
        k_folds = resultado.get("config", {}).get(
            "k_folds", resultado.get("n_models", 5)
        )
        filename_base = f"k{k_folds}_{timestamp_to_filename(timestamp)}"

        print(f"  Generando graficas para {filename_base}...")

        output_combined = output_dir / f"combined_{filename_base}.png"
        generar_grafica_combinada(resultado, duracion, output_combined, normalize=False)
        print(f"    - {output_combined.name}")

        output_combined_frac = output_dir / f"combined_{filename_base}_frac.png"
        generar_grafica_combinada(resultado, duracion, output_combined_frac, normalize=True)
        print(f"    - {output_combined_frac.name}")

        for task in ["plate_thickness", "electrode", "current_type"]:
            cm = np.array(resultado["confusion_matrices"][task])
            clases = resultado["classes"][task]
            clases_cortas = [
                c.replace("Placa_", "").replace("mm", " mm") for c in clases
            ]

            acc = resultado["accuracy"].get(task, None)
            f1 = resultado.get("macro_f1", {}).get(task, None)

            config = resultado.get("config", {})
            segment_dur = resultado.get("segment_duration", duracion)
            k_folds = config.get("k_folds", resultado.get("n_models", "?"))
            test_seconds = config.get("test_seconds", segment_dur)

            titulo = f"{TASK_NAMES[task]} (Audio: {test_seconds}s, K={k_folds})"

            output_path = output_dir / f"{task}_{filename_base}.png"
            generar_grafica_confusion(
                cm,
                clases_cortas,
                titulo,
                output_path,
                color_hex=TASK_CMAP[task],
                accuracy=acc,
                f1_macro=f1,
                normalize=False,
            )

            output_path_frac = output_dir / f"{task}_{filename_base}_frac.png"
            generar_grafica_confusion(
                cm,
                clases_cortas,
                titulo,
                output_path_frac,
                color_hex=TASK_CMAP[task],
                accuracy=acc,
                f1_macro=f1,
                normalize=True,
            )

        print(f"    - 6 graficas individuales (3 cantidad + 3 fracción)")

    print(f"  Guardadas en: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Genera gráficas de matrices de confusión"
    )
    parser.add_argument(
        "--duracion",
        "-d",
        choices=DURACIONES,
        help="Procesar solo una duración específica",
    )
    parser.add_argument(
        "--ultimo",
        "-u",
        action="store_true",
        help="Procesar solo el último resultado de cada duración",
    )
    parser.add_argument(
        "--todas",
        "-a",
        action="store_true",
        help="Procesar todas las duraciones (por defecto)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  GENERADOR DE MATRICES DE CONFUSIÓN")
    print("=" * 60)

    # Determinar qué duraciones procesar
    if args.duracion:
        duraciones = [args.duracion]
    else:
        duraciones = DURACIONES

    for duracion in duraciones:
        procesar_duracion(duracion, solo_ultimo=args.ultimo)

    print("\n" + "=" * 60)
    print("  Proceso completado")
    print("=" * 60)


if __name__ == "__main__":
    main()
