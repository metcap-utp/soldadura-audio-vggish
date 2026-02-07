"""
Grafica métricas vs cantidad de folds para una duración específica.

Uso:
    python graficar_folds.py 5seg              # Grafica para 5 segundos
    python graficar_folds.py 10seg --save      # Guarda la imagen
    python graficar_folds.py 5seg --metric f1  # Solo métrica F1
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Directorio raíz del proyecto
ROOT_DIR = Path(__file__).parent.parent


def load_results(duration_dir: str) -> list:
    """Carga los resultados de entrenamiento."""
    results_path = ROOT_DIR / duration_dir / "results.json"

    if not results_path.exists():
        print(f"Error: No se encontró {results_path}")
        print(f"Ejecuta primero: cd {duration_dir} && python entrenar.py --k-folds N")
        sys.exit(1)

    with open(results_path, "r") as f:
        results = json.load(f)

    if not isinstance(results, list):
        results = [results]

    return results


def load_global_metrics(duration_dir: str) -> dict:
    """Carga métricas globales (Exact Match y Hamming) desde infer.json.
    
    Prioriza las entradas más recientes y sin overlap (overlap_ratio = 0.0 o None).
    """
    infer_path = ROOT_DIR / duration_dir / "infer.json"
    if not infer_path.exists():
        return {}

    with open(infer_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list):
        data = [data]

    # Agrupar por k-folds y filtrar por overlap
    entries_by_k = {}
    
    for entry in data:
        # Solo blind_evaluation
        if entry.get("mode") != "blind_evaluation":
            continue
        
        gm = entry.get("global_metrics") or {}
        if not gm:
            continue
        
        k = (
            entry.get("k_folds")
            or entry.get("n_models")
            or entry.get("config", {}).get("k_folds")
        )
        if not k:
            continue
        
        # Solo usar entradas sin overlap o con overlap = 0.0
        config = entry.get("config", {})
        overlap = config.get("overlap_ratio")
        if overlap is not None and overlap != 0.0:
            continue
        
        if k not in entries_by_k:
            entries_by_k[k] = []
        entries_by_k[k].append(entry)
    
    # Para cada k, tomar la entrada más reciente
    global_by_k = {}
    
    for k, entries in entries_by_k.items():
        # Ordenar por timestamp y tomar el más reciente
        entries_sorted = sorted(
            entries, 
            key=lambda e: e.get("timestamp", ""), 
            reverse=True
        )
        entry = entries_sorted[0]
        gm = entry.get("global_metrics", {})
        
        global_by_k[k] = {
            "exact_match": gm.get("exact_match_accuracy"),
            "hamming": gm.get("hamming_accuracy"),
        }

    return global_by_k


def extract_metrics_by_folds(results: list) -> dict:
    """Extrae métricas de validación cruzada (promedio de fold_results) por K.
    
    Prioriza las entradas más recientes y sin overlap (overlap_ratio = 0.0 o None).
    """
    import numpy as np
    
    # Agrupar por k-folds y filtrar por overlap
    entries_by_k = {}
    
    for entry in results:
        config = entry.get("config", {})
        k = config.get("n_folds", 5)
        overlap = config.get("overlap_ratio")
        
        # Solo usar entradas sin overlap o con overlap = 0.0
        if overlap is not None and overlap != 0.0:
            continue
        
        if k not in entries_by_k:
            entries_by_k[k] = []
        entries_by_k[k].append(entry)
    
    # Para cada k, tomar la entrada más reciente
    metrics_by_k = {}
    
    for k, entries in entries_by_k.items():
        # Ordenar por timestamp y tomar el más reciente
        entries_sorted = sorted(
            entries, 
            key=lambda e: e.get("timestamp", ""), 
            reverse=True
        )
        entry = entries_sorted[0]
        
        metrics_by_k[k] = {
            "plate": {},
            "electrode": {},
            "current": {},
        }

        # Usar fold_results (métricas de validación) en lugar de ensemble_results
        fold_results = entry.get("fold_results", [])
        
        if fold_results:
            # Calcular promedio de las métricas de cada fold
            for task in ["plate", "electrode", "current"]:
                acc_key = f"acc_{task}"
                f1_key = f"f1_{task}"
                
                accs = [fr[acc_key] for fr in fold_results if acc_key in fr]
                f1s = [fr[f1_key] for fr in fold_results if f1_key in fr]
                
                if accs:
                    metrics_by_k[k][task]["accuracy"] = np.mean(accs)
                if f1s:
                    metrics_by_k[k][task]["f1"] = np.mean(f1s)
        else:
            # Fallback: usar ensemble_results (compatibilidad con formato antiguo)
            res = entry.get("ensemble_results", entry.get("results", {}))
            for task in ["plate", "electrode", "current"]:
                if task in res:
                    for metric in ["accuracy", "f1", "precision", "recall"]:
                        if metric in res[task]:
                            metrics_by_k[k][task][metric] = res[task][metric]

    return metrics_by_k


def plot_metrics_vs_folds(
    metrics_by_k: dict,
    duration: str,
    metric: str = "all",
    save: bool = False,
    output_dir: Path = None,
    global_by_k: dict | None = None,
):
    """Grafica métricas vs número de folds.
    
    Cuando metric='all', crea una figura única con todas las métricas
    para las 3 etiquetas en una sola gráfica con una leyenda compartida.
    """

    # Ordenar por k
    k_values = sorted(metrics_by_k.keys())

    if len(k_values) < 2:
        print(f"Advertencia: Solo hay datos para k={k_values}")
        print("Ejecuta entrenar.py con diferentes valores de --k-folds para comparar.")

    tasks = ["plate", "electrode", "current"]
    task_names = {
        "plate": "Plate Thickness",
        "electrode": "Electrode Type",
        "current": "Current Type",
    }
    colors = {
        "plate": "#2ecc71",
        "electrode": "#3498db",
        "current": "#e74c3c",
    }
    
    # Marcadores diferentes para cada métrica
    markers = {
        "accuracy": "o",
        "f1": "s",
        "precision": "^",
        "recall": "D",
    }

    # Si metric es 'all', crear una sola figura con todas las métricas
    if metric == "all":
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        all_values = []
        
        # Graficar cada combinación de tarea y métrica
        for task in tasks:
            for metric_name in ["accuracy", "f1"]:  # Solo accuracy y f1 para claridad
                y_values = []
                
                for k in k_values:
                    value = metrics_by_k[k][task].get(metric_name)
                    if value is not None:
                        y_values.append(value)
                    else:
                        y_values.append(np.nan)
                
                all_values.extend([v for v in y_values if not np.isnan(v)])
                
                # Etiqueta combinada
                label = f"{task_names[task]} - {metric_name.capitalize()}"
                linestyle = "-" if metric_name == "accuracy" else "--"
                
                ax.plot(
                    k_values,
                    y_values,
                    marker=markers[metric_name],
                    label=label,
                    color=colors[task],
                    linewidth=2,
                    markersize=6,
                    linestyle=linestyle,
                )
        
        ax.set_xlabel("Número de Folds (K)", fontsize=12)
        ax.set_ylabel("Valor de Métrica", fontsize=12)
        ax.set_title(f"Métricas vs K-Folds - {duration}", fontsize=14)
        ax.legend(loc="best", fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_values)
        
        # Ajustar límites del eje Y
        if all_values:
            y_min = min(all_values)
            y_max = max(all_values)
            span = y_max - y_min
            pad = max(0.02, span * 0.1)
            y_lower = max(0.0, y_min - pad)
            y_upper = min(1.05, y_max + pad)
            ax.set_ylim([y_lower, y_upper])
        else:
            ax.set_ylim([0.0, 1.05])
        
        if ax.get_ylim()[1] >= 1.0:
            ax.axhline(1.0, color="#666666", linestyle="--", linewidth=1, alpha=0.6)
        
        plt.tight_layout()
        
        if save:
            if output_dir is None:
                output_dir = ROOT_DIR / duration / "metricas"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"metricas_vs_folds_all.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Gráfica guardada en: {output_path}")
        
        plt.close(fig)
    
    else:
        # Para una métrica específica, mantener el formato original
        metrics_to_plot = [metric]
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        all_values = []
        
        for task in tasks:
            y_values = []
            
            for k in k_values:
                value = metrics_by_k[k][task].get(metric)
                if value is not None:
                    y_values.append(value)
                else:
                    y_values.append(np.nan)
            
            all_values.extend([v for v in y_values if not np.isnan(v)])
            
            ax.plot(
                k_values,
                y_values,
                marker="o",
                label=task_names[task],
                color=colors[task],
                linewidth=2,
                markersize=6,
            )
        
        ax.set_xlabel("Número de Folds (K)", fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f"{metric.capitalize()} vs K-Folds - {duration}", fontsize=14)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(k_values)
        
        if all_values:
            y_min = min(all_values)
            y_max = max(all_values)
            span = y_max - y_min
            pad = max(0.02, span * 0.1)
            y_lower = max(0.0, y_min - pad)
            y_upper = min(1.05, y_max + pad)
            ax.set_ylim([y_lower, y_upper])
        else:
            ax.set_ylim([0.0, 1.05])
        
        if ax.get_ylim()[1] >= 1.0:
            ax.axhline(1.0, color="#666666", linestyle="--", linewidth=1, alpha=0.6)
        
        plt.tight_layout()
        
        if save:
            if output_dir is None:
                output_dir = ROOT_DIR / duration / "metricas"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"metricas_vs_folds_{metric}.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Gráfica guardada en: {output_path}")
        
        plt.close(fig)

    # Gráfica adicional: métricas globales (Exact Match y Hamming)
    if global_by_k:
        k_global = sorted(global_by_k.keys())
        exact_values = [global_by_k[k].get("exact_match") for k in k_global]
        hamming_values = [global_by_k[k].get("hamming") for k in k_global]

        fig_g, ax_g = plt.subplots(1, 1, figsize=(6, 5))
        ax_g.plot(
            k_global,
            exact_values,
            marker="o",
            label="Exact Match",
            color="#8e44ad",
            linewidth=2,
            markersize=6,
        )
        ax_g.plot(
            k_global,
            hamming_values,
            marker="o",
            label="Hamming",
            color="#16a085",
            linewidth=2,
            markersize=6,
        )

        ax_g.set_xlabel("Número de Folds (K)")
        ax_g.set_ylabel("Métrica Global")
        ax_g.set_title(f"Métricas Globales vs K-Folds - {duration}")
        ax_g.grid(True, alpha=0.3)
        ax_g.legend(loc="best")
        ax_g.set_xticks(k_global)

        # Ajustar límites del eje Y (rango dinámico)
        vals = [v for v in exact_values + hamming_values if v is not None]
        if vals:
            y_min = min(vals)
            y_max = max(vals)
            span = y_max - y_min
            pad = max(0.02, span * 0.1)
            y_lower = max(0.0, y_min - pad)
            y_upper = min(1.05, y_max + pad)
            ax_g.set_ylim([y_lower, y_upper])
        else:
            ax_g.set_ylim([0.0, 1.05])

        # Línea de referencia en 1.0 si está en el rango
        if ax_g.get_ylim()[1] >= 1.0:
            ax_g.axhline(1.0, color="#666666", linestyle="--", linewidth=1, alpha=0.6)

        plt.tight_layout()

        if save:
            output_path = output_dir / "metricas_globales_vs_folds.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Gráfica guardada en: {output_path}")

        plt.close(fig_g)


def main():
    parser = argparse.ArgumentParser(
        description="Grafica métricas vs cantidad de folds para una duración específica"
    )
    parser.add_argument(
        "duration",
        type=str,
        help="Directorio de duración (ej: 5seg, 10seg, 30seg)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="all",
        choices=["all", "accuracy", "f1", "precision", "recall"],
        help="Métrica a graficar (default: all)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Guardar gráfica como imagen",
    )

    args = parser.parse_args()

    # Verificar que el directorio existe
    duration_dir = ROOT_DIR / args.duration
    if not duration_dir.exists():
        print(f"Error: No se encontró el directorio {duration_dir}")
        print(f"Directorios disponibles: 1seg, 2seg, 5seg, 10seg, 20seg, 30seg, 50seg")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"Graficando métricas vs folds para: {args.duration}")
    print(f"{'=' * 60}")

    # Cargar y procesar resultados
    results = load_results(args.duration)
    print(f"Cargadas {len(results)} entradas de resultados")

    metrics_by_k = extract_metrics_by_folds(results)
    global_by_k = load_global_metrics(args.duration)
    print(f"Valores de K encontrados: {sorted(metrics_by_k.keys())}")

    # Graficar
    metrics_dir = duration_dir / "metricas"
    metrics_dir.mkdir(exist_ok=True)
    plot_metrics_vs_folds(
        metrics_by_k,
        args.duration,
        metric=args.metric,
        save=args.save,
        output_dir=metrics_dir,
        global_by_k=global_by_k,
    )


if __name__ == "__main__":
    main()
