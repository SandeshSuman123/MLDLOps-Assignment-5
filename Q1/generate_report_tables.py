"""
generate_report_tables.py
Reads all results JSON files and prints/saves:
  - Table 3 (test comparison across all runs)
  - Per-experiment train-val tables
  - Loss/accuracy plots per experiment

Usage:
    python generate_report_tables.py
"""

import os, json, glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import config


def load_results():
    files = glob.glob(f"{config.OUTPUT_DIR}/*_results.json")
    results = []
    for f in sorted(files):
        with open(f) as fp:
            data = json.load(fp)
            if isinstance(data, list):
                results.extend(data)   # flatten lists
            else:
                results.append(data)
    return results


def print_test_table(results):
    print("\n" + "="*80)
    print(f"{'LoRA':>6} {'Rank':>5} {'Alpha':>6} {'Dropout':>8} "
          f"{'Test Acc':>10} {'Trainable Params':>18}")
    print("-"*80)
    for r in results:
        lora_flag = "Yes" if r.get("lora") else "No"
        rank   = str(r.get("rank",    "-"))
        alpha  = str(r.get("alpha",   "-"))
        drop   = str(r.get("dropout", "-"))
        acc    = f"{r['test_accuracy']:.4f}"
        params = f"{r['trainable_params']:,}"
        print(f"{lora_flag:>6} {rank:>5} {alpha:>6} {drop:>8} {acc:>10} {params:>18}")
    print("="*80)


def print_epoch_table(result):
    print(f"\nExperiment: {result['run']}  "
          f"Rank={result.get('rank','-')}  Alpha={result.get('alpha','-')}")
    print(f"{'Epoch':>6} {'TrLoss':>10} {'ValLoss':>10} {'TrAcc':>10} {'ValAcc':>10}")
    print("-"*50)
    for row in result["history"]:
        print(f"{row['epoch']:>6} {row['train_loss']:>10.4f} {row['val_loss']:>10.4f} "
              f"{row['train_acc']:>10.4f} {row['val_acc']:>10.4f}")


def plot_experiment(result):
    h   = result["history"]
    epochs = [r["epoch"] for r in h]
    run = result["run"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, [r["train_loss"] for r in h], label="Train")
    ax1.plot(epochs, [r["val_loss"]   for r in h], label="Val")
    ax1.set_title(f"Loss — {run}"); ax1.set_xlabel("Epoch")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [r["train_acc"] for r in h], label="Train")
    ax2.plot(epochs, [r["val_acc"]   for r in h], label="Val")
    ax2.set_title(f"Accuracy — {run}"); ax2.set_xlabel("Epoch")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f"{config.OUTPUT_DIR}/{run}_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


def main():
    results = load_results()
    if not results:
        print("No results found. Run training first.")
        return

    print_test_table(results)

    for r in results:
        print_epoch_table(r)
        plot_experiment(r)

    print("\nAll tables and plots generated.")


if __name__ == "__main__":
    main()