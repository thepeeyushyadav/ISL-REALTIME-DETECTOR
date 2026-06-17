"""
data_collection/verify_data.py
================================
Verifies dataset integrity and generates a quality report.

Checks:
  - Sequence count per sign
  - Shape consistency (30, 225)
  - NaN/Inf values
  - Zero-frame sequences (no hands detected)
  - Class balance

Usage:
    python data_collection/verify_data.py
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ISL_SIGNS, DATASET_DIR, SEQUENCE_LENGTH, NUM_FEATURES, BASE_DIR

MIN_SEQUENCES = 20
IDEAL_SEQUENCES = 50


def check_sequence(npy_path: str) -> dict:
    """Check a single .npy sequence for quality issues."""
    issues = []
    try:
        data = np.load(npy_path)
        if data.shape != (SEQUENCE_LENGTH, NUM_FEATURES):
            issues.append(f"Wrong shape: {data.shape}")
        if np.any(np.isnan(data)):
            issues.append("Contains NaN")
        if np.any(np.isinf(data)):
            issues.append("Contains Inf")
        if np.all(data == 0):
            issues.append("All zeros (no landmarks detected)")

        # Check if hands are mostly invisible
        # Hands are at indices 99:225 (left_hand: 99:162, right_hand: 162:225)
        hand_data = data[:, 99:]
        zero_frames = np.sum(np.all(hand_data == 0, axis=1))
        zero_pct = zero_frames / SEQUENCE_LENGTH
        if zero_pct > 0.5:
            issues.append(f"Hands invisible in {zero_pct*100:.0f}% of frames")

        return {"ok": len(issues) == 0, "issues": issues, "shape": data.shape}
    except Exception as e:
        return {"ok": False, "issues": [f"Load error: {e}"], "shape": None}


def verify_dataset() -> dict:
    """Run full dataset verification and return stats dict."""
    report = {}

    for sign in ISL_SIGNS:
        sign_dir = os.path.join(DATASET_DIR, sign)
        sign_report = {
            "total": 0,
            "ok": 0,
            "issues": defaultdict(int),
            "bad_files": [],
        }

        if not os.path.exists(sign_dir):
            sign_report["issues"]["Directory missing"] = 1
            report[sign] = sign_report
            continue

        npy_files = sorted(Path(sign_dir).glob("*.npy"))
        sign_report["total"] = len(npy_files)

        for npy_file in npy_files:
            result = check_sequence(str(npy_file))
            if result["ok"]:
                sign_report["ok"] += 1
            else:
                sign_report["bad_files"].append(npy_file.name)
                for issue in result["issues"]:
                    sign_report["issues"][issue] += 1

        report[sign] = sign_report

    return report


def print_report(report: dict) -> None:
    """Print a formatted verification report to console."""
    print("\n" + "=" * 65)
    print("  ISL DATASET VERIFICATION REPORT")
    print("=" * 65)
    print(f"  {'Sign':<18} {'Total':>7} {'OK':>7} {'Status':<12} Issues")
    print("-" * 65)

    total_ok = 0
    total_seqs = 0

    for sign in ISL_SIGNS:
        r = report.get(sign, {})
        total = r.get("total", 0)
        ok = r.get("ok", 0)
        total_ok += ok
        total_seqs += total

        if total == 0:
            status = "❌ MISSING"
        elif ok < MIN_SEQUENCES:
            status = "⚠  LOW"
        elif ok < IDEAL_SEQUENCES:
            status = "✓  OK"
        else:
            status = "✅ GREAT"

        issues_str = ", ".join(r.get("issues", {}).keys()) if r.get("issues") else ""
        print(f"  {sign:<18} {total:>7} {ok:>7} {status:<12} {issues_str}")

    print("=" * 65)
    print(f"  TOTAL: {total_seqs} sequences, {total_ok} clean")
    ready = sum(1 for s in ISL_SIGNS if report.get(s, {}).get("ok", 0) >= MIN_SEQUENCES)
    print(f"  Ready for training: {ready}/{len(ISL_SIGNS)} signs")


def plot_class_distribution(report: dict) -> None:
    """Save a bar chart of class distribution."""
    signs = list(ISL_SIGNS)
    counts_ok = [report.get(s, {}).get("ok", 0) for s in signs]
    counts_bad = [
        report.get(s, {}).get("total", 0) - report.get(s, {}).get("ok", 0)
        for s in signs
    ]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(signs))
    bars1 = ax.bar(x, counts_ok, label="Valid sequences", color="#00e876", alpha=0.85)
    bars2 = ax.bar(x, counts_bad, bottom=counts_ok, label="Bad sequences", color="#dc3545", alpha=0.7)

    ax.axhline(MIN_SEQUENCES, color="orange", linestyle="--", linewidth=1.5, label=f"Min threshold ({MIN_SEQUENCES})")
    ax.axhline(IDEAL_SEQUENCES, color="cyan", linestyle="--", linewidth=1.5, label=f"Ideal ({IDEAL_SEQUENCES})")

    ax.set_xticks(x)
    ax.set_xticklabels(signs, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Number of Sequences")
    ax.set_title("ISL Dataset — Class Distribution", fontsize=14, fontweight="bold")
    ax.legend()
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#0f0f23")
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    plt.tight_layout()

    out_path = os.path.join(BASE_DIR, "dataset", "class_distribution.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  📊 Class distribution chart saved: {out_path}")


def main():
    print("Running dataset verification...")

    report = verify_dataset()
    print_report(report)
    plot_class_distribution(report)

    # Save JSON report
    report_path = os.path.join(BASE_DIR, "dataset", "verification_report.json")
    serializable = {
        k: {
            "total": v.get("total", 0),
            "ok": v.get("ok", 0),
            "issues": dict(v.get("issues", {})),
            "bad_files": v.get("bad_files", []),
        }
        for k, v in report.items()
    }
    with open(report_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  📄 Report saved: {report_path}")

    # Check if ready to train
    ready = sum(1 for s in ISL_SIGNS if report.get(s, {}).get("ok", 0) >= MIN_SEQUENCES)
    if ready == len(ISL_SIGNS):
        print("\n✅ All signs ready! Run: python model/train_model.py")
    else:
        missing = [s for s in ISL_SIGNS if report.get(s, {}).get("ok", 0) < MIN_SEQUENCES]
        print(f"\n⚠  {len(missing)} signs need more data:")
        for s in missing:
            have = report.get(s, {}).get("ok", 0)
            print(f"   - {s}: {have}/{MIN_SEQUENCES} sequences")
        print("\nRun: python data_collection/collect_data.py  to record missing signs")


if __name__ == "__main__":
    main()
