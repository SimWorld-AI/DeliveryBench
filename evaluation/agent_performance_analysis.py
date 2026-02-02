# evaluation/agent_performance_analysis.py
# -*- coding: utf-8 -*-

"""
Agent Performance Analysis (Revised, Minimal)
=============================================

- No reliance on any 'per_hour' fields inside JSON.
- Uses lifecycle_hours for all money-per-hour style metrics.
- Adds S_h (stamina_pph) as ratio/hour:
    S_h = energy.personal_consumed_pct / 100 / active_hours
  If missing, treat as 0.0 (no fancy fallbacks).
- Outputs a concise model-level CSV.
"""

import json
from pathlib import Path
from typing import List, Union, Dict, Any

import argparse
import numpy as np
import pandas as pd


# ----------------------------- helpers -----------------------------


def _safe_positive(x: float, default: float = 1.0) -> float:
    try:
        xv = float(x)
        return xv if xv > 0 else default
    except Exception:
        return default


def _sum_interruption_events(interruptions: Dict[str, Any]) -> int:
    return int(interruptions.get('scooter_depleted', 0)) + \
           int(interruptions.get('hospital_rescue', 0)) + \
           int(interruptions.get('rent_insufficient', 0)) + \
           int(interruptions.get('charge_insufficient', 0))


def _compute_violation_rate_union(orders_obj: Dict[str, Any]) -> float:
    """A completed order counts as a violation if ANY of temp/odor/damage is False."""
    details = orders_obj.get('completed_order_details', []) or []
    if not details:
        return 0.0
    violations = 0
    for od in details:
        if not (od.get('temp_ok_all', True)
                and od.get('odor_ok_all', True)
                and od.get('damage_ok_all', True)):
            violations += 1
    return violations / len(details)


def _avg_pick_quality(orders_obj: Dict[str, Any]) -> float:
    """Average pick_score over completed + open orders (if present)."""
    picks = []
    for od in orders_obj.get('completed_order_details', []) or []:
        if od.get('pick_score') is not None:
            picks.append(float(od['pick_score']))
    for od in orders_obj.get('open_order_details', []) or []:
        if od.get('pick_score') is not None:
            picks.append(float(od['pick_score']))
    return float(np.mean(picks)) if picks else 0.0


def _on_time_rate(orders_obj: Dict[str, Any]) -> float:
    return float(orders_obj.get('ontime_rate', 0.0) or 0.0)


def _time_utilization_ratio(meta_obj: Dict[str, Any], orders_obj: Dict[str, Any]) -> float:
    """
    Sum capped order times (completed + open) / total actual time.
    Denominator: meta['active_elapsed_s'] if present, else active_hours * 3600.
    """
    total_used_s = 0.0

    # Completed
    for od in orders_obj.get('completed_order_details', []) or []:
        dur = float(od.get('duration_s', 0.0) or 0.0)
        lim = od.get('time_limit_s')
        if lim is not None:
            dur = min(dur, float(lim))
        total_used_s += max(0.0, dur)

    # Open
    for od in orders_obj.get('open_order_details', []) or []:
        elapsed = float(od.get('elapsed_s', 0.0) or 0.0)
        lim = od.get('time_limit_s')
        if lim is not None:
            elapsed = min(elapsed, float(lim))
        total_used_s += max(0.0, elapsed)

    denom_s = meta_obj.get('active_elapsed_s')
    if denom_s is None:
        denom_s = _safe_positive(meta_obj.get('active_hours', 0.0), 1.0) * 3600.0
    else:
        denom_s = _safe_positive(denom_s, 1.0)

    return total_used_s / denom_s if denom_s > 0 else 0.0


def _stamina_pph(data: Dict[str, Any], active_hours: float) -> float:
    """
    S_h: ratio/hour (NOT percentage). Minimal version.
    S_h = (energy.personal_consumed_pct / 100) / active_hours
    If missing, treat as 0.0.
    """
    energy = data.get("energy") or {}
    consumed_pct = float(energy.get("personal_consumed_pct") or 0.0)
    ah = _safe_positive(active_hours, 1.0)
    return (consumed_pct / 100.0) / ah


# ----------------------------- core analyzer -----------------------------


class AgentPerformanceAnalyzer:
    def __init__(
        self,
        results_dirs: Union[str, List[str]],
        output_dir: Union[str, Path] = None,
    ):
        self.results_dirs = [Path(results_dirs)] if isinstance(results_dirs, str) else [Path(d) for d in results_dirs]
        self.output_dir = Path(output_dir) if output_dir is not None else Path("evaluation") / "summary"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.agents_data: Dict[str, Dict[str, Any]] = {}
        self.by_model: Dict[str, List[Dict[str, Any]]] = {}
        self._load_all()

    def _load_all(self):
        total = 0
        for root in self.results_dirs:
            for f in root.rglob("run_report_agent*.json"):
                try:
                    data = json.loads(Path(f).read_text(encoding="utf-8"))
                except Exception:
                    continue
                meta = data.get("meta", {})
                try:
                    rel = f.parent.relative_to(root).as_posix()
                except Exception:
                    rel = f.parent.as_posix()
                meta["source_dir"] = f"{root.name}" if rel in ("", ".", "/") else f"{root.name}/{rel}"
                data["meta"] = meta

                unique_key = f"{meta.get('source_dir','unk')}__agent{meta.get('agent_id','?')}"
                self.agents_data[unique_key] = data
                self.by_model.setdefault(meta.get("model", "unknown"), []).append(data)
                total += 1

        print(f"Loaded {total} agents from {len(self.results_dirs)} dir(s).")

    # ------------------------- extraction -------------------------

    def _money_per_lifecycle_hour(self, totals: Dict[str, Any], lifecycle_hours: float) -> Dict[str, float]:
        lh = _safe_positive(lifecycle_hours, 1.0)
        return dict(
            per_life_net_growth=float(totals.get("net_growth", 0.0)) / lh,
            per_life_base=float(totals.get("base_wage", 0.0)) / lh,
            per_life_bonus=float(totals.get("bonus", 0.0)) / lh,
            per_life_expense=float(totals.get("expense_total", 0.0)) / lh,
        )

    def extract_agent_row(self, key: str, data: Dict[str, Any]) -> Dict[str, Any]:
        meta = data.get("meta", {})
        money = data.get("money", {})
        totals = money.get("totals", {})
        orders = data.get("orders", {})
        activity = data.get("activity", {})
        interruptions = data.get("interruptions", {})
        prevention = data.get("prevention", {})

        lifecycle_hours = _safe_positive(meta.get("lifecycle_hours", 0.0), 1.0)
        active_hours = _safe_positive(meta.get("active_hours", 0.0), 1.0)

        money_norm = self._money_per_lifecycle_hour(totals, lifecycle_hours)

        # Core metrics
        pick_quality = _avg_pick_quality(orders)
        on_time = _on_time_rate(orders)
        time_util_ratio = _time_utilization_ratio(meta, orders)
        eff_time_ratio = float(activity.get("effective_time_ratio", 0.0) or 0.0)
        stamina_pph = _stamina_pph(data, active_hours)  # ratio/hour
        intr_total = _sum_interruption_events(interruptions)
        intr_per_hour = intr_total / active_hours
        prevent_rate = float(prevention.get("prevention_rate", 0.0) or 0.0)
        viol_rate = _compute_violation_rate_union(orders)
        avg_food_stars = float(orders.get("avg_food_stars", 0.0) or 0.0)
        avg_stars = float(orders.get("avg_stars", 0.0) or 0.0)

        return dict(
            unique_key=key,
            agent_id=meta.get("agent_id"),
            model=meta.get("model"),
            source_dir=meta.get("source_dir"),

            # money normalized by lifecycle time
            per_life_net_growth=money_norm["per_life_net_growth"],
            per_life_base=money_norm["per_life_base"],
            per_life_bonus=money_norm["per_life_bonus"],
            per_life_expense=money_norm["per_life_expense"],

            # new required metrics
            pick_quality_score=pick_quality,
            on_time_rate=on_time,
            time_utilization_ratio=time_util_ratio,
            effective_time_ratio=eff_time_ratio,
            stamina_pph=stamina_pph,                 # ← before interruptions
            interruptions_per_hour=intr_per_hour,
            prevention_rate=prevent_rate,
            violation_rate_union=viol_rate,
            avg_food_stars=avg_food_stars,
            avg_stars=avg_stars,

            # reference counts
            completed_orders=int(orders.get("completed_count", 0) or 0),
            timeout_orders=int(orders.get("timeout_count", 0) or 0),
            active_hours=active_hours,
            lifecycle_hours=lifecycle_hours,
        )

    def extract_all_agents_df(self) -> pd.DataFrame:
        rows = [self.extract_agent_row(k, v) for k, v in self.agents_data.items()]
        return pd.DataFrame(rows)

    # ------------------------- aggregation -------------------------

    def model_average(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute model-level means/std/sums for key fields."""
        agg_map = {
            # money per lifecycle-hour
            "per_life_net_growth": ["mean", "std"],
            "per_life_base": ["mean", "std"],
            "per_life_bonus": ["mean", "std"],
            "per_life_expense": ["mean", "std"],

            # new metrics
            "pick_quality_score": ["mean", "std"],
            "on_time_rate": ["mean", "std"],
            "time_utilization_ratio": ["mean", "std"],
            "effective_time_ratio": ["mean", "std"],
            "stamina_pph": ["mean", "std"],
            "interruptions_per_hour": ["mean", "std"],
            "prevention_rate": ["mean", "std"],
            "violation_rate_union": ["mean", "std"],
            "avg_food_stars": ["mean", "std"],
            "avg_stars": ["mean", "std"],

            # reference counts
            "completed_orders": ["mean", "std", "sum"],
            "timeout_orders": ["mean", "std", "sum"],
            "active_hours": ["mean", "std", "sum"],
            "lifecycle_hours": ["mean", "std", "sum"],
        }
        grouped = df.groupby("model").agg(agg_map)

        # Flatten MultiIndex columns
        grouped.columns = ["_".join([c for c in col if c]) for col in grouped.columns.values]
        grouped = grouped.reset_index()

        # rename to cleaner headers for CSV
        rename = {
            "model": "Model",

            "per_life_net_growth_mean": "Avg_Per_Life_Net_Growth",
            "per_life_net_growth_std": "Std_Per_Life_Net_Growth",
            "per_life_base_mean": "Avg_Per_Life_Base",
            "per_life_base_std": "Std_Per_Life_Base",
            "per_life_bonus_mean": "Avg_Per_Life_Bonus",
            "per_life_bonus_std": "Std_Per_Life_Bonus",
            "per_life_expense_mean": "Avg_Per_Life_Expense",
            "per_life_expense_std": "Std_Per_Life_Expense",

            "pick_quality_score_mean": "Avg_Pick_Quality_Score",
            "pick_quality_score_std": "Std_Pick_Quality_Score",
            "on_time_rate_mean": "Avg_On_Time_Rate",
            "on_time_rate_std": "Std_On_Time_Rate",
            "time_utilization_ratio_mean": "Avg_Time_Utilization_Ratio",
            "time_utilization_ratio_std": "Std_Time_Utilization_Ratio",
            "effective_time_ratio_mean": "Avg_Effective_Time_Ratio",
            "effective_time_ratio_std": "Std_Effective_Time_Ratio",

            "stamina_pph_mean": "Avg_Stamina_Per_Hour",
            "stamina_pph_std": "Std_Stamina_Per_Hour",

            "interruptions_per_hour_mean": "Avg_Interruptions_Per_Hour",
            "interruptions_per_hour_std": "Std_Interruptions_Per_Hour",
            "prevention_rate_mean": "Avg_Prevention_Rate",
            "prevention_rate_std": "Std_Prevention_Rate",
            "violation_rate_union_mean": "Avg_Violation_Rate",
            "violation_rate_union_std": "Std_Violation_Rate",
            "avg_food_stars_mean": "Avg_Food_Stars",
            "avg_food_stars_std": "Std_Food_Stars",
            "avg_stars_mean": "Avg_Stars",
            "avg_stars_std": "Std_Stars",

            "completed_orders_mean": "Avg_Completed_Orders_Per_Agent",
            "completed_orders_std": "Std_Completed_Orders_Per_Agent",
            "completed_orders_sum": "Total_Completed_Orders",
            "timeout_orders_mean": "Avg_Timeout_Orders_Per_Agent",
            "timeout_orders_std": "Std_Timeout_Orders_Per_Agent",
            "timeout_orders_sum": "Total_Timeout_Orders",
            "active_hours_mean": "Avg_Active_Hours_Per_Agent",
            "active_hours_std": "Std_Active_Hours_Per_Agent",
            "active_hours_sum": "Total_Active_Hours",
            "lifecycle_hours_mean": "Avg_Lifecycle_Hours_Per_Agent",
            "lifecycle_hours_std": "Std_Lifecycle_Hours_Per_Agent",
            "lifecycle_hours_sum": "Total_Lifecycle_Hours",
        }
        grouped = grouped.rename(columns=rename)

        # order columns for CSV
        first_cols = [
            "Model",
            "Avg_Per_Life_Net_Growth",
            "Avg_Per_Life_Base",
            "Avg_Per_Life_Bonus",
            "Avg_Per_Life_Expense",

            "Avg_Pick_Quality_Score",
            "Avg_On_Time_Rate",
            "Avg_Time_Utilization_Ratio",
            "Avg_Effective_Time_Ratio",
            "Avg_Stamina_Per_Hour",          # before interruptions
            "Avg_Interruptions_Per_Hour",
            "Avg_Prevention_Rate",
            "Avg_Violation_Rate",

            "Avg_Food_Stars",
            "Avg_Stars",
        ]
        remaining = [c for c in grouped.columns if c not in first_cols]
        grouped = grouped[first_cols + remaining]
        return grouped

    # ------------------------- main entry -------------------------

    def run(self) -> pd.DataFrame:
        df_agents = self.extract_all_agents_df()
        model_summary = self.model_average(df_agents)

        csv_path = self.output_dir / "model_performance_summary.csv"
        model_summary.to_csv(csv_path, index=False)
        print(f"✅ Saved model summary to: {csv_path}")

        return model_summary


# ----------------------------- CLI -----------------------------


def main(paths: Union[str, List[str]], output_dir: Union[str, Path] = "evaluation/summary") -> pd.DataFrame:
    analyzer = AgentPerformanceAnalyzer(results_dirs=paths, output_dir=output_dir)
    return analyzer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize DeliveryBench agent performance over one or more result directories."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more root directories containing run_report_agent*.json files.",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="evaluation/summary",
        help="Output directory for the summary CSV (default: evaluation/summary).",
    )
    args = parser.parse_args()

    results = main(paths=args.paths, output_dir=args.output_dir)