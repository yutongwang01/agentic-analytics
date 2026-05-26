"""End-to-end test runner for the plan-and-execute agent.

Behavior:
  - Loads 15 questions defined in QUESTIONS below (5 basic / 5 intermediate / 5 advanced).
  - Runs each through agent.run_workflow.
  - Writes:
      * results.csv          (question, plan, final_answer)              
      * execution_traces.txt (full step-by-step trace per question)     
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

from agent import run_workflow


# ---------------------------------------------------------------------------
# Question bank
# ---------------------------------------------------------------------------
#
# Starter questions reference the online_retail_II schema:
#   invoice_id, stock_code, description, quantity, invoice_date,
#   unit_price, customer_id, country, is_cancel
#
QUESTIONS: list[dict] = [
    # ---- 5 BASIC (filter + aggregate, or group + sort + limit) -----------
    {"id": 1, "level": "basic",
     "question": "Among non-cancelled orders, which 5 countries have the largest "
                 "number of invoice rows? Return the country names and counts."},
    {"id": 2, "level": "basic",
     "question": "How many distinct stock codes and how many distinct customers "
                 "are present in the non-cancelled transactions?"},
    {"id": 3, "level": "basic",
     "question": "For December 2011 only, which 5 countries had the highest total "
                 "revenue (quantity unit_price) among non-cancelled orders?"},
    {"id": 4, "level": "basic",
     "question": "For non-cancelled orders, which 10 countries have the highest "
                 "fraction of weekend transactions (Saturday or Sunday)? Report "
                 "country, weekend_rows, total_rows, and weekend_ratio."},
    {"id": 5, "level": "basic",
     "question": "Among non-cancelled orders with unit_price greater than 10, which 10 "
                 "countries have the most transaction rows? Return country and count."},

    # ---- 5 INTERMEDIATE (derived column + filter + group + sort + limit) -
    {"id": 6, "level": "intermediate",
     "question": "Compute revenue = quantity * unit_price for non-cancelled orders, "
                 "then return the top 10 stock codes by total revenue along with their "
                 "descriptions and total_revenue."},
    {"id": 7, "level": "intermediate",
     "question": "Show the month-by-month total revenue (quantity * unit_price) for the "
                 "United Kingdom in 2011, considering only non-cancelled orders, sorted chronologically."},
    {"id": 8,  "level": "intermediate",
     "question": "For non-cancelled orders, compute the average daily revenue per quarter "
                 "across the entire 13-month dataset. Return quarter and avg_daily_revenue, sorted by quarter."},
    {"id": 9,  "level": "intermediate",
     "question": "Among non-cancelled orders, how many customers purchased at least 2 distinct stock codes?"},
    {"id": 10, "level": "intermediate",
     "question": "Define a 'large order' as an invoice row with quantity >= 50. Among non-cancelled orders, "
                 "list the top 5 stock codes by row count separately for 'large' (quantity >= 50) orders. "
                 "Return stock_code, description, and row_count."},

    # ---- 5 ADVANCED (>= 5 steps, may need rank / multi-stage aggregation) -
    {"id": 11, "level": "advanced",
     "question": "For each country (top 10 by total revenue), find the single stock code with the "
                 "highest total revenue (quantity * unit_price, non-cancelled). Return country, "
                 "top_stock_code, description, country_total_revenue, and stock_revenue_in_country, "
                 "sorted by country_total_revenue descending."},
    {"id": 12, "level": "advanced",
     "question": "Among customers with at least 5 non-cancelled invoices, who are the 10 "
                 "customers with the highest average revenue per invoice? Return customer_id, "
                 "n_invoices, total_revenue, and avg_invoice_revenue."},
    {"id": 13, "level": "advanced",
     "question": "Compute total revenue per (country, quarter) for non-cancelled orders. Then, "
                  "for each quarter, return the top 3 countries by revenue in that quarter. Output rows: "
                  "quarter, country, revenue, rank_in_quarter."},
    {"id": 14, "level": "advanced",
     "question": "For non-cancelled orders, find the 10 dates on which the average unit_price was highest "
                 "(consider only dates with at least 20 rows to avoid noise). Return date, n_rows, and avg_unit_price."},
    {"id": 15, "level": "advanced",
     "question": "Among non-cancelled orders, compute the average revenue per invoice separately "
                 "for weekend invoices (Sat/Sun) and weekday invoices. Use the invoice's date (each invoice has one date). "
                 "Return: day_type, n_invoices, avg_invoice_revenue."},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _final_table_preview(df: pd.DataFrame | None, max_rows: int = 10) -> str:
    if df is None:
        return "(none)"
    if len(df) == 0:
        return f"<empty: columns={list(df.columns)}>"
    return df.head(max_rows).to_string(index=False)


def _is_placeholder(q: str) -> bool:
    return q.strip().startswith("TODO_")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all(csv_path: str, questions: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv_path = output_dir / "results.csv"
    traces_path = output_dir / "execution_traces.txt"

    results_rows: list[dict] = []
    trace_chunks: list[str] = []

    for q in questions:
        qid = q["id"]
        level = q["level"]
        question = q["question"]

        print(f"\n{'#' * 80}")
        print(f"### Question {qid} [{level}]")
        print(f"### {question}")
        print(f"{'#' * 80}")

        if _is_placeholder(question):
            print("[skipped — placeholder TODO question]")
            results_rows.append({
                "question": question,
                "plan": "",
                "final_answer": "(skipped: placeholder)",
            })
            trace_chunks.append(
                f"### Q{qid} [{level}] {question}\n[skipped — placeholder]\n"
            )
            continue

        try:
            out = run_workflow(question=question, csv_path=csv_path)
        except Exception as exc:  # noqa: BLE001
            print(f"!! pipeline crashed: {exc}")
            results_rows.append({
                "question": question,
                "plan": "",
                "final_answer": f"(pipeline crashed: {exc})",
            })
            trace_chunks.append(
                f"### Q{qid} [{level}] {question}\nPIPELINE CRASH: {exc}\n"
            )
            continue

        plan_json = json.dumps(out["plan"], ensure_ascii=False)
        print("PLAN:")
        print(json.dumps(out["plan"], indent=2, ensure_ascii=False))
        print("\nTRACE:")
        print(out["trace_table"])
        print(f"\nCHECK: {out['check_status']} ({out['check_reason']})")
        print("\nFINAL TABLE (preview):")
        print(_final_table_preview(out["final_table"]))
        print("\nFINAL ANSWER:")
        print(out["final_answer"])

        results_rows.append({
            "question": question,
            "plan": plan_json,
            "final_answer": out["final_answer"],
        })

        chunk = [
            f"### Q{qid} [{level}] {question}",
            "PLAN:",
            json.dumps(out["plan"], indent=2, ensure_ascii=False),
            "",
            "TRACE:",
            out["trace_table"],
            "",
            f"CHECK: {out['check_status']} ({out['check_reason']})",
            "",
            "FINAL TABLE (preview):",
            _final_table_preview(out["final_table"]),
            "",
            "FINAL ANSWER:",
            out["final_answer"],
            "",
            "=" * 80,
            "",
        ]
        trace_chunks.append("\n".join(chunk))

    # results.csv (HW3 §11 mandated format: question, plan, final_answer)
    with results_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "plan", "final_answer"])
        writer.writeheader()
        writer.writerows(results_rows)

    # full traces
    traces_path.write_text("\n".join(trace_chunks), encoding="utf-8")

    print(f"\nSaved results to {results_csv_path}")
    print(f"Saved full traces to {traces_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run the HW3 plan-and-execute agent on the question bank.")
    parser.add_argument("--csv", default="dataset.csv", help="Path to the dataset CSV (default: dataset.csv)")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N questions")
    parser.add_argument("--only", type=int, default=None, help="Only run question with this id (1..15)")
    parser.add_argument("--out", default=".", help="Output directory for results.csv and execution_traces.txt")
    args = parser.parse_args()

    if not Path(args.csv).exists():
        raise SystemExit(f"Dataset not found: {args.csv}")

    questions = QUESTIONS
    if args.only is not None:
        questions = [q for q in QUESTIONS if q["id"] == args.only]
        if not questions:
            raise SystemExit(f"No question with id={args.only}")
    elif args.limit is not None:
        questions = QUESTIONS[: args.limit]

    run_all(csv_path=args.csv, questions=questions, output_dir=Path(args.out))


if __name__ == "__main__":
    main()
