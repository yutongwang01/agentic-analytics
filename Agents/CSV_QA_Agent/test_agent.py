"""Required public entrypoint for testing the CSV QA agent."""

from __future__ import annotations

from pprint import pprint

from agent import run_workflow


def run_agent(question: str, csv_path: str) -> dict:
    """Run the minimal CSV agent and return the required grading dictionary."""
    return run_workflow(question, csv_path)


if __name__ == "__main__":
    mode = input("Choose mode ('sample' or 'interactive'): ").strip().lower()

    if mode == "sample":
        sample_questions = [
            "Identify the top 5 release years with the highest number of titles, and explain what these peaks might suggest about Netflix's content strategy.",
            "Find countries that have relatively few titles overall but a high proportion of TV Shows compared with Movies. What might explain this pattern?",
            "Identify release years where Movies and TV Shows are distributed very differently. What does this suggest about how Netflix's catalog changed over time?",
        ]
        sample_csv_path = "custom_dataset.csv"

        for i, question in enumerate(sample_questions, 1):
            print(f"\n--- Test {i} ---")
            print("Question:", question)
            output = run_agent(question, sample_csv_path)
            pprint(output)

    else:
        csv_path = input("Enter CSV path: ").strip()
        question = input("Enter your question: ").strip()
        output = run_agent(question, csv_path)
        pprint(output)