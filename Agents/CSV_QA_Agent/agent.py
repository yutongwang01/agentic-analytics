"""Minimal workflow wrapper for the CSV QA agent."""

from __future__ import annotations

from nodes import (
    evaluate_result,
    execute_code,
    format_final_answer,
    generate_code,
    load_env_file,
)


def run_workflow(question: str, csv_path: str) -> dict:
    """Run the minimal input -> generate -> execute -> evaluate -> answer workflow."""
    config = load_env_file()
    api_key = config.get("OPENAI_API_KEY", "")
    model_name = config.get("MODEL_NAME", "")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing from .env")
    if not model_name:
        raise ValueError("MODEL_NAME is missing from .env")

    generated_code = generate_code(
        question=question,
        csv_path=csv_path,
        model_name=model_name,
        api_key=api_key,
    )
    execution_result, execution_error = execute_code(generated_code, csv_path)
    evaluation = evaluate_result(execution_error, execution_result)
    final_answer = format_final_answer(
        execution_result,
        evaluation,
        question,
        error_message=execution_error,
    )

    return {
        "generated_code": generated_code,
        "execution_result": execution_result,
        "evaluation": evaluation,
        "final_answer": final_answer,
    }
