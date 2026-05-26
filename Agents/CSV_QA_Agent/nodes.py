"""Helper nodes for a minimal CSV question-answering workflow."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from openai import OpenAI


def load_env_file(env_path: str = ".env") -> dict:
    """Load simple KEY=VALUE pairs from a .env file without extra dependencies."""
    values: dict[str, str] = {}
    path = Path(env_path)

    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")

    return values


def inspect_csv_schema(csv_path: str) -> dict:
    """Read a small amount of the CSV so code generation can adapt to the current schema."""
    df = pd.read_csv(csv_path)
    preview_df = df.head(3).where(pd.notna(df.head(3)), None)
    dtype_map = {column: str(dtype) for column, dtype in df.dtypes.items()}
    preview_rows = preview_df.to_dict(orient="records")

    return {
        "columns": list(df.columns),
        "dtypes": dtype_map,
        "preview_rows": preview_rows,
    }


def generate_code(question: str, csv_path: str, model_name: str, api_key: str) -> str:
    """Ask the OpenAI model to produce Python code that stores the answer in `result`."""
    client = OpenAI(api_key=api_key)
    schema = inspect_csv_schema(csv_path)
    columns_text = ", ".join(schema["columns"])
    dtypes_text = "\n".join(f"- {column}: {dtype}" for column, dtype in schema["dtypes"].items())
    preview_text = "\n".join(f"- {row}" for row in schema["preview_rows"])
    question_lower = question.lower()
    extra_instructions = []

    extra_instructions.append(
        "- when iterating over pandas groupby results, use `for group_name, group_df in ...`; "
        "do not access `group_df.name`"
    )

    if "similar median income" in question_lower and "different median house value" in question_lower:
        extra_instructions.extend(
            [
                "- prefer grouped or area-level analysis instead of returning raw rows",
                "- compare a small number of representative areas with similar median income but noticeably different median house value",
                "- enforce similar income levels by grouping areas into narrow median_income bands or rounded median_income bins before comparing areas",
                "- only compare areas from the same income band or from very close median_income levels",
                "- keep pairs only when the income gap is small but the median house value gap is large",
                "- keep result structured, such as a pandas DataFrame or a list/dict of area summaries",
                "- include useful area-level factors when available, such as ocean_proximity, housing_median_age, total_rooms, population, or households",
                "- include columns such as income_bin, low_value_area, high_value_area, house_value_gap, and income_gap when possible",
            ]
        )

    if "top 5 most expensive geographic areas" in question_lower:
        extra_instructions.extend(
            [
                "- identify geographic areas using simple latitude and longitude bins instead of grouping only by ocean_proximity",
                "- create area-level groups from latitude/longitude ranges and return the top 5 areas by median house value",
                "- keep result structured as a pandas DataFrame",
                "- include columns that help explain the prices, such as median_income, housing_median_age, total_rooms, total_bedrooms, population, households, and ocean_proximity",
            ]
        )

    if "coastal areas" in question_lower and "low" in question_lower and "ocean" in question_lower:
        extra_instructions.extend(
            [
                "- identify low-price coastal areas at an area or group level instead of returning raw rows",
                "- focus on coastal groups such as NEAR OCEAN, <1H OCEAN, or ISLAND when present",
                "- create simple latitude and longitude area bins or another simple geographic grouping for coastal areas",
                "- return a structured pandas DataFrame with a small number of representative low-price coastal areas",
                "- include summary factors such as median_income, housing_median_age, total_rooms, total_bedrooms, population, households, and ocean_proximity",
            ]
        )

    if "distribution of content ratings" in question_lower and "movies" in question_lower and "tv shows" in question_lower:
        extra_instructions.extend(
            [
                "- compare rating distributions between Movies and TV Shows using the `type` and `rating` columns",
                "- before grouping, exclude invalid rating values that look like duration data, such as values ending with `min`, `season`, or `seasons`",
                "- keep the cleaned result structured, for example as a pandas DataFrame or Series showing rating counts by content type",
            ]
        )

    if "relatively few titles overall" in question_lower and "high proportion of tv shows" in question_lower:
        extra_instructions.extend(
            [
                "- group by country and compare Movie vs TV Show counts",
                "- split multi-country cells in the `country` column by commas before analysis so each country is counted separately",
                "- trim whitespace and remove empty country strings after splitting",
                "- restrict to countries with relatively few total titles, using a clear upper bound or a low-count subset",
                "- only keep countries where the TV Show share is clearly high compared with Movies",
                "- prefer countries with both a high TV share and a meaningful gap between TV Show and Movie proportions",
                "- compute results at the single-country level with columns such as country, total_titles, movie_count, tv_show_count, and tv_share",
                "- do not use raw unsplit strings like `Argentina, United States, Mexico` as country groups",
            ]
        )

    if "release years where movies and tv shows are distributed very differently" in question_lower:
        extra_instructions.extend(
            [
                "- compare Movie and TV Show distributions by release_year",
                "- compute total titles per year before ranking differences",
                "- filter out years with very small totals before ranking, using a minimum total such as 50 or 100",
                "- after filtering, rank years by a meaningful difference measure such as share difference or absolute Movie vs TV Show count difference",
                "- keep the result structured, for example with columns such as release_year, total_titles, movie_count, tv_show_count, movie_share, tv_share, and share_gap",
            ]
        )

    extra_text = "\n".join(extra_instructions)

    prompt = f"""
You are generating Python code for a CSV question-answering task.

Return only valid Python code. Do not use markdown fences. Do not explain anything.

Requirements:
- import pandas as pd
- read the CSV with: df = pd.read_csv(csv_path)
- answer this question: {question}
- store the final output in a variable named result
- prefer a raw analysis result such as a number, dict, pandas Series, or pandas DataFrame
- do not store explanatory English sentences in result unless the question explicitly asks for text
- do not print anything
- use the dataframe variable name df
- make sure the code reads the CSV exactly once with pandas
- use only columns that actually exist in the provided schema
- never reference a column unless it appears in the provided schema
- do not assume housing.csv columns or any fixed dataset-specific columns
- adapt the analysis to the current csv_path and schema
- if the question refers to a concept that is not present in the dataset, return a structured message in `result`
  such as {{"message": "The dataset does not contain a column for ..."}} instead of hallucinating columns
{extra_text}

Current dataset columns:
{columns_text}

Current dataset dtypes:
{dtypes_text}

Current dataset preview:
{preview_text}
""".strip()

    response = client.responses.create(
        model=model_name,
        input=prompt,
    )
    code = response.output_text.strip()

    if code.startswith("```"):
        code = code.strip("`")
        if code.startswith("python"):
            code = code[len("python") :].lstrip()

    return code


def execute_code(code: str, csv_path: str) -> tuple[object, str | None]:
    """Execute generated code exactly as required and read the `result` variable."""
    env = {}
    env["csv_path"] = csv_path
    try:
        exec(code, env)
    except Exception as exc:  # noqa: BLE001 - simple evaluator needs the raw error.
        error_message = f"{type(exc).__name__}: {exc}"
        return {"error": error_message}, error_message

    return env.get("result"), None


def is_empty_result(result: object) -> bool:
    """Treat missing or empty containers/strings/pandas objects as empty results."""
    if result is None:
        return True

    if isinstance(result, str):
        return result.strip() == ""

    if isinstance(result, (list, tuple, dict, set)):
        return len(result) == 0

    if isinstance(result, (pd.DataFrame, pd.Series)):
        return len(result) == 0

    try:
        return hasattr(result, "__len__") and len(result) == 0
    except TypeError:
        return False


def evaluate_result(execution_error: str | None, result: object) -> str:
    """Return FAIL on execution errors, missing results, or empty results."""
    if execution_error is not None:
        return "FAIL"

    if is_empty_result(result):
        return "FAIL"

    return "PASS"


def _to_plain_value(value: object) -> object:
    """Convert numpy/pandas scalar-like values into plain Python values."""
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (ValueError, TypeError):
            return value
    return value


def _format_scalar_answer(result: object) -> str:
    """Format a scalar result as a short natural-language sentence."""
    plain_result = _to_plain_value(result)
    return f"The result is {plain_result}."


def _format_dict_answer(result: dict) -> str:
    """Format dictionary results by summarizing each section briefly."""
    parts = []
    for key, value in list(result.items())[:3]:
        readable_key = str(key).replace("_", " ")
        if isinstance(value, pd.DataFrame):
            if value.empty:
                parts.append(f"{readable_key}: no rows")
            else:
                preview = value.head(1).to_dict(orient="records")[0]
                preview_text = ", ".join(
                    f"{col}={_format_number_for_text(val)}" for col, val in preview.items()
                )
                parts.append(f"{readable_key}: top row shows {preview_text}")
        elif isinstance(value, pd.Series):
            if value.empty:
                parts.append(f"{readable_key}: no values")
            else:
                top_label = value.idxmax() if pd.api.types.is_numeric_dtype(value) else value.index[0]
                top_value = value.loc[top_label]
                parts.append(f"{readable_key}: {top_label} ({_format_number_for_text(top_value)})")
        else:
            plain_value = _to_plain_value(value)
            parts.append(f"{readable_key}: {_format_number_for_text(plain_value)}")

    return "Key results: " + "; ".join(parts) + "."


def _format_number_for_text(value: object) -> str:
    """Format numeric values compactly for natural-language output."""
    plain_value = _to_plain_value(value)
    if isinstance(plain_value, float):
        return f"{plain_value:.4f}"
    return str(plain_value)


def _describe_correlation(value: float) -> str:
    """Describe the direction and rough strength of a correlation value."""
    abs_value = abs(value)
    if abs_value < 0.1:
        strength = "a very weak"
    elif abs_value < 0.3:
        strength = "a weak"
    elif abs_value < 0.5:
        strength = "a moderate"
    else:
        strength = "a strong"

    if value > 0:
        direction = "positive"
    elif value < 0:
        direction = "negative"
    else:
        direction = "no"

    if value == 0:
        return "There is no linear relationship."

    return f"There is {strength} {direction} relationship (correlation = {_format_number_for_text(value)})."


def _format_scalar_answer_with_question(result: object, question: str) -> str:
    """Format scalar results with light question-aware wording."""
    plain_result = _to_plain_value(result)
    question_lower = question.lower()

    if isinstance(plain_result, (int, float)) and "correlation" in question_lower:
        target = "the variables"
        if "population density" in question_lower:
            target = "Population density"
        elif "median income" in question_lower:
            target = "Median income"
        return (
            f"{target} has"
            if target != "the variables"
            else "The variables have"
        ) + " " + _describe_correlation(float(plain_result)).lower()

    return f"Based on the analysis, the answer is {_format_number_for_text(plain_result)}."


def _format_series_answer(result: pd.Series, question: str) -> str:
    """Format grouped pandas Series results in a more natural way."""
    if result.empty:
        return "The result is empty."

    question_lower = question.lower()
    series = result.copy()

    if pd.api.types.is_numeric_dtype(series):
        if "top" in question_lower or "highest" in question_lower or "largest" in question_lower:
            top_items = series.sort_values(ascending=False).head(3)
            summary = ", ".join(
                f"{idx} ({_format_number_for_text(val)})" for idx, val in top_items.items()
            )
            return f"The ranking is led by {summary}."

        highest_label = series.idxmax()
        highest_value = series.loc[highest_label]
        lowest_label = series.idxmin()
        lowest_value = series.loc[lowest_label]
        return (
            f"The main pattern is that {highest_label} has the highest value "
            f"({_format_number_for_text(highest_value)}), while the lowest is "
            f"{lowest_label} ({_format_number_for_text(lowest_value)})."
        )

    preview = series.head(3)
    summary = ", ".join(f"{idx}: {val}" for idx, val in preview.items())
    return f"The grouped results start with {summary}."


def _format_dataframe_answer(result: pd.DataFrame, question: str) -> str:
    """Format DataFrame results by summarizing the leading rows."""
    if result.empty:
        return "The result is empty."

    question_lower = question.lower()
    preview = result.head(3)

    if "top 5 most expensive geographic areas" in question_lower:
        area_summaries = []
        for _, row in preview.iterrows():
            pieces = []
            for col in [
                "latitude_bin",
                "longitude_bin",
                "area",
                "median_house_value",
                "median_income",
                "housing_median_age",
                "total_rooms",
                "total_bedrooms",
                "population",
                "households",
                "ocean_proximity",
            ]:
                if col in row:
                    label = col.replace("_", " ")
                    pieces.append(f"{label} {_format_number_for_text(row[col])}")
            area_summaries.append(", ".join(str(piece) for piece in pieces if piece))

        return (
            "The most expensive geographic areas are concentrated in these top area groups: "
            + "; ".join(area_summaries)
            + ". Higher prices in these areas appear to be associated with stronger incomes, local housing characteristics, and coastal or neighborhood context."
        )

    if "coastal areas" in question_lower and "low" in question_lower and "ocean" in question_lower:
        area_summaries = []
        for _, row in preview.iterrows():
            pieces = []
            for col in [
                "latitude_bin",
                "longitude_bin",
                "area",
                "ocean_proximity",
                "median_house_value",
                "median_income",
                "housing_median_age",
                "total_rooms",
                "total_bedrooms",
                "population",
                "households",
            ]:
                if col in row:
                    label = col.replace("_", " ")
                    pieces.append(f"{label} {_format_number_for_text(row[col])}")
            area_summaries.append(", ".join(str(piece) for piece in pieces if piece))

        return (
            "The analysis highlights several coastal area groups where house prices remain relatively low despite ocean access: "
            + "; ".join(area_summaries)
            + ". These lower prices may be linked to weaker incomes, older housing stock, denser local conditions, or less favorable housing and neighborhood characteristics compared with other coastal areas."
        )

    if "similar median income" in question_lower and "different median house value" in question_lower:
        summary_rows = []
        for _, row in preview.iterrows():
            low_area = row.get("low_value_area")
            high_area = row.get("high_value_area")
            income_bin = row.get("income_bin", row.get("median_income_bin", "unknown"))
            gap = row.get("house_value_gap", row.get("value_gap", row.get("median_house_value_gap")))
            income_gap = row.get("income_gap", row.get("median_income_gap"))

            if isinstance(low_area, dict) and isinstance(high_area, dict):
                low_ocean = low_area.get("ocean_proximity", "unknown area")
                low_value = low_area.get("median_house_value", "unknown")
                low_income = low_area.get("median_income")
                high_ocean = high_area.get("ocean_proximity", "unknown area")
                high_value = high_area.get("median_house_value", "unknown")
                high_income = high_area.get("median_income")
                gap_text = (
                    _format_number_for_text(gap)
                    if gap is not None
                    else _format_number_for_text(_to_plain_value(high_value) - _to_plain_value(low_value))
                    if isinstance(_to_plain_value(high_value), (int, float))
                    and isinstance(_to_plain_value(low_value), (int, float))
                    else "unknown"
                )
                income_gap_text = (
                    _format_number_for_text(income_gap)
                    if income_gap is not None
                    else _format_number_for_text(abs(_to_plain_value(high_income) - _to_plain_value(low_income)))
                    if isinstance(_to_plain_value(high_income), (int, float))
                    and isinstance(_to_plain_value(low_income), (int, float))
                    else "unknown"
                )
                summary_rows.append(
                    f"For income bin {_format_number_for_text(income_bin)}, "
                    f"an {low_ocean} area with median house value around "
                    f"{_format_number_for_text(low_value)} contrasts with a {high_ocean} area around "
                    f"{_format_number_for_text(high_value)}, with a house value gap of about {gap_text} "
                    f"and an income gap of about {income_gap_text}"
                )
                continue

            pieces = []
            if "ocean_proximity" in row:
                pieces.append(f"ocean proximity {row['ocean_proximity']}")
            if "median_income" in row:
                pieces.append(f"median income {_format_number_for_text(row['median_income'])}")
            if "median_house_value" in row:
                pieces.append(f"median house value {_format_number_for_text(row['median_house_value'])}")
            if "housing_median_age" in row:
                pieces.append(f"housing median age {_format_number_for_text(row['housing_median_age'])}")
            if "population" in row:
                pieces.append(f"population {_format_number_for_text(row['population'])}")
            if pieces:
                summary_rows.append(", ".join(str(piece) for piece in pieces if piece))

        return (
            "The analysis found areas with similar median income but different house values. "
            "The main differences appear in these representative areas: "
            + "; ".join(summary_rows)
            + ". These gaps may be related to factors such as ocean proximity, housing age, and local population patterns."
        )

    if "top" in question_lower or "highest" in question_lower or "largest" in question_lower:
        rows = []
        for _, row in preview.iterrows():
            row_text = ", ".join(f"{col}={_format_number_for_text(val)}" for col, val in row.items())
            rows.append(row_text)
        return "The top entries are: " + "; ".join(rows) + "."

    row_summaries = []
    for _, row in preview.iterrows():
        row_text = ", ".join(f"{col}={_format_number_for_text(val)}" for col, val in row.items())
        row_summaries.append(row_text)
    return "The main result is summarized by these leading rows: " + "; ".join(row_summaries) + "."


def format_final_answer(
    result: object,
    evaluation: str,
    question: str,
    error_message: str | None = None,
) -> str:
    """Convert the execution outcome into the final answer string."""
    if evaluation == "PASS":
        if isinstance(result, dict):
            return _format_dict_answer(result)
        if isinstance(result, pd.Series):
            return _format_series_answer(result, question)
        if isinstance(result, pd.DataFrame):
            return _format_dataframe_answer(result, question)
        return _format_scalar_answer_with_question(result, question)

    if error_message:
        return f"Execution failed: {error_message}"

    return "Execution failed because no usable result was produced."
