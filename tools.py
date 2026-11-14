import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def generate_demo_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2026-01-01", periods=120, freq="D")
    regions = ["Львів", "Київ", "Одеса", "Дніпро", "Харків"]
    categories = {
        "Ноутбуки": (28000, 45000),
        "Смартфони": (12000, 26000),
        "Навушники": (1800, 5500),
        "Монітори": (7500, 16000),
    }

    rows = []
    for d in dates:
        cat = rng.choice(list(categories.keys()))
        low, high = categories[cat]
        units = int(rng.integers(1, 14))
        price = round(float(rng.uniform(low, high)), 2)
        revenue = round(units * price, 2)
        marketing = round(revenue * float(rng.uniform(0.05, 0.16)), 2)
        rows.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "region": str(rng.choice(regions)),
                "category": cat,
                "units_sold": units,
                "unit_price": price,
                "revenue": revenue,
                "marketing_spend": marketing,
            }
        )

    df = pd.DataFrame(rows)

    df.loc[15, ["region", "category", "units_sold", "revenue"]] = [
        "Київ",
        "Ноутбуки",
        85,
        3450000.0,
    ]
    df.loc[64, ["region", "category", "marketing_spend", "revenue"]] = [
        "Львів",
        "Навушники",
        195000.0,
        12400.0,
    ]
    df.loc[98, ["region", "category", "units_sold", "revenue"]] = [
        "Одеса",
        "Смартфони",
        62,
        1480000.0,
    ]
    return df


def _get_df() -> pd.DataFrame:
    if "df" not in st.session_state or st.session_state.df is None:
        st.session_state.df = generate_demo_dataset()
    return st.session_state.df


def get_dataset_info() -> str:
    df = _get_df()
    info = {
        "rows_count": int(len(df)),
        "columns_count": int(len(df.columns)),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
        "sample_rows": df.head(5).fillna("NULL").to_dict(orient="records"),
    }
    return json.dumps(info, ensure_ascii=False)


def get_summary_statistics(columns: str = "") -> str:
    df = _get_df()
    requested = [c.strip() for c in columns.split(",") if c.strip()]
    target_cols = requested if requested else list(df.columns)
    valid_cols = [c for c in target_cols if c in df.columns]

    if not valid_cols:
        return f"Помилка: колонок не знайдено. Доступні: {list(df.columns)}"

    stats = {}
    for col in valid_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            series = df[col].dropna()
            stats[col] = {
                "type": "numeric",
                "count": int(series.count()),
                "mean": round(float(series.mean()), 2) if len(series) else None,
                "median": round(float(series.median()), 2) if len(series) else None,
                "std": round(float(series.std()), 2) if len(series) > 1 else None,
                "min": round(float(series.min()), 2) if len(series) else None,
                "max": round(float(series.max()), 2) if len(series) else None,
            }
        else:
            top_vals = df[col].fillna("NULL").value_counts().head(5).to_dict()
            stats[col] = {
                "type": "categorical",
                "unique_count": int(df[col].nunique()),
                "top_values": {str(k): int(v) for k, v in top_vals.items()},
            }
    return json.dumps(stats, ensure_ascii=False)


def run_pandas_query(code: str) -> str:
    df = _get_df().copy()
    safe_globals = {"__builtins__": {"len": len, "sum": sum, "min": min, "max": max, "round": round}}
    local_vars = {"df": df, "pd": pd, "np": np, "result": None}

    try:
        try:
            output = eval(code, safe_globals, local_vars)
        except SyntaxError:
            exec(code, safe_globals, local_vars)
            output = local_vars.get("result")

        if output is None:
            return "Код виконано, але він повернув None. Передайте вираз або запишіть результат у змінну `result`."

        if isinstance(output, (pd.DataFrame, pd.Series)):
            return output.head(30).to_string()

        return str(output)
    except Exception as exc:
        return f"Помилка виконання Pandas-запиту: {exc}"


def detect_anomalies(column: str, z_threshold: float = 2.5) -> str:
    df = _get_df()
    if column not in df.columns:
        return f"Помилка: колонку '{column}' не знайдено. Доступні: {list(df.columns)}"

    if not pd.api.types.is_numeric_dtype(df[column]):
        return f"Помилка: колонка '{column}' не є числовою."

    series = df[column].dropna()
    std = float(series.std())
    if std == 0:
        return "Стандартне відхилення дорівнює 0, викидів немає."

    mean = float(series.mean())
    z_scores = (df[column] - mean).abs() / std
    outliers = df[z_scores > z_threshold].copy()
    outliers["z_score"] = z_scores[z_scores > z_threshold].round(2)

    result = {
        "column": column,
        "mean": round(mean, 2),
        "std": round(std, 2),
        "z_threshold": float(z_threshold),
        "anomalies_count": int(len(outliers)),
        "anomalous_rows": outliers.head(15).fillna("NULL").to_dict(orient="records"),
    }
    return json.dumps(result, ensure_ascii=False)


def build_chart(
    chart_type: str,
    x_col: str,
    y_col: str = "",
    color_col: str = "",
    agg_func: str = "none",
    title: str = "Графік аналізу даних",
) -> str:
    df = _get_df().copy()

    if x_col not in df.columns:
        return f"Помилка: колонку x_col='{x_col}' не знайдено. Доступні: {list(df.columns)}"
    if y_col and y_col not in df.columns:
        return f"Помилка: колонку y_col='{y_col}' не знайдено. Доступні: {list(df.columns)}"
    if color_col and color_col not in df.columns:
        color_col = ""

    plot_df = df
    color_arg = color_col if color_col else None

    if agg_func != "none" and y_col:
        if agg_func in ("sum", "mean", "median") and not pd.api.types.is_numeric_dtype(df[y_col]):
            return f"Помилка: для агрегації '{agg_func}' колонка '{y_col}' має бути числовою."
        group_cols = [x_col] + ([color_col] if color_col else [])
        plot_df = df.groupby(group_cols, as_index=False)[y_col].agg(agg_func)

    try:
        ctype = chart_type.lower().strip()
        if ctype == "bar":
            fig = px.bar(plot_df, x=x_col, y=y_col or None, color=color_arg, title=title, barmode="group")
        elif ctype == "line":
            plot_df = plot_df.sort_values(by=x_col)
            fig = px.line(plot_df, x=x_col, y=y_col, color=color_arg, markers=True, title=title)
        elif ctype == "scatter":
            fig = px.scatter(plot_df, x=x_col, y=y_col, color=color_arg, title=title)
        elif ctype == "box":
            fig = px.box(plot_df, x=x_col, y=y_col, color=color_arg, title=title)
        elif ctype == "histogram":
            fig = px.histogram(plot_df, x=x_col, color=color_arg, title=title)
        elif ctype == "pie":
            fig = px.pie(plot_df, names=x_col, values=y_col, title=title)
        else:
            return f"Непідтримуваний тип графіка '{chart_type}'. Оберіть: bar, line, scatter, box, histogram, pie."

        fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=50, b=20))

        if "current_turn_charts" not in st.session_state:
            st.session_state.current_turn_charts = []
        st.session_state.current_turn_charts.append(fig)

        return f"Графік '{title}' ({ctype}) успішно побудовано та відображено в інтерфейсі."
    except Exception as exc:
        return f"Помилка побудови графіка: {exc}"


ALL_TOOLS = [
    get_dataset_info,
    get_summary_statistics,
    run_pandas_query,
    detect_anomalies,
    build_chart,
]

TOOL_MAP = {fn.__name__: fn for fn in ALL_TOOLS}