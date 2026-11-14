import json
import time
import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

from tools import ALL_TOOLS, TOOL_MAP, generate_demo_dataset

SYSTEM_PROMPT = """
Ти — автономний Senior Data Analyst (ШІ-аналітик даних).
Ти допомагаєш користувачу досліджувати табличні дані, знаходити інсайти, виявляти аномалії та будувати графіки.
Відповідай чітко, структуровано й українською мовою.

Правила роботи з інструментами:
1. Якщо ти ще не знаєш точну структуру активного датасету то першим кроком виклич `get_dataset_info()`.
2. Якщо для відповіді потрібно виконати кілька незалежних дій (наприклад, перевірити аномалії, порахувати статистику та побудувати графік), викликай усі потрібні інструменти одночасно в одному кроці, а не по черзі.
3. Ніколи не вигадуй цифри чи назви колонок. Для будь-яких підрахунків і групувань викликай `run_pandas_query(...)` або `get_summary_statistics(...)`.
4. Якщо користувач просить знайти підозрілі значення чи викиди то використай `detect_anomalies(...)`.
5. Якщо користувач просить побудувати графік або візуалізувати тренд/розподіл то обов'язково виклич `build_chart(...)`.
6. Після отримання результатів одразу сформулюй зрозумілий аналітичний висновок із конкретними цифрами, не викликаючи повторно ті самі функції.
"""

st.set_page_config(
    page_title="AI Data Analyst — LLM Agent",
    layout="wide",
)

if "df" not in st.session_state:
    st.session_state.df = generate_demo_dataset()
    st.session_state.dataset_name = "demo_sales_2026.csv"

if "available_models" not in st.session_state:
    st.session_state.available_models = [
        "gemini-4-hydrogen",
        "gemini-3.8-flash",
        "gemini-3.8-flash-lite",
        "gemini-3-flash-preview",
        "gemini-2.5-flash-lite",
    ]

if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = [
        {
            "role": "assistant",
            "content": (
                "Вітаю! Я автономний AI Data Analyst. Зараз завантажено демо-датасет продажів "
                "(ви можете завантажити власний `.csv` файл у бічній панелі зліва).\n\n"
                "Запитайте мене про статистику, тренди, аномалії або попросіть побудувати графік"
            ),
            "tool_logs": [],
            "charts": [],
        }
    ]

if "gemini_history" not in st.session_state:
    st.session_state.gemini_history = []


def reset_conversation():
    st.session_state.gemini_history = []
    st.session_state.ui_messages = [
        {
            "role": "assistant",
            "content": f"Датасет {st.session_state.dataset_name} готовий до аналізу. Що потрібно дослідити?",
            "tool_logs": [],
            "charts": [],
        }
    ]

with st.sidebar:
    st.title("AI Data Analyst")

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="...",
    )

    if st.button("Завантажити моделі з API", use_container_width=True):
        if not api_key:
            st.warning("Спочатку вставте API Key вище.")
        else:
            try:
                tmp_client = genai.Client(api_key=api_key.strip())
                found = []
                for m in tmp_client.models.list():
                    name = (m.name or "").replace("models/", "")
                    low = name.lower()
                    if "gemini" in low and not any(
                        bad in low for bad in ["embedding", "imagen", "veo", "tts", "audio", "pro", "deep"]
                    ):
                        found.append(name)
                found.sort(key=lambda x: (0 if "lite" in x else 1, 0 if "flash" in x else 1, x))
                if found:
                    st.session_state.available_models = found
                    st.success(f"Знайдено швидких моделей: {len(found)}")
                else:
                    st.warning("Не знайдено текстових моделей Gemini.")
            except Exception as e:
                st.error(f"Помилка отримання моделей: {e}")

    model_name = st.selectbox(
        "Модель Gemini",
        options=st.session_state.available_models,
        index=0,
        help="Якщо одна модель перевантажена, оберіть іншу",
    )

    st.divider()
    st.subheader("Джерело даних")
    uploaded_file = st.file_uploader("Завантажте свій CSV-файл", type=["csv"])

    if uploaded_file is not None:
        if st.session_state.get("last_uploaded_name") != uploaded_file.name:
            try:
                st.session_state.df = pd.read_csv(uploaded_file, sep=None, engine="python")
                st.session_state.dataset_name = uploaded_file.name
                st.session_state.last_uploaded_name = uploaded_file.name
                reset_conversation()
                st.rerun()
            except Exception as e:
                st.error(f"Не вдалося прочитати CSV: {e}")

    if st.button("Повернути демо-датасет", use_container_width=True):
        st.session_state.df = generate_demo_dataset()
        st.session_state.dataset_name = "demo_sales_2026.csv"
        st.session_state.pop("last_uploaded_name", None)
        reset_conversation()
        st.rerun()

    st.divider()
    if st.button("Очистити історію чату", use_container_width=True):
        reset_conversation()
        st.rerun()

st.header("Автономний аналітик даних із побудовою графіків")

df = st.session_state.df
with st.expander(
    f"Перегляд таблиці: `{st.session_state.dataset_name}` ({len(df)} рядків × {len(df.columns)} колонок)",
    expanded=False,
):
    c1, c2, c3 = st.columns(3)
    c1.metric("Рядків", len(df))
    c2.metric("Колонок", len(df.columns))
    c3.metric("Пропусків (NaN)", int(df.isna().sum().sum()))
    st.dataframe(df.head(15), use_container_width=True)

st.caption("Швидкі приклади запитів:")
q_cols = st.columns(3)
quick_prompt = None
if q_cols[0].button("Виручка по регіонах + графік", use_container_width=True):
    quick_prompt = "Порахуй сумарну виручку по кожному регіону та побудуй стовпчастий графік."
if q_cols[1].button("Знайди аномалії у даних", use_container_width=True):
    quick_prompt = "Перевір датасет на статистичні аномалії у виручці та маркетингових витратах."
if q_cols[2].button("Огляд і кореляція категорій", use_container_width=True):
    quick_prompt = "Проведи загальний огляд датасету та порівняй середній чек і продажі по категоріях на графіку."

def run_agent_turn(user_text: str, key: str, model: str, status_box) -> tuple[str, list[dict], list]:
    client = genai.Client(
        api_key=key.strip(),
        http_options=types.HttpOptions(timeout=180_000),
    )
    st.session_state.current_turn_charts = []

    st.session_state.gemini_history.append(
        types.Content(role="user", parts=[types.Part.from_text(text=user_text)])
    )

    clean_model = model.strip()
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=ALL_TOOLS,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0.2,
    )

    tool_logs = []
    seen_calls = set()

    for step in range(1, 6):
        status_box.write(f"Крок {step}: звернення до `{clean_model}`...")

        response = None
        for attempt in range(1, 3):
            try:
                response = client.models.generate_content(
                    model=clean_model,
                    contents=st.session_state.gemini_history,
                    config=config,
                )
                break
            except Exception as err:
                err_str = str(err)
                if ("504" in err_str or "503" in err_str) and attempt < 2:
                    status_box.write("Таймаут сервера, повторна спроба...")
                    time.sleep(2)
                    continue
                raise

        if response.candidates and response.candidates[0].content:
            st.session_state.gemini_history.append(response.candidates[0].content)

        if not response.function_calls:
            status_box.write("Фінальну відповідь сформовано.")
            charts = list(st.session_state.current_turn_charts)
            return response.text or "Аналіз завершено.", tool_logs, charts

        tool_response_parts = []
        for fc in response.function_calls:
            fn_name = fc.name
            fn_args = dict(fc.args) if fc.args else {}
            call_sig = f"{fn_name}:{json.dumps(fn_args, sort_keys=True, ensure_ascii=False)}"

            if call_sig in seen_calls:
                result = "Цей інструмент із тими самими аргументами вже викликався. Використайте попередні дані."
            elif fn_name in TOOL_MAP:
                seen_calls.add(call_sig)
                status_box.write(f"Виклик інструменту `{fn_name}`...")
                try:
                    result = TOOL_MAP[fn_name](**fn_args)
                except Exception as exc:
                    result = f"Помилка виклику {fn_name}: {exc}"
            else:
                result = f"Невідомий інструмент: {fn_name}"

            tool_logs.append(
                {
                    "tool": fn_name,
                    "args": fn_args,
                    "result": result,
                }
            )

            tool_response_parts.append(
                types.Part.from_function_response(
                    name=fn_name,
                    response={"result": result},
                )
            )

        st.session_state.gemini_history.append(
            types.Content(role="user", parts=tool_response_parts)
        )

    charts = list(st.session_state.current_turn_charts)
    return "Досягнуто ліміт кроків аналізу (5) за один запит", tool_logs, charts

for idx, msg in enumerate(st.session_state.ui_messages):
    with st.chat_message(msg["role"]):
        if msg.get("tool_logs"):
            with st.expander(f"Виклики інструментів ({len(msg['tool_logs'])})", expanded=False):
                for log in msg["tool_logs"]:
                    st.markdown(f"Інструмент: `{log['tool']}`")
                    st.code(
                        f"Аргументи: {json.dumps(log['args'], ensure_ascii=False)}\n"
                        f"Результат: {str(log['result'])[:1200]}",
                        language="json",
                    )

        st.markdown(msg["content"])

        for c_idx, fig in enumerate(msg.get("charts", [])):
            st.plotly_chart(fig, use_container_width=True, key=f"chart_{idx}_{c_idx}")

chat_input = st.chat_input("Поставте запитання до даних (наприклад: «Побудуй динаміку виручки по днях»)...")
user_query = quick_prompt or chat_input

if user_query:
    if not api_key:
        st.warning("Введіть ваш Gemini API Key у бічній панелі зліва!")
        st.stop()

    st.session_state.ui_messages.append(
        {"role": "user", "content": user_query, "tool_logs": [], "charts": []}
    )
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.status("Агент аналізує дані...", expanded=True) as status:
            try:
                reply_text, logs, generated_charts = run_agent_turn(
                    user_query, api_key, model_name, status
                )
                status.update(label="Аналіз завершено!", state="complete", expanded=False)
                st.session_state.ui_messages.append(
                    {
                        "role": "assistant",
                        "content": reply_text,
                        "tool_logs": logs,
                        "charts": generated_charts,
                    }
                )
                st.rerun()
            except Exception as e:
                status.update(label="Помилка під час виконання", state="error", expanded=True)
                st.error(f"Помилка API: {e}")