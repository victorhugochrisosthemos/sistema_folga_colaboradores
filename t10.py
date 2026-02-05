import os
import json
import calendar
from datetime import date
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
import requests

from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode, DataReturnMode

try:
    import holidays
except Exception:
    holidays = None


# ----------------------------
# Constantes
# ----------------------------
MESES_PT = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril", 5: "Maio", 6: "Junho",
    7: "Julho", 8: "Agosto", 9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
}
DIAS_SEMANA_PT = ["SEG", "TER", "QUA", "QUI", "SEX", "SÁB", "DOM"]  # weekday(): 0..6

APP_DIR = Path(__file__).resolve().parent
SHIFTS_JSON = APP_DIR / "shifts.json"
EMPLOYEES_JSON = APP_DIR / "employees.json"
RULES_JSON = APP_DIR / "rules.json"

# ----------------------------
# Feriados 2026 (base ANBIMA)
# ----------------------------
FERIADOS_2026_BR = {
    "2026-01-01": "Confraternização Universal",
    "2026-02-16": "Carnaval",
    "2026-02-17": "Carnaval",
    "2026-04-03": "Paixão de Cristo",
    "2026-04-21": "Tiradentes",
    "2026-05-01": "Dia do Trabalho",
    "2026-06-04": "Corpus Christi",
    "2026-09-07": "Independência do Brasil",
    "2026-10-12": "Nossa Sra.ª Aparecida - Padroeira do Brasil",
    "2026-11-02": "Finados",
    "2026-11-15": "Proclamação da República",
    "2026-11-20": "Dia Nacional de Zumbi e da Consciência Negra",
    "2026-12-25": "Natal",
}

# Modelos comuns na Groq (OpenAI-compatible)
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]


# ----------------------------
# JSON helpers
# ----------------------------
def load_json(path: Path, default):
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def save_json(path: Path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def persist_turnos_and_employees(turnos: dict):
    save_json(SHIFTS_JSON, {"turnos": turnos})
    employees = []
    for turno, pessoas in turnos.items():
        for p in pessoas:
            employees.append({"name": p, "turno": turno})
    save_json(EMPLOYEES_JSON, {"employees": employees})


def persist_rules(tags: list):
    save_json(RULES_JSON, {"tags": tags})


# ----------------------------
# Data helpers
# ----------------------------
def month_days(year: int, month: int):
    last_day = calendar.monthrange(year, month)[1]
    return [date(year, month, d) for d in range(1, last_day + 1)]


def prev_month_year(year: int, month: int):
    if month == 1:
        return year - 1, 12
    return year, month - 1


def days_by_weekday_map(year: int, month: int):
    days = month_days(year, month)
    out = {k: [] for k in DIAS_SEMANA_PT}
    for d in days:
        out[DIAS_SEMANA_PT[d.weekday()]].append(d.day)
    return out


def get_holidays_for_month(year: int, month: int):
    """
    Prioridade:
      1) Se for 2026, usa o dicionário fixo (ANBIMA).
      2) Se não for 2026, tenta usar `holidays` se disponível.
    Retorna: [(date, "Nome"), ...]
    """
    out = []

    if year == 2026:
        for iso, name in FERIADOS_2026_BR.items():
            d = date.fromisoformat(iso)
            if d.month == month:
                out.append((d, name))
        out.sort(key=lambda x: x[0])
        return out

    if holidays is None:
        return []

    try:
        br_holidays = holidays.country_holidays("BR", years=[year])
        for day, name in br_holidays.items():
            if isinstance(day, date) and day.year == year and day.month == month:
                out.append((day, str(name)))
        out.sort(key=lambda x: x[0])
        return out
    except Exception:
        return []


# ----------------------------
# Schedule DF
# ----------------------------
def build_schedule_df(year: int, month: int, turnos: dict):
    days = month_days(year, month)
    day_cols = [d.strftime("%d") for d in days]

    rows = []
    for turno, pessoas in turnos.items():
        header_row = {"Turno": turno, "Colaborador": "", "_tipo": "TURN_HEADER"}
        for c in day_cols:
            header_row[c] = ""
        rows.append(header_row)

        for p in pessoas:
            row = {"Turno": turno, "Colaborador": p, "_tipo": "NORMAL"}
            for c in day_cols:
                row[c] = "T"
            rows.append(row)

    df = pd.DataFrame(rows)

    weekday_row = {"Turno": "", "Colaborador": "DIA", "_tipo": "WEEKDAY"}
    for d in days:
        weekday_row[d.strftime("%d")] = DIAS_SEMANA_PT[d.weekday()]

    return df, weekday_row, day_cols


def ensure_month_shape(df: pd.DataFrame, year: int, month: int, turnos: dict):
    new_df, new_weekday_row, day_cols = build_schedule_df(year, month, turnos)

    if df is None or df.empty:
        return new_df, new_weekday_row, day_cols

    base_cols = ["Turno", "Colaborador", "_tipo"]
    for c in base_cols:
        if c not in df.columns:
            df[c] = ""

    for c in day_cols:
        if c not in df.columns:
            df[c] = pd.NA

    keep_cols = base_cols + day_cols
    df = df[[c for c in df.columns if c in keep_cols]]

    for c in day_cols:
        mask_normal = (df["_tipo"] == "NORMAL")
        mask_header = (df["_tipo"] == "TURN_HEADER")
        mask_weekday = (df["_tipo"] == "WEEKDAY")

        empty_normal = df[c].isna() | (df[c].astype(str).str.strip() == "") | (df[c].astype(str).str.lower() == "none")
        df.loc[mask_normal & empty_normal, c] = "T"

        empty_header = df[c].isna() | (df[c].astype(str).str.lower() == "none")
        df.loc[mask_header & empty_header, c] = ""

        if mask_weekday.any():
            df.loc[mask_weekday, c] = new_weekday_row[c]

        invalid_normal = mask_normal & ~df[c].isin(["T", "F"])
        df.loc[invalid_normal, c] = "T"

    df = df[base_cols + day_cols]
    return df, new_weekday_row, day_cols


# ----------------------------
# PDF
# ----------------------------
def make_pdf(schedule_df: pd.DataFrame, weekday_row: dict, year: int, month: int, tags: list, feriados: list):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=18,
        rightMargin=18,
        topMargin=18,
        bottomMargin=18
    )

    styles = getSampleStyleSheet()
    story = []

    titulo = f"Escala 6x1 - {MESES_PT.get(month, str(month))}/{year}"
    story.append(Paragraph(titulo, styles["Title"]))
    story.append(Spacer(1, 8))

    if schedule_df is None or schedule_df.empty:
        story.append(Paragraph("Sem dados (adicione turnos e colaboradores).", styles["Normal"]))
        doc.build(story)
        buffer.seek(0)
        return buffer

    day_cols = [c for c in schedule_df.columns if c.isdigit()]
    header = ["Colaborador"] + day_cols
    data = [header]

    for _, r in schedule_df.iterrows():
        if r.get("_tipo") == "TURN_HEADER":
            data.append([r["Turno"]] + [""] * len(day_cols))
        else:
            data.append([r["Colaborador"]] + [r[c] for c in day_cols])

    data.append(["DIA"] + [weekday_row.get(c, "") for c in day_cols])

    page_w, _ = landscape(A4)
    usable_w = page_w - doc.leftMargin - doc.rightMargin

    n_days = len(day_cols)
    colab_w = 130
    remaining = max(usable_w - colab_w, 50)
    day_w = remaining / max(n_days, 1)
    col_widths = [colab_w] + [day_w] * n_days

    table = Table(data, colWidths=col_widths, repeatRows=1)

    base_style = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),

        ("BACKGROUND", (0, 0), (-1, 0), colors.white),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

        ("ALIGN", (0, 1), (0, -1), "LEFT"),
    ]

    for r in range(1, len(data) - 1):
        is_turn_header = (data[r][0] != "" and data[r][1:] == [""] * n_days and data[r][0] != "DIA")
        if is_turn_header:
            base_style.append(("SPAN", (1, r), (n_days, r)))
            base_style.append(("BACKGROUND", (0, r), (n_days, r), colors.whitesmoke))
            base_style.append(("FONTNAME", (0, r), (n_days, r), "Helvetica-Bold"))
            base_style.append(("ALIGN", (0, r), (n_days, r), "LEFT"))
        else:
            for c in range(1, 1 + n_days):
                val = data[r][c]
                if val == "T":
                    base_style.append(("BACKGROUND", (c, r), (c, r), colors.lightgreen))
                elif val == "F":
                    base_style.append(("BACKGROUND", (c, r), (c, r), colors.lightblue))

    last_r = len(data) - 1
    base_style += [
        ("BACKGROUND", (0, last_r), (-1, last_r), colors.white),
        ("FONTNAME", (0, last_r), (-1, last_r), "Helvetica-Bold"),
        ("FONTSIZE", (0, last_r), (-1, last_r), 7),
    ]

    table.setStyle(TableStyle(base_style))
    story.append(table)
    story.append(Spacer(1, 10))

    # ✅ Restrições em LISTA (uma por linha)
    if tags:
        restr_html = "<b>Restrições:</b><br/>" + "<br/>".join([f"• {t}" for t in tags])
        story.append(Paragraph(restr_html, styles["Normal"]))
        story.append(Spacer(1, 6))
    else:
        story.append(Paragraph("<b>Restrições:</b> (nenhuma)", styles["Normal"]))
        story.append(Spacer(1, 6))

    if feriados:
        fer_html = "<b>Feriados do mês:</b><br/>" + "<br/>".join([f"• {d.strftime('%d/%m')}: {name}" for d, name in feriados])
        story.append(Paragraph(fer_html, styles["Normal"]))
    else:
        story.append(Paragraph("<b>Feriados do mês:</b> (não informado)", styles["Normal"]))

    doc.build(story)
    buffer.seek(0)
    return buffer


# ----------------------------
# Groq (LLM)
# ----------------------------
def groq_chat(api_key: str, model: str, messages: list, max_tokens: int = 120, temperature: float = 0.2) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "temperature": temperature, "max_tokens": max_tokens, "messages": messages}

    r = requests.post(url, headers=headers, json=payload, timeout=30)

    if not r.ok:
        try:
            err = r.json()
        except Exception:
            err = r.text
        raise RuntimeError(f"Groq API erro {r.status_code}: {err}")

    data = r.json()
    return (data["choices"][0]["message"]["content"] or "").strip()


def groq_suggest_short(prompt: str, api_key: str, model: str) -> str:
    return groq_chat(
        api_key=api_key,
        model=model,
        max_tokens=140,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "Você é um assistente de escala 6x1. Responda em PT-BR com uma sugestão curta (1–3 linhas). "
                    "Objetivo: transformar a consideração em ação prática (ex.: quais dias marcar F, quantos domingos existem, etc.). "
                    "Use SOMENTE o contexto (mapa de dias por semana e feriados). "
                    "NÃO diga que é redundante, a menos que a consideração seja exatamente igual a uma já registrada. "
                    "Se estiver ambígua, faça 1 pergunta curta."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )


def build_llm_context(year_i: int, month_i: int, prev_tags: list) -> str:
    feriados_mes = get_holidays_for_month(year_i, month_i)
    dias_mes_atual = calendar.monthrange(year_i, month_i)[1]
    y_prev, m_prev = prev_month_year(year_i, month_i)
    dias_mes_passado = calendar.monthrange(y_prev, m_prev)[1]

    folgas_prev_txt = f"{(dias_mes_passado/7):.2f}".replace(".", ",")
    folgas_atual_txt = f"{(dias_mes_atual/7):.2f}".replace(".", ",")

    mapa = days_by_weekday_map(year_i, month_i)

    fer_str = "(não informado)"
    if feriados_mes:
        fer_str = " | ".join([f"{d.strftime('%d/%m/%Y')}: {name}" for d, name in feriados_mes])

    mapa_str = "\n".join([f"{k}: {', '.join(str(n) for n in mapa[k])}" for k in DIAS_SEMANA_PT])

    shifts = load_json(SHIFTS_JSON, {"turnos": {}})
    employees = load_json(EMPLOYEES_JSON, {"employees": []})

    turnos_nomes = list((shifts.get("turnos") or {}).keys())
    employees_nomes = [e.get("name") for e in (employees.get("employees") or []) if isinstance(e, dict)][:30]

    ctx = f"""
MÊS/ANO: {MESES_PT.get(month_i, str(month_i))}/{year_i}
FERIADOS DO MÊS: {fer_str}

MÊS PASSADO: ({dias_mes_passado}/7) = {folgas_prev_txt} folgas (referência)
MÊS ATUAL: ({dias_mes_atual}/7) = {folgas_atual_txt} folgas por colaborador (referência)

DIAS DO MÊS POR DIA DA SEMANA:
{mapa_str}

TURNOS CADASTRADOS: {', '.join(turnos_nomes) if turnos_nomes else '(nenhum)'}
COLABORADORES (amostra): {', '.join([n for n in employees_nomes if n]) if employees_nomes else '(nenhum)'}

CONSIDERAÇÕES JÁ REGISTRADAS:
- """ + ("\n- ".join(prev_tags) if prev_tags else "(nenhuma)") + "\n"
    return ctx.strip()


def groq_test_minimal(api_key: str, model: str) -> str:
    return groq_chat(
        api_key=api_key,
        model=model,
        max_tokens=20,
        temperature=0.0,
        messages=[{"role": "user", "content": "Responda apenas: OK"}],
    )


def normalize_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="Escala 6x1", layout="wide")
st.title("Sistema de Folga de Escala 6x1")

st.markdown("""
<style>
.ag-theme-alpine .ag-cell-value,
.ag-theme-alpine .ag-cell,
.ag-theme-alpine .ag-group-value,
.ag-theme-alpine .ag-group-child-count {
  overflow: visible !important;
  text-overflow: clip !important;
  white-space: nowrap !important;
}
.ag-theme-alpine .ag-header-cell-label { justify-content: center !important; }
.ag-theme-alpine .ag-cell { padding-left: 8px !important; padding-right: 8px !important; }
</style>
""", unsafe_allow_html=True)

# Estado inicial
if "turnos" not in st.session_state:
    saved = load_json(SHIFTS_JSON, {"turnos": {}})
    st.session_state.turnos = saved.get("turnos", {}) if isinstance(saved, dict) else {}
    if not isinstance(st.session_state.turnos, dict):
        st.session_state.turnos = {}

if "tags" not in st.session_state:
    saved_rules = load_json(RULES_JSON, {"tags": []})
    st.session_state.tags = saved_rules.get("tags", []) if isinstance(saved_rules, dict) else []
    if not isinstance(st.session_state.tags, list):
        st.session_state.tags = []

if "schedule_df" not in st.session_state:
    st.session_state.schedule_df = None
if "weekday_row" not in st.session_state:
    st.session_state.weekday_row = None
if "active_year" not in st.session_state:
    st.session_state.active_year = None
if "active_month" not in st.session_state:
    st.session_state.active_month = None

if "ai_suggestions" not in st.session_state:
    st.session_state.ai_suggestions = {}  # {tag: suggestion}
if "groq_model" not in st.session_state:
    st.session_state.groq_model = GROQ_MODELS[0]


# Sidebar
with st.sidebar:
    st.header("Configurações")
    c1, c2 = st.columns(2)
    with c1:
        year = st.number_input("Ano", min_value=2020, max_value=2100, value=date.today().year, step=1)
    with c2:
        month = st.number_input("Mês", min_value=1, max_value=12, value=date.today().month, step=1)

    st.divider()
    st.subheader("IA (Groq)")

    st.session_state.groq_model = st.selectbox("Modelo Groq", GROQ_MODELS, index=0)

    api_key_env = os.getenv("GROQ_API_KEY", "").strip()
    st.caption(f"GROQ_API_KEY detectada: {'SIM' if bool(api_key_env) else 'NÃO'}")

    if st.button("Testar GROQ agora"):
        if not api_key_env:
            st.error("Não encontrei GROQ_API_KEY no ambiente. Defina nas Secrets (Streamlit Cloud) ou no terminal e reinicie.")
        else:
            try:
                resp = groq_test_minimal(api_key_env, st.session_state.groq_model)
                st.success(f"Teste OK. Resposta: {resp}")
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.subheader("Turnos e Colaboradores")

    new_turno = st.text_input("Novo turno", placeholder="Ex: Manhã")
    if st.button("Adicionar turno"):
        t = new_turno.strip()
        if t and t not in st.session_state.turnos:
            st.session_state.turnos[t] = []
            persist_turnos_and_employees(st.session_state.turnos)
            st.success(f"Turno '{t}' adicionado.")
        else:
            st.warning("Turno vazio ou já existe.")

    if st.session_state.turnos:
        turno_sel = st.selectbox("Selecione um turno", list(st.session_state.turnos.keys()))
        new_person = st.text_input("Adicionar colaborador", placeholder="Nome do colaborador")
        if st.button("Adicionar colaborador ao turno"):
            p = new_person.strip()
            if p:
                if p not in st.session_state.turnos[turno_sel]:
                    st.session_state.turnos[turno_sel].append(p)
                    persist_turnos_and_employees(st.session_state.turnos)
                    st.success(f"'{p}' adicionado em {turno_sel}.")
                else:
                    st.warning("Esse colaborador já está nesse turno.")
            else:
                st.warning("Nome vazio.")

        if st.session_state.turnos[turno_sel]:
            rem = st.selectbox("Remover colaborador", ["(nenhum)"] + st.session_state.turnos[turno_sel])
            if st.button("Remover selecionado"):
                if rem != "(nenhum)":
                    st.session_state.turnos[turno_sel].remove(rem)
                    persist_turnos_and_employees(st.session_state.turnos)
                    st.success(f"'{rem}' removido de {turno_sel}.")
        else:
            st.caption("Sem colaboradores neste turno.")
    else:
        st.info("Nenhum turno ainda. Adicione um turno para começar.")

    st.divider()
    if st.button("Gerar / Regerar planilha do mês"):
        df, wrow, _ = build_schedule_df(int(year), int(month), st.session_state.turnos)
        st.session_state.schedule_df = df
        st.session_state.weekday_row = wrow
        st.session_state.active_year = int(year)
        st.session_state.active_month = int(month)
        st.success("Planilha gerada.")


# Ajuste automático ao mudar mês/ano
year_i = int(year)
month_i = int(month)

if st.session_state.active_year != year_i or st.session_state.active_month != month_i:
    df_current, wrow_current, _ = ensure_month_shape(
        st.session_state.schedule_df,
        year_i,
        month_i,
        st.session_state.turnos
    )
    st.session_state.schedule_df = df_current
    st.session_state.weekday_row = wrow_current
    st.session_state.active_year = year_i
    st.session_state.active_month = month_i

schedule_df = st.session_state.schedule_df
weekday_row = st.session_state.weekday_row

st.subheader(f"{MESES_PT.get(month_i, str(month_i))} / {year_i}")

# ----------------------------
# Grid (AgGrid)
# ----------------------------
if schedule_df is None or schedule_df.empty:
    st.warning("Adicione turnos e colaboradores no menu à esquerda para a planilha aparecer.")
else:
    day_cols = [c for c in schedule_df.columns if c.isdigit()]
    first_day = day_cols[0] if day_cols else None
    n_days = len(day_cols)

    cell_style_all = JsCode("""
    function(params) {
        if (params.data && params.data._tipo === 'WEEKDAY') {
            return { fontSize: '12px', fontWeight: '900', textAlign: 'center', backgroundColor: '#ffffff' };
        }
        if (params.data && params.data._tipo === 'TURN_HEADER') {
            return { backgroundColor: '#f2f2f2', fontWeight: '900', textAlign: 'left', paddingLeft: '10px' };
        }
        if (params.value === 'T') {
            return { backgroundColor: '#a7f3a7', fontWeight: 'bold', textAlign: 'center' };
        }
        if (params.value === 'F') {
            return { backgroundColor: '#a7c7ff', fontWeight: 'bold', textAlign: 'center' };
        }
        return { textAlign: 'center' };
    }
    """)

    on_cell_clicked = JsCode("""
    function(e) {
        const colId = e.column.getId();
        const isDay = /^[0-9]{2}$/.test(colId);
        const isHeader = (e.data && e.data._tipo === 'TURN_HEADER');
        const isDiaRow = (e.data && e.data._tipo === 'WEEKDAY');
        if (!isDay || isHeader || isDiaRow) return;

        const current = e.node.data[colId];
        const next = (current === 'T') ? 'F' : 'T';
        e.node.setDataValue(colId, next);
    }
    """)

    gb = GridOptionsBuilder.from_dataframe(schedule_df)
    gb.configure_column("_tipo", hide=True)
    gb.configure_column("Turno", hide=True)
    gb.configure_column("Colaborador", pinned="left", width=200, minWidth=140)

    for c in day_cols:
        gb.configure_column(
            c,
            width=70,
            minWidth=66,
            editable=False,
            cellStyle=cell_style_all,
        )

    rows_count = len(schedule_df) + 1
    row_h = 34
    header_h = 34
    pinned_h = 34
    extra = 28
    grid_h = header_h + rows_count * row_h + pinned_h + extra
    grid_h = max(220, min(520, grid_h))

    gb.configure_grid_options(
        onCellClicked=on_cell_clicked,
        suppressRowClickSelection=True,
        rowSelection="single",
        domLayout="normal",
        rowHeight=row_h,
        headerHeight=header_h,
        suppressColumnVirtualisation=True,
    )

    grid_options = gb.build()
    grid_options["defaultColDef"] = {"resizable": True}
    grid_options["pinnedBottomRowData"] = [weekday_row]

    if first_day is not None and n_days > 0:
        colspan_js = JsCode(f"""
        function(params) {{
          if (params.data && params.data._tipo === "TURN_HEADER") {{
            return {n_days};
          }}
          return 1;
        }}
        """)

        turno_text_js = JsCode("""
        function(params) {
          if (params.data && params.data._tipo === "TURN_HEADER") {
            return params.data.Turno;
          }
          return params.value;
        }
        """)

        hide_if_header_js = JsCode("""
        function(params) {
          if (params.data && params.data._tipo === "TURN_HEADER") return "";
          return params.data[params.colDef.field];
        }
        """)

        for col in grid_options["columnDefs"]:
            if col.get("field") == first_day:
                col["colSpan"] = colspan_js
                col["cellRenderer"] = turno_text_js
                col["cellStyle"] = cell_style_all

        for col in grid_options["columnDefs"]:
            if col.get("field") in day_cols and col.get("field") != first_day:
                col["valueGetter"] = hide_if_header_js
                col["cellStyle"] = cell_style_all

    st.caption("Clique em uma célula do dia para alternar T/F. Você pode redimensionar colunas arrastando o cabeçalho.")
    grid_response = AgGrid(
        schedule_df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.AS_INPUT,
        reload_data=False,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=False,
        height=grid_h,
        theme="alpine",
        key="escala_grid",
    )

    updated = pd.DataFrame(grid_response["data"])
    updated, weekday_row, _ = ensure_month_shape(updated, year_i, month_i, st.session_state.turnos)
    st.session_state.schedule_df = updated
    st.session_state.weekday_row = weekday_row


# ----------------------------
# Considerações do mês
# ----------------------------
st.markdown("---")
st.markdown("<h3 style='text-align:center;'>Considerações do mês</h3>", unsafe_allow_html=True)

left_sp, center, right_sp = st.columns([0.2, 9.6, 0.2])
with center:
    feriados_mes = get_holidays_for_month(year_i, month_i)
    if feriados_mes:
        st.write("**Feriados do mês:**")
        st.write(" • " + " | ".join([f"{d.strftime('%d/%m')}: {name}" for d, name in feriados_mes]))
    else:
        st.write("**Feriados do mês:** (não informado)")

    dias_mes_atual = calendar.monthrange(year_i, month_i)[1]
    y_prev, m_prev = prev_month_year(year_i, month_i)
    dias_mes_passado = calendar.monthrange(y_prev, m_prev)[1]

    folgas_prev_txt = f"{(dias_mes_passado / 7):.2f}".replace(".", ",")
    folgas_atual_txt = f"{(dias_mes_atual / 7):.2f}".replace(".", ",")

    st.write(f"**Mês passado tiveram resultado ({dias_mes_passado} / 7) = {folgas_prev_txt} dias de folga**")
    st.write(f"**Nesse mês cada colaborador deve ter resultado ({dias_mes_atual} / 7) = {folgas_atual_txt} dias de folga**")

    st.write("**Dias do mês por dia da semana:**")
    mapa = days_by_weekday_map(year_i, month_i)
    for dia_sem in DIAS_SEMANA_PT:
        nums = ", ".join(str(n) for n in mapa[dia_sem])
        st.write(f"- **{dia_sem}**: {nums}")


# ----------------------------
# Considerações (com sugestão da IA)
# ----------------------------
st.markdown("---")
st.markdown("<h3 style='text-align:center;'>Considerações</h3>", unsafe_allow_html=True)

left_sp, center, right_sp = st.columns([0.2, 9.6, 0.2])
with center:
    new_tag = st.text_input(
        "Adicionar consideração",
        placeholder="Ex: Maria folga 2 domingos no mês",
        label_visibility="collapsed"
    )
    add_tag = st.button("Adicionar Consideração", use_container_width=True)

    if add_tag:
        t = new_tag.strip()
        if t:
            norm_t = normalize_text(t)
            existing_norm = [normalize_text(x) for x in st.session_state.tags]

            # ✅ duplicata REAL (igual)
            if norm_t in existing_norm:
                st.session_state.ai_suggestions[t] = "Essa consideração já existe exatamente igual."
            else:
                # adiciona e salva
                st.session_state.tags.append(t)
                persist_rules(st.session_state.tags)

                api_key = os.getenv("GROQ_API_KEY", "").strip()
                if api_key:
                    try:
                        prev_tags = st.session_state.tags[:-1]  # ✅ NÃO inclui a nova
                        ctx = build_llm_context(year_i, month_i, prev_tags)

                        prompt = (
                            f"{ctx}\n\n"
                            f"NOVA CONSIDERAÇÃO: {t}\n\n"
                            "Sugira como aplicar isso na escala deste mês usando os dias corretos (ex.: DOM = datas listadas). "
                            "Se envolver 'domingo(s)', cite as datas de DOM do mês. "
                            "Resposta curta."
                        )

                        model = st.session_state.groq_model
                        sug = groq_suggest_short(prompt, api_key=api_key, model=model)
                        st.session_state.ai_suggestions[t] = sug if sug else "(sem sugestão)"
                    except Exception as e:
                        st.session_state.ai_suggestions[t] = f"(Não consegui gerar sugestão agora: {e})"
                else:
                    st.session_state.ai_suggestions[t] = "(Defina GROQ_API_KEY para ver sugestão da IA.)"

            st.rerun()

    if st.session_state.tags:
        api_key_exists = bool(os.getenv("GROQ_API_KEY", "").strip())
        if not api_key_exists:
            st.info("Para ativar sugestões da IA: defina a variável de ambiente GROQ_API_KEY e reinicie o app.")

        st.write("**Considerações feitas:**")
        for i, t in enumerate(list(st.session_state.tags)):
            box = st.container()
            with box:
                cA, cB = st.columns([10, 2])
                with cA:
                    st.write(f"• {t}")
                with cB:
                    if st.button("Remover", key=f"rm_{i}"):
                        st.session_state.tags.remove(t)
                        persist_rules(st.session_state.tags)
                        st.session_state.ai_suggestions.pop(t, None)
                        st.rerun()

                sug = st.session_state.ai_suggestions.get(t)
                if sug:
                    st.caption(f"**Sugestão da LLM:** {sug}")
                else:
                    st.caption("**Sugestão da LLM:** (ainda não gerada)")
    else:
        st.caption("Nenhuma consideração adicionada ainda.")


# ----------------------------
# Export PDF
# ----------------------------
st.markdown("---")
st.subheader("Exportar")

feriados = get_holidays_for_month(year_i, month_i)

pdf_buffer = make_pdf(
    st.session_state.schedule_df,
    st.session_state.weekday_row if st.session_state.weekday_row is not None else {},
    year_i,
    month_i,
    st.session_state.tags,
    feriados
)

st.download_button(
    "Baixar em PDF",
    data=pdf_buffer.getvalue(),
    file_name=f"escala_{month_i:02d}_{year_i}.pdf",
    mime="application/pdf"
)

if holidays is None and year_i != 2026:
    st.info("Obs.: o pacote 'holidays' não está disponível; feriados automáticos só estão garantidos para 2026 (dicionário fixo).")
