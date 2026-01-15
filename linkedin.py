import os
import time
import json
import re
import sqlite3
import io
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Linkedin - Cogna", layout="wide")

ENDPOINT = "https://scraperapi.thordata.com/request"

DB_PATH = Path("./db/sourcing_profiles.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

USER_AGENT = "Mozilla/5.0 (compatible; ThordataBot/1.0)"
PER_REQUEST_DELAY = 0.5

BRAZIL_STATE_NAMES = [
    "acre","alagoas","amapá","amapa","amazonas","bahia","ceará","ceara","distrito federal",
    "espírito santo","espirito santo","goiás","goias","maranhão","maranhao","mato grosso",
    "mato grosso do sul","minas gerais","pará","para","paraíba","paraiba","paraná","parana",
    "pernambuco","piauí","piaui","rio de janeiro","rio grande do norte","rio grande do sul",
    "rondônia","rondonia","roraima","santa catarina","são paulo","sao paulo","sergipe","tocantins"
]
BRAZIL_STATE_ABBR = ["AC","AL","AP","AM","BA","CE","DF","ES","GO","MA","MT","MS","MG","PA","PB","PR","PE","PI","RJ","RN","RS","RO","RR","SC","SP","SE","TO"]

_state_names_re = r"(?:%s)" % "|".join([re.escape(s) for s in BRAZIL_STATE_NAMES])
_state_abbr_re = r"(?:%s)" % "|".join([re.escape(s) for s in BRAZIL_STATE_ABBR])

_LOCATION_SEP = r"(?:,|\u2022|\u00B7|\||–|—|-)"

# Aplica heurísticas regex para extrair o local de moradia

def extract_residence_from_description(text: Optional[str]) -> Optional[str]:
    if not text or not isinstance(text, str):
        return None
    txt = text.strip()
    norm = re.sub(r"[\u2022\u00B7\|–—\-]", ",", txt)
    norm = re.sub(r"\s+", " ", norm)

    pattern_start = re.compile(
        rf"^\s*([A-ZÀ-Ý][\wÀ-ÿ\.\- ]{{1,80}}){_LOCATION_SEP}\s*([A-Za-zÀ-ÿ ]{{2,40}})\s*(?:{_LOCATION_SEP}\s*(Brasil|Brazil))?",
        flags=re.IGNORECASE,
    )
    m = pattern_start.search(norm)
    if m:
        city = m.group(1).strip(" ,–—-")
        state_candidate = m.group(2).strip(" ,–—-")
        sc = state_candidate.lower()
        if sc in BRAZIL_STATE_NAMES or state_candidate.upper() in BRAZIL_STATE_ABBR or len(state_candidate) <= 4:
            location = city
            if state_candidate:
                location = f"{city}, {state_candidate}"
            country = m.group(3)
            if country:
                location = f"{location}, {country}"
            return location

    pattern_any = re.compile(
        rf"([A-ZÀ-Ý][\wÀ-ÿ\.\- ]{{1,80}}){_LOCATION_SEP}\s*({_state_names_re}|{_state_abbr_re})\b",
        flags=re.IGNORECASE,
    )
    m2 = pattern_any.search(norm)
    if m2:
        city = m2.group(1).strip(" ,")
        state_candidate = m2.group(2).strip(" ,")
        return f"{city}, {state_candidate}"

    pattern_city_country = re.compile(r"([A-ZÀ-Ý][\wÀ-ÿ\.\- ]{1,80})\s*,\s*(Brasil|Brazil|Portugal|Espanha|Argentina|Chile)\b", flags=re.IGNORECASE)
    m3 = pattern_city_country.search(norm)
    if m3:
        return f"{m3.group(1).strip()}, {m3.group(2).strip()}"

    bad_kw = ["desenvolvedor","engenheiro","analista","professor","estudante","bacharel","consultor","experiência","atuação","especialista","colega","graduado","estágio","graduada","sênior","junior","manager","founder","owner","cto","ceo","cfo","co-founder","aluno","estudou","curso","formado"]
    first_segment = norm.split(",")[0].lower()
    for kw in bad_kw:
        if kw in first_segment:
            return None

    m4 = re.search(r"\b([A-ZÀ-Ý][a-zà-ÿ]+(?:\s+[A-ZÀ-Ý][a-zà-ÿ]+){0,2})\b", norm)
    if m4:
        candidate = m4.group(1).strip()
        if len(candidate.split()) <= 3 and len(candidate) <= 40:
            return candidate

    return None

# Inicializa a conexão com o banco de dados SQLite
def init_db():
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sourcing_profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nome TEXT,
        profile_link TEXT,
        local_desc TEXT,
        created_at TEXT
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_profile_link ON sourcing_profiles(profile_link);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_local_desc ON sourcing_profiles(local_desc);")
    conn.commit()
    return conn

_conn_for_app = init_db()

#Salva novos perfis, ignorando links de perfis duplicados
def save_profiles_to_db(df: pd.DataFrame, conn: sqlite3.Connection) -> Tuple[int, int]:
    if df is None or df.empty:
        return 0, 0

    inserted = 0
    ignored = 0
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    with conn:
        for _, r in df.iterrows():
            nome = (r.get("nome") or "").strip()
            link = (r.get("Link de perfil") or "").strip()
            local_desc = (r.get("Local e descrição") or "").strip()

            if link:
                cur.execute("SELECT 1 FROM sourcing_profiles WHERE profile_link = ?", (link,))
                if cur.fetchone():
                    ignored += 1
                    continue

            cur.execute(
                "INSERT INTO sourcing_profiles (nome, profile_link, local_desc, created_at) VALUES (?, ?, ?, ?)",
                (nome or None, link or None, local_desc or None, now)
            )
            inserted += 1
    return inserted, ignored


# Busca todos os registros do banco de dados para fins de exportação
def fetch_all_profiles(conn: sqlite3.Connection) -> pd.DataFrame:
    cur = conn.cursor()
    cur.execute("SELECT id, nome, profile_link, local_desc, created_at FROM sourcing_profiles ORDER BY id DESC")
    rows = cur.fetchall()
    cols = ["id", "nome", "Link de perfil", "Local e descrição", "created_at"]
    return pd.DataFrame(rows, columns=cols)


# Executa consultas filtradas no banco de dados, buscando registros por localização e/ou competência
def query_profiles(conn: sqlite3.Connection, location: str = "", competence: str = "") -> pd.DataFrame:
    cur = conn.cursor()
    sql = "SELECT id, nome, profile_link, local_desc, created_at FROM sourcing_profiles WHERE 1=1"
    params: List[str] = []
    if location and location.strip():
        sql += " AND lower(local_desc) LIKE ?"
        params.append(f"%{location.strip().lower()}%")
    if competence and competence.strip():
        sql += " AND lower(local_desc) LIKE ?"
        params.append(f"%{competence.strip().lower()}%")
    sql += " ORDER BY id DESC"
    cur.execute(sql, params)
    rows = cur.fetchall()
    cols = ["id", "nome", "Link de perfil", "Local e descrição", "created_at"]
    return pd.DataFrame(rows, columns=cols)

# Tenta extrair o nome do perfil
def extract_name_from_linkedin_title(title: Optional[str]) -> Optional[str]:
    if not title or not isinstance(title, str):
        return None
    s = title.split("|")[0]
    s = s.split(" - ")[0].split(" — ")[0].strip()
    return s if s else None

#Localiza e retorna a lista de resultados
def safe_json_load(obj: Any) -> Tuple[Optional[Dict], Optional[str]]:
    if isinstance(obj, dict):
        return obj, None
    if isinstance(obj, str):
        try:
            parsed = json.loads(obj)
            if isinstance(parsed, dict):
                return parsed, None
            return None, "JSON carregado não é um objeto dict (esperado)."
        except Exception as e:
            return None, f"Erro ao desserializar string JSON: {e}"
    return None, "Tipo de objeto inesperado; esperava dict ou str contendo JSON."

def find_organic_list(resp: Dict) -> List[Dict]:
    if not isinstance(resp, dict):
        return []
    for key in ("organic", "organic_results", "results", "items"):
        v = resp.get(key)
        if isinstance(v, list):
            return v
    v = resp.get("data")
    if isinstance(v, list):
        return v
    return []

# Normaliza um único resultado de busca
def normalize_item_for_table(item: Dict) -> Dict[str, Optional[str]]:
    title = (item.get("title") or item.get("job_title") or "") if isinstance(item, dict) else ""
    link = item.get("link") or item.get("url") or item.get("source_url") or item.get("final_url") or None
    loc_fields = ["location", "place", "displayed_location", "city", "region", "locale", "area"]
    loc_candidate = None
    for f in loc_fields:
        v = item.get(f)
        if isinstance(v, str) and v.strip():
            loc_candidate = v.strip()
            break

    description = (item.get("description") or item.get("snippet") or "") if isinstance(item, dict) else ""
    extracted_residence = extract_residence_from_description(description)

    chosen_local = None
    if loc_candidate:
        if "," in loc_candidate or any(s.lower() in loc_candidate.lower() for s in ["brasil","brazil","portugal"]):
            chosen_local = loc_candidate
        else:
            maybe = extract_residence_from_description(loc_candidate)
            chosen_local = maybe or loc_candidate

    if not chosen_local and extracted_residence:
        chosen_local = extracted_residence

    local_desc = chosen_local if chosen_local else (description or "")

    name = extract_name_from_linkedin_title(title)
    if not name:
        source = item.get("source") or ""
        if isinstance(source, str) and "LinkedIn" in source:
            parts = re.split(r"·|-", source)
            if parts:
                candidate = parts[-1].strip()
                if candidate and len(candidate) > 1:
                    name = candidate
    if not name:
        m = re.search(r'\b([A-ZÀ-Ÿ][a-zà-ÿ]+(?:\s+[A-ZÀ-Ÿ][a-zà-ÿ]+){0,2})\b', title or "")
        if m:
            name = m.group(1)

    return {
        "nome": name or "",
        "Link de perfil": link or "",
        "Local e descrição": local_desc or ""
    }

# Converte a resposta completa (JSON) da API em um DataFrame
def resp_to_table(resp_obj: Any, max_rows: int = 10) -> Tuple[pd.DataFrame, Optional[str]]:
    parsed, err = safe_json_load(resp_obj)
    if err:
        return pd.DataFrame(columns=["nome", "Link de perfil", "Local e descrição"]), err

    organic_list = find_organic_list(parsed)
    rows = []
    for item in organic_list:
        row = normalize_item_for_table(item)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["nome", "Link de perfil", "Local e descrição"]), None

    df = pd.DataFrame(rows)
    df = df.head(max_rows).reset_index(drop=True)
    return df, None

#   Implementa um atraso de espera com backoff exponencial
def exponential_backoff_sleep(attempt: int):
    wait = min(30, 2 ** attempt)
    time.sleep(wait)

# Executa a chamada POST para o endpoint da API Thordata
def thordata_search(token: str,
                    q: str,
                    engine: str = "google",
                    domain: Optional[str] = None,
                    gl: Optional[str] = None,
                    hl: Optional[str] = None,
                    start: Optional[int] = None,
                    num: Optional[int] = None,
                    render_js: bool = False,
                    extra_params: Optional[Dict[str, Any]] = None,
                    max_retries: int = 6) -> Any:
    if not token:
        raise RuntimeError("Token não informado. Defina THORDATA_TOKEN no ambiente ou cole na UI.")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"engine": engine, "q": q, "json": "1"}
    if domain: data["domain"] = domain
    if gl: data["gl"] = gl
    if hl: data["hl"] = hl
    if start is not None: data["start"] = str(start)
    if num is not None: data["num"] = str(num)
    if render_js: data["render_js"] = "1"
    if extra_params:
        for k, v in (extra_params.items() if isinstance(extra_params, dict) else []):
            if v is not None:
                data[k] = str(v)

    attempt = 0
    while True:
        resp = requests.post(ENDPOINT, headers=headers, data=data, timeout=60)
        if resp.status_code == 200:
            try:
                return resp.json()
            except ValueError:
                return resp.text
        if resp.status_code == 429:
            if attempt >= max_retries:
                raise RuntimeError("Rate limited: excedeu tentativas (429).")
            exponential_backoff_sleep(attempt)
            attempt += 1
            continue
        if resp.status_code == 401:
            raise RuntimeError("401 Unauthorized - token inválido ou expirado.")
        if resp.status_code == 402:
            raise RuntimeError("402 Payment Required - saldo insuficiente.")
        resp.raise_for_status()

st.title("🔎   Pesquisa de Perfis - Linkedin ")
st.caption(f"DB: {DB_PATH.resolve()}")
st.markdown("Execute a busca; o resultado será automaticamente estruturado em tabela (nome, Link de perfil, Local e descrição).")

with st.sidebar:
    st.header("Configurações API / Query")
    env_token = os.getenv("THORDATA_TOKEN", "")
    api_token = st.text_input("TheirData API Key (Bearer)", value=env_token, type="password",
                              help="Recomendado: defina THORDATA_TOKEN no ambiente.")
    st.markdown("---")
    st.header("Parâmetros padrão")
    engine = st.selectbox("Mecanismo", options=["google", "bing"], index=0, key="mecanismo")
    domain = st.selectbox("Domínio Google", options=["google.com", "google.com.br", "google.co.uk"], index=0, key="domain_select")
    gl = st.selectbox("País (gl)", options=["BR", "US", "CA", "UK", ""], index=0, key="gl_select")
    hl = st.selectbox("Idioma (hl)", options=["pt-BR", "en", "pt", ""], index=0, key="hl_select")
    render_js = st.checkbox("Renderizar JS (mais lento/custoso)", value=False, key="render_js_sidebar")

with st.form("search_form"):
    st.subheader("Filtros de busca")
    area = st.selectbox("Área (ex.:)", ["Data Science", "Software Engineering", "DevOps", "Security", "Product", "Design", "Sales/Marketing", "Outro"], index=0, key="area_sel")
    competence = st.text_input("Competência / skill (ex.: Python, AWS, Spark)", placeholder="python, aws, spark", key="competence_input")
    location = st.text_input("Localidade (cidade / estado / país)", placeholder="São Paulo, Brazil", key="location_input")
    free_text = st.text_input("Termos adicionais (ex.: 'Bacharel', 'Mestrado', 'Sênior')", placeholder="", key="free_text_input")
    linkedin_only = st.checkbox("Somente LinkedIn (perfils) — site:linkedin.com/in", value=False,
                                 help="Se marcado, a query será prefixada com site:linkedin.com/in OR site:linkedin.com/pub",
                                 key="linkedin_only_cb")
    per_page = st.slider("Resultados por página (limite para tabela)", min_value=5, max_value=50, value=10, step=5, key="per_page_slider")
    page_idx = st.number_input("Página (0 = primeira)", min_value=0, value=0, step=1, key="page_idx_num")
    show_raw = st.checkbox("Mostrar JSON cru (após consulta)", value=False, key="show_raw_cb")
    submitted = st.form_submit_button("🔎 Pesquisar", key="form_submit_btn")

if "last_resp" not in st.session_state:
    st.session_state["last_resp"] = None
if "last_df" not in st.session_state:
    st.session_state["last_df"] = pd.DataFrame(columns=["nome", "Link de perfil", "Local e descrição"])
if "consulta_open" not in st.session_state:
    st.session_state["consulta_open"] = False

# Constrói a string final de consulta
def build_query(area: str, competence: str, location: str, free_text: str, linkedin_only: bool) -> str:
    parts = []
    if area and area != "Outro":
        parts.append(area)
    if competence:
        parts.append(competence)
    if location:
        parts.append(location)
    if free_text:
        parts.append(free_text)
    q = " ".join(parts).strip()
    if linkedin_only:
        if q:
            q = f"(site:linkedin.com/in OR site:linkedin.com/pub) {q}"
        else:
            q = "(site:linkedin.com/in OR site:linkedin.com/pub)"
    return q

if submitted:
    token_to_use = api_token.strip() or os.getenv("THORDATA_TOKEN", "").strip()
    if not token_to_use:
        st.error("Token não fornecido. Defina THORDATA_TOKEN no ambiente ou cole a chave no campo da lateral.")
    else:
        q = build_query(area, competence, location, free_text, linkedin_only)
        if not q:
            st.warning("Query vazia — informe ao menos uma competência, área ou localidade.")
        else:
            start = page_idx * per_page
            with st.spinner("Consultando Thordata (SERP)..."):
                try:
                    resp_obj = thordata_search(token=token_to_use, q=q, engine=engine,
                                                 domain=domain, gl=(gl or None), hl=(hl or None),
                                                 start=start, num=per_page, render_js=render_js)
                except Exception as e:
                    st.error(f"Erro na busca: {e}")
                    resp_obj = None

            st.session_state["last_resp"] = resp_obj

            if resp_obj is not None:
                df_table, err = resp_to_table(resp_obj, max_rows=per_page)
                if err:
                    st.warning(err)
                st.session_state["last_df"] = df_table
            else:
                st.session_state["last_df"] = pd.DataFrame(columns=["nome", "Link de perfil", "Local e descrição"])

st.markdown("---")
st.subheader("Ou: cole / carregue um JSON retornado pela API (opcional)")
col1, col2 = st.columns([3, 1])
with col1:
    pasted = st.text_area("Cole o JSON aqui (opcional)", height=140, placeholder='Cole aqui o JSON retornado pela API...', key="pasted_json")
with col2:
    upload = st.file_uploader("Ou faça upload do arquivo JSON", type=["json"], key="upload_json")

if st.button("🔧 Montar tabela a partir do JSON colado/subido", key="montar_json_btn"):
    content = None
    if upload is not None:
        try:
            raw = upload.read()
            content = raw.decode("utf-8")
        except Exception as e:
            st.error(f"Erro lendo arquivo: {e}")
    elif pasted and pasted.strip():
        content = pasted.strip()

    if content:
        df_table, err = resp_to_table(content, max_rows=per_page)
        if err:
            st.warning(err)
        st.session_state["last_df"] = df_table
        parsed, jerr = safe_json_load(content)
        if parsed:
            st.session_state["last_resp"] = parsed
    else:
        st.info("Nenhum JSON fornecido para montar a tabela.")

st.markdown("---")
df_table = st.session_state.get("last_df", pd.DataFrame(columns=["nome", "Link de perfil", "Local e descrição"]))
count = int(df_table.shape[0]) if hasattr(df_table, "shape") else 0
st.markdown(f"### Resultados estruturados — {count} registros (mostrando até {per_page})")

if count == 0:
    st.info("Nenhum registro extraído para a tabela após limpeza heurística.")
else:
    display_df = df_table[["nome", "Link de perfil", "Local e descrição"]].copy()
    display_df["Local e descrição"] = display_df["Local e descrição"].astype(str).str.replace("\n", " ").str.slice(0, 500)
    st.dataframe(display_df, use_container_width=True)

    export_col1, export_col2, export_col3 = st.columns([1,1,1])

    with export_col1:
        if st.button("⬇️ Exportar CSV (DB — todos perfis)", key="export_db_csv_btn"):
            try:
                df_db = fetch_all_profiles(_conn_for_app)
                if df_db.empty:
                    st.info("Banco vazio — nada para exportar.")
                else:
                    csv_bytes = df_db.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Clique para baixar CSV (DB)",
                        data=csv_bytes,
                        file_name="sourcing_profiles_db.csv",
                        mime="text/csv",
                        key="download_db_csv_btn"
                    )
            except Exception as e:
                st.error(f"Falha ao exportar CSV do DB: {e}")

    with export_col2:
        if st.button("⬇️ Exportar XLSX (DB — todos perfis)", key="export_db_xlsx_btn"):
            try:
                df_db = fetch_all_profiles(_conn_for_app)
                if df_db.empty:
                    st.info("Banco vazio — nada para exportar.")
                else:
                    out = io.BytesIO()
                    with pd.ExcelWriter(out, engine="openpyxl") as xw:
                        df_db.to_excel(xw, sheet_name="Perfis", index=False)
                    out.seek(0)
                    st.download_button(
                        "Clique para baixar XLSX (DB)",
                        data=out.getvalue(),
                        file_name="sourcing_profiles_db.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_db_xlsx_btn"
                    )
            except Exception as e:
                st.error(f"Falha ao exportar XLSX do DB: {e}\n(Verifique se openpyxl está instalado: pip install openpyxl)")

    with export_col3:
        csv_shown = display_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Exportar CSV (tabela exibida)", csv_shown, file_name="sourcing_perfis_displayed.csv", mime="text/csv", key="download_displayed_csv_btn")

    btn_col1, btn_col2 = st.columns([1,1])
    with btn_col1:
        if st.button("✅ Cadastrar (salvar no DB)", key="cadastrar_db_btn"):
            try:
                conn = _conn_for_app
                inserted, ignored = save_profiles_to_db(df_table, conn)
                st.success(f"Registros salvos: {inserted}. Ignorados (duplicados): {ignored}.")
            except Exception as e:
                st.error(f"Erro ao salvar no DB: {e}")
    with btn_col2:
        if st.button("🔁 Reset sessão (remove last_df)", key="reset_last_df_btn"):
            st.session_state["last_df"] = pd.DataFrame(columns=["nome", "Link de perfil", "Local e descrição"])
            st.success("Sessão reiniciada (last_df limpo).")

if count > 0:
    st.markdown("### Links (clique para abrir)")
    for _, r in df_table.head(per_page).iterrows():
        nome = r.get("nome") or "(sem nome)"
        link = r.get("Link de perfil") or ""
        local_desc = (r.get("Local e descrição") or "")[:200]
        if link:
            st.write(f"- [{nome}]({link}) — {local_desc}")
        else:
            st.write(f"- {nome} — {local_desc}")

st.markdown("---")
st.subheader("📚 Registros cadastrados no banco")
btns = st.columns([1,1,1])
with btns[0]:
    if st.button("Carregar registros do DB", key="carregar_db_btn"):
        try:
            df_db = fetch_all_profiles(_conn_for_app)
            if df_db.empty:
                st.info("Nenhum registro cadastrado ainda.")
            else:
                st.dataframe(df_db, use_container_width=True)
                csv_db = df_db.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Exportar CSV (DB)", csv_db, file_name="sourcing_profiles_db.csv", mime="text/csv", key="download_db_btn_panel")
        except Exception as e:
            st.error(f"Erro ao ler DB: {e}")
with btns[1]:
    if st.button("🔎 Consulta", key="open_consulta_btn"):
        st.session_state["consulta_open"] = True
with btns[2]:
    if st.button("🔁 Reset sessão (remove last_df)", key="reset_session_btn"):
        st.session_state["last_df"] = pd.DataFrame(columns=["nome", "Link de perfil", "Local e descrição"])
        st.success("Sessão reiniciada (last_df limpo).")

if st.session_state.get("consulta_open", False):
    st.markdown("---")
    st.header("🔍 Consulta no banco — Filtrar por Localização e Competência")
    st.markdown("Preencha um ou ambos os campos. A busca fará LIKE (case-insensitive) sobre o campo `Local e descrição`.")
    col_a, col_b = st.columns(2)
    with col_a:
        consulta_location = st.text_input("Localização (ex.: São Paulo, Campinas, Brasil)", value="", key="consulta_location_input")
    with col_b:
        consulta_competence = st.text_input("Competência (ex.: Python, AWS, DevOps)", value="", key="consulta_competence_input")

    consulta_cols = st.columns([1,1,1])
    with consulta_cols[0]:
        if st.button("🔎 Buscar", key="consulta_buscar_btn"):
            try:
                df_res = query_profiles(_conn_for_app, location=consulta_location, competence=consulta_competence)
                if df_res.empty:
                    st.info("Nenhum registro encontrado para os critérios fornecidos.")
                else:
                    st.success(f"Encontrados {len(df_res)} registros.")
                    st.dataframe(df_res, use_container_width=True)
                    csv_r = df_res.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Exportar CSV (consulta)", csv_r, file_name="consulta_sourcing_profiles.csv", mime="text/csv", key="download_consulta_btn")
            except Exception as e:
                st.error(f"Erro na consulta: {e}")
    with consulta_cols[1]:
        if st.button("🧾 Mostrar todos", key="consulta_mostrar_todos_btn"):
            try:
                df_all = fetch_all_profiles(_conn_for_app)
                if df_all.empty:
                    st.info("Nenhum registro cadastrado.")
                else:
                    st.dataframe(df_all, use_container_width=True)
                    csv_all = df_all.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Exportar CSV (todos)", csv_all, file_name="all_sourcing_profiles.csv", mime="text/csv", key="download_all_btn")
            except Exception as e:
                st.error(f"Erro ao carregar todos: {e}")
    with consulta_cols[2]:
        if st.button("✖ Fechar / Voltar", key="consulta_fechar_btn"):
            st.session_state["consulta_open"] = False
            st.experimental_rerun()

if show_raw:
    resp_obj = st.session_state.get("last_resp")
    if resp_obj is None:
        st.info("Sem resposta em cache para exibir.")
    else:
        with st.expander("🔧 JSON cru (data retornada pela API) — expandir para inspecionar"):
            try:
                pretty = json.dumps(resp_obj, ensure_ascii=False, indent=2)
            except Exception:
                pretty = str(resp_obj)
            st.code(pretty[:20000], language="json")

st.markdown("---")
st.markdown(
    "**Aviso de Privacidade e Uso:** Mesmo sem busca por e-mails, trate nomes e links com responsabilidade (LGPD/GDPR)."
)
