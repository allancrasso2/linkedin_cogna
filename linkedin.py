# streamlit_thordata_sourcing_secure.py
# -*- coding: utf-8 -*-
"""
Pesquisa de Perfis - Linkedin · Thordata SERP — integrado (consulta -> tabela)
Versão: oculta API key por padrão, valida entrada (location vs competence),
filtro pós-processamento para garantir residência e presença da competência.
"""
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

st.set_page_config(page_title="Cogna Talentos", layout="wide")

# ---------------------------------------------------------------------
# Toggle para mostrar/ocultar a seção "Cole / carregue JSON retornado"
# ---------------------------------------------------------------------
# Por padrão ocultamos conforme solicitado. Mude para True para reativar.
SHOW_JSON_SECTION = False
# ---------------------------------------------------------------------

# ---------------- Config / endpoint / DB ----------------
ENDPOINT = "https://scraperapi.thordata.com/request"
DB_PATH = Path("./db/sourcing_profiles.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------- Settings ----------------
USER_AGENT = "Mozilla/5.0 (compatible; ThordataBot/1.0)"
PER_REQUEST_DELAY = 0.5

# ---------------- Residence extraction heuristics ----------------
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

_LOCATION_SEP = r"(?:,|\u2022|\u00B7\||–|—|-)"

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

# ---------------- Database helpers ----------------
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

def fetch_all_profiles(conn: sqlite3.Connection) -> pd.DataFrame:
    cur = conn.cursor()
    cur.execute("SELECT id, nome, profile_link, local_desc, created_at FROM sourcing_profiles ORDER BY id DESC")
    rows = cur.fetchall()
    cols = ["id", "nome", "Link de perfil", "Local e descrição", "created_at"]
    return pd.DataFrame(rows, columns=cols)

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

# ---------------- Utils / Heurísticas de validação ----------------
COMMON_SKILLS = {
    "python","java","javascript","typescript","go","golang","c#","c++","c","sql","aws","azure","gcp",
    "docker","kubernetes","terraform","ansible","spark","hadoop","react","angular","node","django","flask",
    "scala","ruby","php","rust","matlab","r","swift","objective-c","android","ios","git","linux","nosql",
    "power bi","powerbi","excel","tableau"
}

def tokenize_input(s: str) -> List[str]:
    if not s or not isinstance(s, str):
        return []
    tokens = re.split(r"[,\|/;]+", s)
    clean = []
    for t in tokens:
        t = t.strip().lower()
        if t:
            clean.append(t)
    return clean

def is_probable_competence(text: str) -> bool:
    toks = tokenize_input(text)
    for tok in toks:
        if tok in COMMON_SKILLS:
            return True
        # generic pattern for short tokens like "python", "aws"
        if re.fullmatch(r"[a-z0-9\+\-\#\. ]{1,30}", tok):
            if tok.upper() not in BRAZIL_STATE_ABBR and len(tok) > 1:
                return True
    return False

def is_probable_location(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    t = text.strip()
    low = t.lower()
    # contains state full name or abbreviation
    for s in BRAZIL_STATE_NAMES + [s.lower() for s in BRAZIL_STATE_ABBR]:
        if s in low:
            return True
    # patterns like "City, ST" or "City/ST"
    if re.search(r",\s*[A-Za-z]{1,4}$", t) or re.search(r"/\s*[A-Za-z]{1,4}$", t):
        return True
    # single capitalized word (City) with length > 3
    if re.fullmatch(r"[A-ZÀ-Ý][a-zà-ÿ]+(?:\s+[A-ZÀ-Ý][a-zà-ÿ]+){0,2}", t):
        if len(t) > 3:
            return True
    # patterns like "City - State" or "City | State"
    if re.search(r"[A-Za-zÀ-ÿ]+(?:\s+[A-Za-zÀ-ÿ]+)*\s*[-|]\s*[A-Za-zÀ-ÿ]{1,20}", t):
        return True
    return False

def is_strict_location_ok(text: str) -> bool:
    """
    Requisito mais rígido para permitir uma busca:
      - deve conter vírgula ou barra com UF (ex.: 'Campinas, SP' ou 'Campinas/SP')
      - ou conter nome completo de estado (ex.: 'São Paulo')
      - ou ser 'Cidade - Estado' / 'Cidade | Estado'
    """
    if not text or not isinstance(text, str):
        return False
    t = text.strip()
    if re.search(r",\s*[A-Za-z]{1,4}$", t) or re.search(r"/\s*[A-Za-z]{1,4}$", t):
        return True
    low = t.lower()
    for s in BRAZIL_STATE_NAMES + [s.lower() for s in BRAZIL_STATE_ABBR]:
        if s in low:
            return True
    if re.search(r"[A-Za-zÀ-ÿ]+(?:\s+[A-Za-zÀ-ÿ]+)*\s*[-|]\s*[A-Za-zÀ-ÿ]{1,20}", t):
        return True
    return False

# ---------------- JSON handling / normalization ----------------
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

def extract_name_from_linkedin_title(title: Optional[str]) -> Optional[str]:
    if not title or not isinstance(title, str):
        return None
    s = title.split("|")[0]
    s = s.split(" - ")[0].split(" — ")[0].strip()
    return s if s else None

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

# ---------------- filtering: ensure competence and location actually present ----------------
def text_contains_token_set(text: str, tokens: List[str]) -> bool:
    t = (text or "").lower()
    for tok in tokens:
        if tok and tok not in t:
            return False
    return True

def profile_has_competence(item: Dict, competence: str) -> bool:
    if not competence:
        return True
    comp = competence.strip().lower()
    tokens = [c.strip() for c in re.split(r"[,\|/;]+", comp) if c.strip()]
    search_fields = []
    if isinstance(item, dict):
        for f in ("title","job_title","description","snippet","skills","education","experience","text","body"):
            v = item.get(f)
            if isinstance(v, (list, tuple)):
                search_fields.extend([str(x) for x in v if x])
            elif v:
                search_fields.append(str(v))
        if "raw" in item and isinstance(item["raw"], dict):
            for k,v in item["raw"].items():
                if isinstance(v, str):
                    search_fields.append(v)
    combined = " ".join(search_fields).lower()
    for tok in tokens:
        if tok.lower() not in combined:
            return False
    return True

def profile_matches_location(item: Dict, location_query: str) -> bool:
    if not location_query:
        return True
    tokens = [t.strip().lower() for t in re.split(r"[,\|/]+", location_query) if t.strip()]
    loc_candidates = []
    for f in ("location","place","displayed_location","city","region","locale","area"):
        v = item.get(f)
        if isinstance(v, str) and v.strip():
            loc_candidates.append(v)
    vdesc = item.get("description") or item.get("snippet") or ""
    if vdesc:
        loc_candidates.append(vdesc)
    normalized_local = normalize_item_for_table(item).get("Local e descrição") or ""
    if normalized_local:
        loc_candidates.append(normalized_local)
    combined = " ".join(loc_candidates).lower()
    for tok in tokens:
        if tok not in combined:
            return False
    return True

def filter_api_results(parsed_resp: Dict, df_table: pd.DataFrame, competence: str, location: str) -> pd.DataFrame:
    if df_table is None or df_table.empty:
        return df_table
    items = find_organic_list(parsed_resp)
    item_by_link = {}
    for it in items:
        link = (it.get("link") or it.get("url") or it.get("source_url") or it.get("final_url") or "").strip()
        if link:
            item_by_link[link] = it
    kept_rows = []
    for _, r in df_table.iterrows():
        link = (r.get("Link de perfil") or "").strip()
        item = item_by_link.get(link)
        if item is None:
            name = (r.get("nome") or "").strip().lower()
            for it in items:
                tit = (it.get("title") or it.get("job_title") or "")
                sn = (it.get("description") or it.get("snippet") or "")
                if name and (name in str(tit).lower() or name in str(sn).lower()):
                    item = it
                    break
        if item is None:
            cand_text = (r.get("Local e descrição") or "") + " " + (r.get("nome") or "")
            if competence and competence.strip() and competence.strip().lower() not in cand_text.lower():
                continue
            if location and location.strip() and location.strip().lower() not in cand_text.lower():
                continue
            kept_rows.append(r)
            continue
        if not profile_has_competence(item, competence):
            continue
        if not profile_matches_location(item, location):
            continue
        kept_rows.append(r)
    if not kept_rows:
        return pd.DataFrame(columns=df_table.columns)
    return pd.DataFrame(kept_rows).reset_index(drop=True)

# ---------------- Thordata API call / backoff ----------------
def exponential_backoff_sleep(attempt: int):
    wait = min(30, 2 ** attempt)
    time.sleep(wait)

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
        raise RuntimeError("Token não informado. Defina thordata_token.txt ou THORDATA_TOKEN no ambiente.")
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

# ---------------- API key loading + sidebar unlock mechanism ----------------
SECRETS_FILE = Path(__file__).parent / "thordata_token.txt"
THORDATA_TOKEN = ""

if SECRETS_FILE.exists():
    try:
        THORDATA_TOKEN = SECRETS_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        THORDATA_TOKEN = ""

if not THORDATA_TOKEN:
    try:
        import secrets as _secrets  # optional module
        THORDATA_TOKEN = getattr(_secrets, "THORDATA_TOKEN", "") or ""
    except Exception:
        THORDATA_TOKEN = THORDATA_TOKEN or ""

if not THORDATA_TOKEN:
    THORDATA_TOKEN = os.getenv("THORDATA_TOKEN", "").strip()

# ---------------- UI ----------------
st.title("🔎   Pesquisa de Perfis - Linkedin ")
st.caption(f"DB: {DB_PATH.resolve()}")
st.markdown("Execute a busca; o resultado será automaticamente estruturado em tabela (nome, Link de perfil, Local e descrição).")

with st.sidebar:
    st.header("Configurações API / Query")
    if THORDATA_TOKEN:
        st.markdown("**API Key:** carregada a partir de arquivo local (oculta).")
    else:
        st.warning("Nenhuma API Key local encontrada. Defina `thordata_token.txt`, `secrets.py`, ou variável de ambiente `THORDATA_TOKEN`.")
    st.markdown("---")
    st.caption("Desbloqueie a edição da API Key com senha (apenas admin).")
    passwd = st.text_input("Senha para desbloquear (admin)", type="password", key="unlock_pwd")
    if st.button("Desbloquear", key="unlock_btn"):
        if passwd == "Cogna26":
            st.session_state["api_unlocked"] = True
            st.success("Sidebar desbloqueada. Campo da API agora visível (oculto por padrão).")
        else:
            st.session_state["api_unlocked"] = False
            st.error("Senha incorreta.")
    api_token_field_value = THORDATA_TOKEN or ""
    if st.session_state.get("api_unlocked", False):
        api_token = st.text_input("TheirData API Key (Bearer)", value=api_token_field_value, type="password", key="api_token_input")
        if api_token and api_token != THORDATA_TOKEN:
            THORDATA_TOKEN = api_token.strip()
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

# Validation: require both fields and ensure not swapped
def validate_field_positions(location_input: str, competence_input: str) -> Tuple[bool, Optional[str]]:
    loc = (location_input or "").strip()
    comp = (competence_input or "").strip()
    # require location and competence present
    if not comp:
        return False, "Campo 'Competência' obrigatório — informe a(s) skill(s) (ex.: Python)."
    if not loc:
        return False, "Campo 'Localidade' obrigatório — informe cidade/estado/país (ex.: Campinas/SP)."
    # if location looks like competence -> reject
    if is_probable_competence(loc) and not is_probable_location(loc):
        return False, "Localidade parece conter competências (ex.: 'Python'). Corrija os campos."
    # if competence looks like location -> reject
    if is_probable_location(comp) and not is_probable_competence(comp):
        return False, "Competência parece conter localização (ex.: 'Campinas/SP'). Corrija os campos."
    # Additional sanity: ensure location is probable location
    if not is_probable_location(loc):
        # allow some flexibility: if user writes long textual location, still accept; otherwise warn
        if len(loc) < 4 or not re.search(r"[A-Za-zÀ-ÿ]", loc):
            return False, "Localidade inserida não parece válida. Use formato 'Cidade, Estado' ou 'Cidade/UF' (ex.: Campinas/SP)."
    return True, None

# When the user submits:
if submitted:
    token_to_use = ""
    if st.session_state.get("api_unlocked", False) and "api_token_input" in st.session_state:
        token_to_use = (st.session_state.get("api_token_input") or "").strip()
    if not token_to_use:
        token_to_use = THORDATA_TOKEN.strip()
    if not token_to_use:
        st.error("Token não fornecido. Defina thordata_token.txt ou THORDATA_TOKEN no ambiente, ou desbloqueie e cole a chave.")
    else:
        # primeira validação (mais amigável)
        ok, reason = validate_field_positions(location, competence)
        if not ok:
            st.error(reason)
        else:
            # --- NOVA guarda if/else rigorosa ---
            # tokens
            loc_tokens = tokenize_input(location)
            comp_tokens = tokenize_input(competence)

            # 1) bloqueia se qualquer token de localidade é claramente uma skill
            loc_has_skill_token = any(tok in COMMON_SKILLS for tok in loc_tokens)

            # 2) bloqueia se localidade não satisfaz critério "estrito" (ex.: sem UF/estado)
            loc_not_strict = not is_strict_location_ok(location)

            # 3) bloqueia se competence parecer uma localidade (ex.: 'Campinas/SP' no campo competence)
            comp_looks_like_loc = is_probable_location(competence) and not is_probable_competence(competence)

            # 4) bloqueia se os dois campos forem identicos (provável erro)
            fields_identical = location.strip().lower() == competence.strip().lower() and bool(location.strip())

            # If/else: decidir se seguimos com a consulta
            if loc_has_skill_token:
                st.error("Bloqueado: o campo 'Localidade' contém tokens que parecem competências (ex.: 'Python', 'Java'). Corrija o campo Localidade.")
                st.info("Dica: escreva 'Campinas, SP' ou 'Campinas/SP' no campo Localidade.")
                # interrompe aqui
                st.stop()
            if comp_looks_like_loc:
                st.error("Bloqueado: o campo 'Competência' parece conter uma localidade (ex.: 'Campinas/SP'). Corrija o campo Competência.")
                st.stop()
            if fields_identical:
                st.error("Bloqueado: os campos 'Localidade' e 'Competência' não podem ser iguais.")
                st.stop()
            # Se a localidade não estiver em formato estrito, bloqueia (para evitar buscas por skills no campo Local)
            if loc_not_strict:
                # mas dê uma mensagem clara e orientações
                st.error("Bloqueado: o campo 'Localidade' não está em formato esperado (ex.: 'Cidade, UF' ou 'Cidade/UF').")
                st.info("Formate a localidade como 'Campinas, SP' ou 'Campinas/SP' para prosseguir.")
                st.stop()
            # --- fim da guarda if/else ---

            # se chegamos aqui, os campos parecem consistentes — monta query e faz a chamada
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
                    filtered_df = filter_api_results(resp_obj, df_table, competence or "", location or "")
                    if filtered_df.empty and (competence or location):
                        st.warning("Nenhum resultado após aplicar validações de Localidade/Competência. Tente relaxar os filtros ou revisar os campos.")
                        # keep original df_table for inspection (but still warn)
                        st.session_state["last_df"] = df_table
                    else:
                        st.session_state["last_df"] = filtered_df if not filtered_df.empty else df_table
                else:
                    st.session_state["last_df"] = pd.DataFrame(columns=["nome", "Link de perfil", "Local e descrição"])

# --- Seção de COLAR/FAZER UPLOAD JSON (opcional) ---
# Abaixo o bloco fica oculto por padrão (SHOW_JSON_SECTION=False).
if SHOW_JSON_SECTION:
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
# ---------------------------------------------------------------------

# ----- Mostrar resultados estruturados (da última resposta / upload) -----
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

# painel: visualizar registros já cadastrados no banco (com botão Consulta)
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

