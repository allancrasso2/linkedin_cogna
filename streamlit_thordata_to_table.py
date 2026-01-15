# streamlit_thordata_to_table.py
# -*- coding: utf-8 -*-
"""
Script mínimo: carrega JSON (upload ou paste), extrai 'organic' (ou 'results','items') e
monta DataFrame com colunas:
 - nome
 - Link de perfil
 - Local e descrição
"""
import json
import re
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

st.set_page_config(page_title="JSON -> Tabela (Thordata -> Perfis)", layout="wide")

st.title("Gerar tabela a partir do JSON retornado (Thordata / Scraper API)")
st.markdown("Faça upload do JSON retornado pela API ou cole o JSON no campo. O app exibirá a tabela com as colunas `nome`, `Link de perfil`, `Local e descrição`.")

# helpers
EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+', re.I)
# heurística simples para extrair nomes de titles LinkedIn
def extract_name_from_title(title: Optional[str]) -> Optional[str]:
    if not title:
        return None
    # corta após pipe e traço que normalmente separam cargo/LinkedIn
    t = title.split("|")[0].split(" - ")[0].split(" — ")[0].strip()
    if not t:
        return None
    # se título contém "LinkedIn" ou "perfil", ignore
    if "linkedin" in t.lower():
        return None
    # se tem vírgula ou '·' pode ser que comece com local; prefer palavras iniciais capitalizadas
    # pegar até 4 palavras iniciais com capitalização adequada (heurística)
    parts = t.split()
    candidate = []
    for w in parts:
        if len(candidate) >= 4:
            break
        # aceita palavras iniciando com letra maiúscula
        if re.match(r'^[A-ZÀ-Ÿ]', w):
            candidate.append(w)
        else:
            # caso encontremos uma palavra não capitalizada logo no começo, aborta heurística
            if not candidate:
                candidate = parts[:2]  # fallback: primeira e segunda palavra
                break
            else:
                break
    if candidate:
        name = " ".join(candidate)
        # sanity: se muito curto, devolver full t
        if len(name) < 3:
            return t
        return name
    return t

def find_candidates_from_item(item: Dict[str, Any]) -> Dict[str, str]:
    """
    Mapeia um objeto 'organic' / item -> dicionário com campos nome, link, local_desc
    """
    title = item.get("title") or ""
    # possíveis campos de link
    link = item.get("link") or item.get("url") or item.get("source_url") or ""
    # possíveis campos de descrição
    desc = item.get("description") or item.get("snippet") or ""
    # extrair nome preferencialmente do title
    nome = extract_name_from_title(title)
    # se extracao falhar, tentar heurísticas na description
    if not nome:
        # buscar padrões do tipo "Nome Sobrenome" na descrição
        m = re.search(r'\b([A-ZÀ-Ÿ][a-zà-ÿ]+(?:\s+[A-ZÀ-Ÿ][a-zà-ÿ]+){0,3})\b', desc)
        if m:
            nome = m.group(1)
    # trim
    nome = (nome or "").strip()
    local_desc = (desc or "").strip()
    return {"nome": nome, "Link de perfil": link, "Local e descrição": local_desc}

def load_json_from_upload_or_text():
    uploaded = st.file_uploader("Upload do arquivo JSON (opcional)", type=["json"])
    raw_text = st.text_area("Ou cole o JSON aqui (opcional)", height=200)
    data = None
    if uploaded is not None:
        try:
            data = json.load(uploaded)
        except Exception as e:
            st.error(f"Erro ao ler arquivo JSON: {e}")
    elif raw_text and raw_text.strip():
        try:
            data = json.loads(raw_text)
        except Exception as e:
            st.error(f"Erro ao parsear JSON colado: {e}")
    return data

data = load_json_from_upload_or_text()
if data is None:
    st.info("Envie um arquivo JSON ou cole o JSON retornado pela API para montar a tabela.")
    st.stop()

# localizar array de resultados: 'organic' preferencialmente, senão 'results'/'items'/'data'
candidates_arrays = ("organic", "organic_results", "results", "items", "data")
items = []
for key in candidates_arrays:
    arr = data.get(key) if isinstance(data, dict) else None
    if isinstance(arr, list) and arr:
        # se 'data' tem objetos com key 'items' ou 'organic' aninhado, tentar descobrir
        items = arr
        break

# Se nenhum array encontrado, tentar procurar 'organic' dentro do JSON aninhado (fallback simples)
if not items and isinstance(data, dict):
    # procura primeira lista dentro do dicionário (heurística)
    for k, v in data.items():
        if isinstance(v, list) and v:
            items = v
            break

if not items:
    st.warning("Não foi encontrada uma lista de resultados no JSON (chaves esperadas: organic, results, items, data). Mostrando o JSON cru.")
    st.code(json.dumps(data, ensure_ascii=False, indent=2)[:20000], language="json")
    st.stop()

# montar tabela
rows = []
for it in items:
    if not isinstance(it, dict):
        continue
    row = find_candidates_from_item(it)
    # ignorar linhas que correspondem a cabeçalho/metadata ruído
    # heurística simples: se 'Local e descrição' começa com '{' ou contém 'search_metadata', pula
    ld = (row["Local e descrição"] or "").strip()
    if ld.startswith("{") and "search_metadata" in ld:
        continue
    # se tudo vazio -> ainda assim incluir (se quiser filtrar depois)
    rows.append(row)

df = pd.DataFrame(rows, columns=["nome", "Link de perfil", "Local e descrição"])

# remover linhas em branco (opcional)
df_display = df.copy()
# remover linhas onde os três campos estão vazios
df_display = df_display[~((df_display["nome"].str.strip() == "") & (df_display["Link de perfil"].str.strip() == "") & (df_display["Local e descrição"].str.strip() == ""))]

st.markdown(f"### Tabela gerada — {len(df_display)} registros")
if df_display.empty:
    st.info("A tabela ficou vazia após limpeza heurística.")
else:
    st.dataframe(df_display, use_container_width=True)
    st.download_button("⬇️ Exportar CSV", df_display.to_csv(index=False).encode("utf-8"), file_name="perfis_mapeados.csv", mime="text/csv")
    st.markdown("Observação: o campo `nome` é extraído por heurística do `title` (LinkedIn) ou da `description` quando possível.")

