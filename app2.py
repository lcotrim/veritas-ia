import re
from datetime import datetime
import sqlite3
import json
from pathlib import Path
import tempfile
import io

import streamlit as st
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Caminho do banco de dados local
PASTA_PROJETO = Path(__file__).resolve().parent
ARQUIVO_BANCO = PASTA_PROJETO / "veritas_compliance.db"


def conectar_banco():
    """Abre uma conexão com o banco SQLite."""
    return sqlite3.connect(ARQUIVO_BANCO)


def preparar_banco():
    """Cria a tabela de casos de auditoria se ela ainda não existir."""
    with conectar_banco() as conexao:
        conexao.execute("""
            CREATE TABLE IF NOT EXISTS casos_auditoria (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                criado_em TEXT NOT NULL,
                identificacao TEXT,
                contexto TEXT,
                relato_a TEXT NOT NULL,
                relato_b TEXT NOT NULL,
                evidencias TEXT,
                divergencias TEXT NOT NULL,
                perguntas TEXT NOT NULL
            )
        """)


preparar_banco()


# --------------------------------------------------
# CONFIGURACAO DA APLICACAO
# --------------------------------------------------

st.set_page_config(
    page_title="Veritas | Auditoria de Consistência Factual",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Veritas — Auditoria de Conformidade e Entrevistas")
st.caption("Cruzamento de depoimentos, gravação por microfone, transcrição via IA e exportação de relatório em PDF")

st.info(
    "Esta ferramenta auxilia auditores a capturar depoimentos, identificar "
    "inconsistências, gerar perguntas cirúrgicas e emitir relatórios oficiais em PDF."
)


# --------------------------------------------------
# MODELOS DE IA (CACHE)
# --------------------------------------------------

@st.cache_resource
def carregar_modelo_semantico():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


@st.cache_resource
def carregar_modelo_nli():
    return pipeline(
        "text-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        top_k=None
    )


@st.cache_resource
def carregar_modelo_asr():
    """Carrega o modelo Whisper para transcrição automática de áudio."""
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base"
    )


# --------------------------------------------------
# FUNÇÕES DE PROCESSAMENTO E PDF
# --------------------------------------------------

def transcrever_audio(audio_bytes, modelo_asr):
    """Transcreve arquivo de áudio temporário usando Whisper."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes.getbuffer())
        tmp_path = tmp.name
    
    resultado = modelo_asr(tmp_path)
    return resultado["text"]


def gerar_pdf_relatorio(identificacao, contexto, relato_a, relato_b, evidencias, divergencias, perguntas):
    """Gera o relatório em PDF formatado utilizando ReportLab."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'SubtitleStyle',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#64748B'),
        spaceAfter=15
    )
    heading_style = ParagraphStyle(
        'HeadingStyle',
        parent=styles['Heading2'],
        fontSize=11,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=10,
        spaceAfter=4
    )
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#334155'),
        spaceAfter=6
    )
    
    # Cabeçalho
    story.append(Paragraph("Veritas — Relatório de Auditoria de Conformidade e Entrevista", title_style))
    data_geracao = datetime.now().strftime("%d/%m/%Y às %H:%M:%S")
    story.append(Paragraph(f"Emitido em: {data_geracao}", subtitle_style))
    story.append(Spacer(1, 5))
    
    # Escopo
    if identificacao:
        story.append(Paragraph(f"<b>Processo / Caso:</b> {identificacao}", body_style))
    if contexto:
        story.append(Paragraph(f"<b>Escopo / Contexto:</b> {contexto}", body_style))
    story.append(Spacer(1, 8))
    
    # Depoimentos
    story.append(Paragraph("1. Depoimentos Registrados", heading_style))
    story.append(Paragraph(f"<b>Depoimento 01:</b> {relato_a}", body_style))
    story.append(Paragraph(f"<b>Depoimento 02:</b> {relato_b}", body_style))
    story.append(Spacer(1, 8))
    
    # Evidências
    story.append(Paragraph("2. Evidências Documentais de Apoio", heading_style))
    story.append(Paragraph(f"{evidencias if evidencias.strip() else 'Nenhuma evidência informada.'}", body_style))
    story.append(Spacer(1, 8))
    
    # Divergências
    story.append(Paragraph("3. Divergências Estruturais Detectadas", heading_style))
    if divergencias:
        for idx, item in enumerate(divergencias, 1):
            val_a = ", ".join(item["relato_a"]) if isinstance(item["relato_a"], list) else item["relato_a"]
            val_b = ", ".join(item["relato_b"]) if isinstance(item["relato_b"], list) else item["relato_b"]
            story.append(Paragraph(f"• <b>{item['categoria']}</b> — Depoimento 1: '{val_a}' | Depoimento 2: '{val_b}'", body_style))
    else:
        story.append(Paragraph("Nenhuma divergência numérica ou temporal explícita detectada.", body_style))
    story.append(Spacer(1, 8))
    
    # Perguntas
    story.append(Paragraph("4. Perguntas Cirúrgicas de Confronto (Modo Inquirição)", heading_style))
    if perguntas:
        for p in perguntas:
            story.append(Paragraph(f"• {p}", body_style))
    else:
        story.append(Paragraph("Nenhuma pergunta gerada.", body_style))
        
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def extrair_dados(texto):
    return {
        "Datas": re.findall(r"\b\d{1,2}/\d{1,2}/(?:\d{2}|\d{4})\b", texto),
        "Horarios": re.findall(r"\b\d{1,2}:\d{2}\b", texto),
        "Numeros": re.findall(r"\b\d+(?:[.,]\d+)?\b", texto)
    }


def comparar_relatos(relato_a, relato_b):
    dados_a = extrair_dados(relato_a)
    dados_b = extrair_dados(relato_b)
    divergencias = []

    for categoria in dados_a:
        valores_a = dados_a[categoria]
        valores_b = dados_b[categoria]

        if valores_a and valores_b:
            if valores_a != valores_b:
                divergencias.append({
                    "categoria": categoria,
                    "relato_a": valores_a,
                    "relato_b": valores_b
                })
    return divergencias


def gerar_perguntas_confronto(divergencias, resultados_nli):
    perguntas = []
    
    for item in divergencias:
        cat = item["categoria"]
        val_a = ", ".join(item["relato_a"]) if isinstance(item["relato_a"], list) else item["relato_a"]
        val_b = ", ".join(item["relato_b"]) if isinstance(item["relato_b"], list) else item["relato_b"]
        
        perguntas.append(
            f"Divergência de {cat}: O depoimento 01 menciona '{val_a}', enquanto o depoimento 02 aponta '{val_b}'. Qual registro documental valida a informação correta?"
        )
        
    for item in resultados_nli:
        for res in item["resultado_ab"]:
            if "contradict" in res["label"].lower() and res["score"] > 0.6:
                perguntas.append(
                    f"Contradição lógica identificada: A frase '{item['frase_a']}' conflita com '{item['frase_b']}' ({res['score']:.1%} de confiança). Como esclarecer essa incompatibilidade?"
                )

    if not perguntas:
        perguntas.append("Os relatos mostram alinhamento estrutural básico. Recomenda-se explorar detalhes operacionais complementares.")

    return list(dict.fromkeys(perguntas))


def separar_frases(texto):
    frases = re.split(r"(?<=[.!?])\s+|\n+", texto)
    return [f.strip() for f in frases if f.strip()]


def comparar_semanticamente(texto_a, texto_b, modelo):
    frases_a = separar_frases(texto_a)
    frases_b = separar_frases(texto_b)

    if not frases_a or not frases_b:
        return []

    vetores_a = modelo.encode(frases_a, convert_to_tensor=True)
    vetores_b = modelo.encode(frases_b, convert_to_tensor=True)
    matriz = util.cos_sim(vetores_a, vetores_b)

    correspondencias = []
    for indice_a, frase_a in enumerate(frases_a):
        indice_b = int(matriz[indice_a].argmax().item())
        similaridade = float(matriz[indice_a][indice_b].item())
        correspondencias.append({
            "frase_a": frase_a,
            "frase_b": frases_b[indice_b],
            "similaridade": similaridade
        })
    return correspondencias


def analisar_relacao_textual(premissa, hipotese, modelo):
    resultado = modelo({"text": premissa, "text_pair": hipotese})
    if resultado and isinstance(resultado[0], list):
        resultado = resultado[0]
    return sorted(resultado, key=lambda item: item["score"], reverse=True)


def comparar_relatos_nli_otimizado(relato_a, relato_b, modelo_nli, modelo_semantico, limite_pares=5):
    frases_a = separar_frases(relato_a)
    frases_b = separar_frases(relato_b)
    
    if not frases_a or not frases_b:
        return []

    correspondencias = comparar_semanticamente(relato_a, relato_b, modelo_semantico)
    correspondencias_ordenadas = sorted(correspondencias, key=lambda x: x["similaridade"], reverse=True)[:limite_pares]

    resultados = []
    for item in correspondencias_ordenadas:
        fa = item["frase_a"]
        fb = item["frase_b"]
        
        resultado_ab = analisar_relacao_textual(fa, fb, modelo_nli)
        resultado_ba = analisar_relacao_textual(fb, fa, modelo_nli)
        
        resultados.append({
            "frase_a": fa,
            "frase_b": fb,
            "similaridade": item["similaridade"],
            "resultado_ab": resultado_ab,
            "resultado_ba": resultado_ba
        })
    return resultados


def salvar_caso(identificacao, contexto, relato_a, relato_b, evidencias, divergencias, perguntas):
    with conectar_banco() as conexao:
        cursor = conexao.execute("""
            INSERT INTO casos_auditoria (
                criado_em, identificacao, contexto, relato_a, relato_b, evidencias, divergencias, perguntas
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().astimezone().isoformat(timespec="seconds"),
            identificacao, contexto, relato_a, relato_b, evidencias,
            json.dumps(divergencias, ensure_ascii=False),
            json.dumps(perguntas, ensure_ascii=False)
        ))
        return cursor.lastrowid


# Inicialização do Session State
if "texto_a" not in st.session_state:
    st.session_state["texto_a"] = ""
if "texto_b" not in st.session_state:
    st.session_state["texto_b"] = ""


# --------------------------------------------------
# INTERFACE DA APLICAÇÃO
# --------------------------------------------------

aba_analise, aba_orientacoes = st.tabs(["Nova Auditoria / Entrevista", "Diretrizes Metodológicas"])

with aba_analise:
    st.subheader("1. Escopo da Investigação")

    identificacao = st.text_input("Identificação do Processo", placeholder="Ex.: AUD-2026-103 (Entrevista de Conformidade)")
    contexto = st.text_area("Contexto e Objetivo", placeholder="Ex.: Gravação e cruzamento de depoimentos colhidos em diligência.")

    st.divider()
    st.subheader("2. Depoimentos (Gravação Direta ou Texto)")

    coluna_a, coluna_b = st.columns(2)

    # --- DEPOIMENTO 01 ---
    with coluna_a:
        st.markdown("#### Depoimento 01")
        modo_a = st.selectbox("Entrada de Dados (Depoimento 1)", ["Gravar Áudio ao Vivo (Microfone)", "Digitar Texto", "Enviar Arquivo MP3/WAV"], key="sel_a")

        if modo_a == "Gravar Áudio ao Vivo (Microfone)":
            audio_gravado_a = st.audio_input("Fale o Depoimento 01 (Toque para gravar)", key="mic_a")
            if audio_gravado_a:
                st.audio(audio_gravado_a)
                if st.button("Transcrever Áudio Gravado 1", key="t_btn_a"):
                    with st.spinner("Transcrevendo com Whisper..."):
                        asr = carregar_modelo_asr()
                        st.session_state["texto_a"] = transcrever_audio(audio_gravado_a, asr)
                        st.success("Áudio 1 transcrito com sucesso!")

        elif modo_a == "Enviar Arquivo MP3/WAV":
            arq_a = st.file_uploader("Selecione o arquivo de áudio 1", type=["mp3", "wav", "m4a"], key="file_a")
            if arq_a and st.button("Transcrever Arquivo 1", key="f_btn_a"):
                with st.spinner("Transcrevendo arquivo..."):
                    asr = carregar_modelo_asr()
                    st.session_state["texto_a"] = transcrever_audio(arq_a, asr)
                    st.success("Arquivo 1 transcrito!")

        st.session_state["texto_a"] = st.text_area(
            "Texto do Depoimento 01 (Editável)", 
            value=st.session_state["texto_a"], 
            height=160, 
            key="box_a",
            placeholder="O texto transcrito aparecerá aqui ou você pode digitar diretamente..."
        )

    # --- DEPOIMENTO 02 ---
    with coluna_b:
        st.markdown("#### Depoimento 02")
        modo_b = st.selectbox("Entrada de Dados (Depoimento 2)", ["Gravar Áudio ao Vivo (Microfone)", "Digitar Texto", "Enviar Arquivo MP3/WAV"], key="sel_b")

        if modo_b == "Gravar Áudio ao Vivo (Microfone)":
            audio_gravado_b = st.audio_input("Fale o Depoimento 02 (Toque para gravar)", key="mic_b")
            if audio_gravado_b:
                st.audio(audio_gravado_b)
                if st.button("Transcrever Áudio Gravado 2", key="t_btn_b"):
                    with st.spinner("Transcrevendo com Whisper..."):
                        asr = carregar_modelo_asr()
                        st.session_state["texto_b"] = transcrever_audio(audio_gravado_b, asr)
                        st.success("Áudio 2 transcrito com sucesso!")

        elif modo_b == "Enviar Arquivo MP3/WAV":
            arq_b = st.file_uploader("Selecione o arquivo de áudio 2", type=["mp3", "wav", "m4a"], key="file_b")
            if arq_b and st.button("Transcrever Arquivo 2", key="f_btn_b"):
                with st.spinner("Transcrevendo arquivo..."):
                    asr = carregar_modelo_asr()
                    st.session_state["texto_b"] = transcrever_audio(arq_b, asr)
                    st.success("Arquivo 2 transcrito!")

        st.session_state["texto_b"] = st.text_area(
            "Texto do Depoimento 02 (Editável)", 
            value=st.session_state["texto_b"], 
            height=160, 
            key="box_b",
            placeholder="O texto transcrito aparecerá aqui ou você pode digitar diretamente..."
        )

    st.divider()
    st.subheader("3. Evidências Documentais de Apoio")
    evidencias = st.text_area("Logs, e-mails ou relatórios de referência", height=100, placeholder="Ex.: Log do sistema confirma horário da operação.")

    executar = st.button("Executar Cruzamento e Gerar Relatório", type="primary", use_container_width=True)

    if executar:
        relato_a = st.session_state["texto_a"]
        relato_b = st.session_state["texto_b"]

        if not relato_a.strip() or not relato_b.strip():
            st.error("Preencha ou grave os dois depoimentos antes de prosseguir.")
        else:
            with st.spinner("Processando similaridade, análise lógica e gerando relatório..."):
                modelo_sem = carregar_modelo_semantico()
                modelo_nli = carregar_modelo_nli()

                divergencias = comparar_relatos(relato_a, relato_b)
                correspondencias = comparar_semanticamente(relato_a, relato_b, modelo_sem)
                resultados_nli = comparar_relatos_nli_otimizado(relato_a, relato_b, modelo_nli, modelo_sem, limite_pares=5)
                
                perguntas = gerar_perguntas_confronto(divergencias, resultados_nli)

            salvar_caso(
                identificacao=identificacao.strip(),
                contexto=contexto.strip(),
                relato_a=relato_a,
                relato_b=relato_b,
                evidencias=evidencias,
                divergencias=divergencias,
                perguntas=perguntas
            )

            st.success("Auditoria processada e salva com sucesso no banco de dados!")
            st.divider()
            
            st.header("Relatório de Divergências e Inconsistências")
            if identificacao.strip():
                st.write(f"**Processo:** {identificacao}")
            if contexto.strip():
                st.write(f"**Escopo:** {contexto}")

            st.write(f"**Divergências estruturais encontradas:** {len(divergencias)}")

            if divergencias:
                for idx, item in enumerate(divergencias, start=1):
                    with st.expander(f"Divergência {idx}: {item['categoria']}", expanded=True):
                        st.write(f"**Depoimento 01:** {item['relato_a']}")
                        st.write(f"**Depoimento 02:** {item['relato_b']}")
            else:
                st.info("Nenhuma divergência numérica ou temporal explícita detectada nas categorias analisadas.")

            st.subheader("🎯 Perguntas Cirúrgicas de Confronto (Modo Inquirição)")
            for p in perguntas:
                st.markdown(f"- {p}")

            st.subheader("Análise Avançada de Relações Lógicas (NLI)")
            for item in resultados_nli:
                st.markdown("---")
                st.write("**Trecho Depoimento 01:**", item["frase_a"])
                st.write("**Trecho Depoimento 02:**", item["frase_b"])
                principal = item["resultado_ab"][0]
                st.write(f"**Relação Lógica Detectada:** {principal['label']} ({principal['score']:.1%})")

            # Botão para download do PDF
            st.divider()
            pdf_bytes = gerar_pdf_relatorio(
                identificacao=identificacao,
                contexto=contexto,
                relato_a=relato_a,
                relato_b=relato_b,
                evidencias=evidencias,
                divergencias=divergencias,
                perguntas=perguntas
            )
            
            st.download_button(
                label="📥 Baixar Relatório Completo em PDF",
                data=pdf_bytes,
                file_name=f"Relatorio_Auditoria_{identificacao.replace(' ', '_') or 'caso'}.pdf",
                mime="application/pdf",
                type="primary"
            )


with aba_orientacoes:
    st.subheader("Guia Metodológico para Uso em Campo (Celular/Tablet)")
    st.markdown("""
    1. **Gravação Direta:** Selecione **"Gravar Áudio ao Vivo (Microfone)"**, toque no botão de gravação do celular, fale o depoimento da testemunha e clique em transcrever.
    2. **Revisão Textual:** O texto gerado pelo Whisper é inserido automaticamente na caixa de texto, onde você pode corrigir pequenos detalhes se necessário.
    3. **Exportação do Relatório:** Após gerar o cruzamento, clique no botão de download para baixar o laudo em PDF para arquivo ou envio imediato.
    """)

# --------------------------------------------------
# HISTÓRICO DE CASOS
# --------------------------------------------------
st.divider()
st.header("Histórico de Casos Registrados")

with conectar_banco() as conexao:
    conexao.row_factory = sqlite3.Row
    historico = conexao.execute("SELECT * FROM casos_auditoria ORDER BY id DESC LIMIT 20").fetchall()

if not historico:
    st.info("Nenhum caso registrado no histórico.")
else:
    for caso in historico:
        with st.expander(f"#{caso['id']} — {caso['identificacao'] or 'Caso sem ID'} ({caso['criado_em']})"):
            st.write(f"**Contexto:** {caso['contexto'] or 'Não informado'}")
            st.write(f"**Depoimento 01:** {caso['relato_a']}")
            st.write(f"**Depoimento 02:** {caso['relato_b']}")
            st.write(f"**Evidências:** {caso['evidencias'] or 'Nenhuma'}")
            
            perguntas_salvas = json.loads(caso["perguntas"])
            if perguntas_salvas:
                st.write("**Perguntas de Confronto Geradas:**")
                for p in perguntas_salvas:
                    st.markdown(f"- {p}")

            # Botão de download do PDF também direto no histórico
            divergencias_salvas = json.loads(caso["divergencias"])
            pdf_bytes_hist = gerar_pdf_relatorio(
                identificacao=caso["identificacao"],
                contexto=caso["contexto"],
                relato_a=caso["relato_a"],
                relato_b=caso["relato_b"],
                evidencias=caso["evidencias"],
                divergencias=divergencias_salvas,
                perguntas=perguntas_salvas
            )
            st.download_button(
                label=f"📥 Baixar PDF (Caso #{caso['id']})",
                data=pdf_bytes_hist,
                file_name=f"Relatorio_Caso_{caso['id']}.pdf",
                mime="application/pdf",
                key=f"pdf_hist_{caso['id']}"
            )