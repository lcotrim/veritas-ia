
import re
from datetime import datetime
import sqlite3
import json
from pathlib import Path

import streamlit as st
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline


# Caminho do banco de dados local
PASTA_PROJETO = Path(__file__).resolve().parent
ARQUIVO_BANCO = PASTA_PROJETO / "veritas.db"


def conectar_banco():
    """Abre uma conexão com o banco SQLite."""
    return sqlite3.connect(ARQUIVO_BANCO)


def preparar_banco():
    """Cria a tabela de casos se ela ainda não existir."""

    with conectar_banco() as conexao:
        conexao.execute("""
            CREATE TABLE IF NOT EXISTS casos (
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
        # Adiciona a coluna de análise NLI caso ainda não exista
        colunas = conexao.execute(
            "PRAGMA table_info(casos)"
        ).fetchall()

        nomes_colunas = [coluna[1] for coluna in colunas]

        if "resultado_nli" not in nomes_colunas:
            conexao.execute(
                "ALTER TABLE casos ADD COLUMN resultado_nli TEXT"
            )


def salvar_caso(
    identificacao,
    contexto,
    relato_a,
    relato_b,
    evidencias,
    divergencias,
    perguntas
):
    """Grava uma análise no banco de dados."""

    with conectar_banco() as conexao:
        conexao.execute("""
            INSERT INTO casos (
                criado_em,
                identificacao,
                contexto,
                relato_a,
                relato_b,
                evidencias,
                divergencias,
                perguntas
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            identificacao,
            contexto,
            relato_a,
            relato_b,
            evidencias,
            json.dumps(
                divergencias,
                ensure_ascii=False
            ),
            json.dumps(
                perguntas,
                ensure_ascii=False
            )
        ))


# Prepara o banco ao iniciar a aplicação
preparar_banco()


# --------------------------------------------------
# CONFIGURACAO DA APLICACAO
# --------------------------------------------------

st.set_page_config(
    page_title="Veritas IA",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Veritas IA")
st.caption("Análise de declarações, divergências e evidências")

st.info(
    "Esta ferramenta identifica diferenças e ajuda a formular "
    "perguntas. Ela não determina se uma pessoa está mentindo."
)


# --------------------------------------------------
# FUNCAO 1: EXTRAIR INFORMACOES DO TEXTO
# --------------------------------------------------

def extrair_dados(texto):
    """
    Procura datas, horarios e numeros explicitamente
    escritos no relato.

    Esta funcao utiliza regras, nao uma IA treinada.
    """

    return {
        "Datas": re.findall(
            r"\b\d{1,2}/\d{1,2}/(?:\d{2}|\d{4})\b",
            texto
        ),
        "Horarios": re.findall(
            r"\b\d{1,2}:\d{2}\b",
            texto
        ),
        "Numeros": re.findall(
            r"\b\d+(?:[.,]\d+)?\b",
            texto
        )
    }


# --------------------------------------------------
# FUNCAO 2: COMPARAR OS RELATOS
# --------------------------------------------------

def comparar_relatos(relato_a, relato_b):

    dados_a = extrair_dados(relato_a)
    dados_b = extrair_dados(relato_b)

    divergencias = []

    for categoria in dados_a:

        valores_a = dados_a[categoria]
        valores_b = dados_b[categoria]

        # Compara somente categorias presentes
        # nos dois relatos.
        if valores_a and valores_b:

            if valores_a != valores_b:

                divergencias.append({
                    "categoria": categoria,
                    "relato_a": valores_a,
                    "relato_b": valores_b
                })

    return divergencias


# --------------------------------------------------
# FUNCAO 3: GERAR PERGUNTAS DE ESCLARECIMENTO
# --------------------------------------------------

def gerar_perguntas(divergencias):

    perguntas = []

    for item in divergencias:

        categoria = item["categoria"]

        if categoria == "Datas":
            perguntas.append(
                "Qual foi a data exata do acontecimento? "
                "Existe algum registro que permita confirmá-la?"
            )

        elif categoria == "Horarios":
            perguntas.append(
                "Qual foi o horário aproximado ou exato? "
                "Há algum registro com essa informação?"
            )

        elif categoria == "Numeros":
            perguntas.append(
                "A que se referem os números diferentes "
                "encontrados nos relatos? Podemos verificar "
                "documentos ou registros relacionados?"
            )

    return list(dict.fromkeys(perguntas))


# --------------------------------------------------
# FUNCAO 4: VALIDAR DATAS INFORMADAS
# --------------------------------------------------

def validar_data(data_texto):

    formatos = ["%d/%m/%Y", "%d/%m/%y"]

    for formato in formatos:
        try:
            return datetime.strptime(data_texto, formato)
        except ValueError:
            pass

    return None


# --------------------------------------------------
# ANALISE SEMANTICA LOCAL
# --------------------------------------------------

@st.cache_resource
def carregar_modelo_semantico():
    """
    Carrega o modelo uma vez e reutiliza na aplicacao.
    O download inicial exige acesso a internet.
    """

    return SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )


def separar_frases(texto):
    """
    Divide um texto em frases para permitir comparacoes.
    """

    frases = re.split(r"(?<=[.!?])\s+|\n+", texto)

    return [
        frase.strip()
        for frase in frases
        if frase.strip()
    ]


def comparar_semanticamente(texto_a, texto_b, modelo):
    """
    Localiza frases semanticamente semelhantes.
    Similaridade nao significa concordancia ou contradicao.
    """

    frases_a = separar_frases(texto_a)
    frases_b = separar_frases(texto_b)

    if not frases_a or not frases_b:
        return []

    vetores_a = modelo.encode(
        frases_a,
        convert_to_tensor=True
    )

    vetores_b = modelo.encode(
        frases_b,
        convert_to_tensor=True
    )

    matriz = util.cos_sim(vetores_a, vetores_b)

    correspondencias = []

    for indice_a, frase_a in enumerate(frases_a):

        # Localiza a frase mais semelhante do outro relato.
        indice_b = int(
            matriz[indice_a].argmax().item()
        )

        similaridade = float(
            matriz[indice_a][indice_b].item()
        )

        correspondencias.append({
            "frase_a": frase_a,
            "frase_b": frases_b[indice_b],
            "similaridade": similaridade
        })

    return correspondencias


# --------------------------------------------------
# MODELO EXPERIMENTAL DE INFERENCIA TEXTUAL (NLI)
# --------------------------------------------------

@st.cache_resource
def carregar_modelo_nli():
    """
    Carrega um modelo multilíngue para comparar
    duas afirmações.

    A primeira frase sera a premissa.
    A segunda sera a hipotese analisada.
    """

    return pipeline(
        "text-classification",
        model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
        top_k=None
    )


def analisar_relacao_textual(premissa, hipotese, modelo):
    """
    Retorna as classificacoes produzidas pelo modelo.
    Elas sao estimativas, nao provas.
    """

    resultado = modelo({
        "text": premissa,
        "text_pair": hipotese
    })

    # Algumas versoes da biblioteca retornam
    # uma lista dentro de outra lista.
    if resultado and isinstance(resultado[0], list):
        resultado = resultado[0]

    return sorted(
        resultado,
        key=lambda item: item["score"],
        reverse=True
    )

def comparar_relatos_nli_otimizado(relato_a, relato_b, modelo_nli, modelo_semantico, limite_pares=5):
    """
    Compara apenas os pares mais relevantes semanticamente usando NLI,
    evitando o loop cartesiano completo de O(N*M) que trava a interface.
    """
    frases_a = separar_frases(relato_a)
    frases_b = separar_frases(relato_b)
    
    if not frases_a or not frases_b:
        return []

    # Utiliza o modelo semântico para encontrar as correspondências mais fortes
    correspondencias = comparar_semanticamente(relato_a, relato_b, modelo_semantico)
    
    # Seleciona os N pares com maior similaridade semântica
    correspondencias_ordenadas = sorted(
        correspondencias, 
        key=lambda x: x["similaridade"], 
        reverse=True
    )[:limite_pares]

    resultados = []
    for item in correspondencias_ordenadas:
        frase_a = item["frase_a"]
        frase_b = item["frase_b"]
        
        # Executa NLI apenas nos pares filtrados
        resultado_ab = analisar_relacao_textual(frase_a, frase_b, modelo_nli)
        resultado_ba = analisar_relacao_textual(frase_b, frase_a, modelo_nli)
        
        resultados.append({
            "frase_a": frase_a,
            "frase_b": frase_b,
            "similaridade": item["similaridade"],
            "resultado_ab": resultado_ab,
            "resultado_ba": resultado_ba
        })

    return resultados


# --------------------------------------------------
# INTERFACE PRINCIPAL
# --------------------------------------------------

aba_analise, aba_orientacoes = st.tabs([
    "Analisar caso",
    "Como interpretar os resultados"
])

with aba_analise:

    st.subheader("1. Identificação do caso")

    identificacao = st.text_input(
        "Nome ou código do caso",
        placeholder="Ex.: Apuração interna 001"
    )

    contexto = st.text_area(
        "O que está sendo investigado?",
        placeholder=(
            "Ex.: Verificar a data de entrega de um equipamento "
            "mencionada em dois relatos."
        )
    )

    st.divider()

    st.subheader("2. Declarações")

    coluna_a, coluna_b = st.columns(2)

    with coluna_a:
        relato_a = st.text_area(
            "Declaração A",
            height=180,
            placeholder=(
                "Ex.: Entreguei o equipamento em "
                "12/10/2026 às 10:00."
            )
        )

    with coluna_b:
        relato_b = st.text_area(
            "Declaração B",
            height=180,
            placeholder=(
                "Ex.: Entreguei o equipamento em "
                "13/10/2026 às 15:00."
            )
        )

    st.divider()

    st.subheader("3. Evidências disponíveis")

    evidencias = st.text_area(
        "Documentos, registros ou outras informações",
        height=150,
        placeholder=(
            "Ex.: O protocolo de recebimento registra "
            "a data 12/10/2026. Informar a origem do registro."
        )
    )

    st.caption(
        "Informe a fonte e o que ela demonstra. "
        "Este protótipo ainda não verifica automaticamente "
        "a autenticidade dos documentos."
    )

    executar = st.button(
        "Analisar caso",
        type="primary",
        use_container_width=True
    )

    if executar:

        if not relato_a.strip() or not relato_b.strip():

            st.error(
                "Preencha as duas declarações antes de analisar."
            )

        else:

            divergencias = comparar_relatos(
                relato_a,
                relato_b
            )

            perguntas = gerar_perguntas(divergencias)
            
            # Executa a comparacao semantica local
            # Executa a comparacao semantica local e NLI otimizado
            with st.spinner(
                "Analisando significado e relações textuais..."
            ):
                modelo_sem = carregar_modelo_semantico()
                modelo_nli = carregar_modelo_nli()

                correspondencias = comparar_semanticamente(
                    relato_a,
                    relato_b,
                    modelo_sem
                )
                
                resultados_nli = comparar_relatos_nli_otimizado(
                    relato_a,
                    relato_b,
                    modelo_nli,
                    modelo_sem,
                    limite_pares=5
                )
            
            # Salva esta análise no banco local
            salvar_caso(
                identificacao=identificacao.strip(),
                contexto=contexto.strip(),
                relato_a=relato_a,
                relato_b=relato_b,
                evidencias=evidencias,
                divergencias=divergencias,
                perguntas=perguntas
            )

            st.success("Caso salvo no banco de dados local.")

            st.divider()
            st.header("Relatório da análise")

            if identificacao.strip():
                st.write(f"**Caso:** {identificacao}")

            if contexto.strip():
                st.write(f"**Objetivo:** {contexto}")

            st.write(
                f"**Divergências identificadas:** "
                f"{len(divergencias)}"
            )

            if divergencias:

                st.warning(
                    "Foram encontrados valores diferentes. "
                    "Eles precisam ser interpretados no contexto "
                    "antes de qualquer conclusão."
                )

                for indice, item in enumerate(
                    divergencias, start=1
                ):

                    with st.expander(
                        f"{indice}. {item['categoria']}",
                        expanded=True
                    ):

                        st.write(
                            f"**Declaração A:** "
                            f"{item['relato_a']}"
                        )

                        st.write(
                            f"**Declaração B:** "
                            f"{item['relato_b']}"
                        )

                        st.write(
                            "Verifique se os valores se referem "
                            "ao mesmo acontecimento ou informação."
                        )

            else:

                st.info(
                    "As regras atuais não encontraram diferenças "
                    "explícitas nas categorias comparadas. "
                    "Isso não significa que os relatos foram "
                    "confirmados."
                )

            st.subheader("Perguntas para esclarecer")

            if perguntas:

                for pergunta in perguntas:
                    st.markdown(f"- {pergunta}")

            else:

                st.write(
                    "Examine o objetivo do caso e formule "
                    "perguntas específicas para as afirmações "
                    "que ainda precisam ser verificadas."
                )

            st.subheader("Evidências registradas")

            if evidencias.strip():

                st.write(evidencias)

                st.caption(
                    "A evidência foi registrada como informação "
                    "fornecida pelo usuário; o sistema ainda "
                    "não verificou sua autenticidade ou força."
                )

            else:

                st.write(
                    "Nenhuma evidência foi informada. "
                    "Considere quais registros independentes "
                    "poderiam esclarecer o caso."
                )

            st.divider()
            
            
            st.subheader("Correspondencias semanticas")

            st.caption(
                "O modelo identifica frases semelhantes. "
                "Isso nao significa que sejam verdadeiras, "
                "falsas ou contraditorias."
            )

            if correspondencias:

                for indice, item in enumerate(
                    correspondencias, start=1
                ):

                    with st.expander(
                        f"Correspondencia {indice} — "
                        f"similaridade {item['similaridade']:.2f}"
                    ):

                        st.write("**Frase do relato A:**")
                        st.write(item["frase_a"])

                        st.write("**Frase mais semelhante do relato B:**")
                        st.write(item["frase_b"])

            else:

                st.info(
                    "Nao foi possivel comparar as frases "
                    "informadas."
                )

            # Análise de relações entre os relatos usando NLI
            st.subheader("Análise de contradições com IA")

            with st.spinner("Comparando as afirmações dos relatos..."):

                modelo_nli = carregar_modelo_nli()

                resultados_nli = comparar_relatos_nli(
                    relato_a,
                    relato_b,
                    modelo_nli
                )

            # Mostra os resultados da comparação
            for item in resultados_nli:

                st.markdown("---")

                st.write("**Relato A:**", item["frase_a"])
                st.write("**Relato B:**", item["frase_b"])

                # Apresenta a classificação principal de A para B
                principal = item["resultado_ab"][0]

                st.write(
                    "**Relação identificada:**",
                    principal["label"]
                )

                st.write(
                    "**Pontuação do modelo:**",
                    f'{principal["score"]:.1%}'
                )

                # Mostra também a comparação inversa
                inversa = item["resultado_ba"][0]

                st.caption(
                    f'Comparação inversa B → A: '
                    f'{inversa["label"]} '
                    f'({inversa["score"]:.1%})'
                )

            st.info(
                "A IA identifica relações linguísticas que merecem revisão. "
                "Uma contradição textual não comprova mentira ou culpa."
            )

            st.subheader("Conclusão provisória")

            st.write(
                "Este relatório descreve diferenças textuais "
                "e pontos que merecem verificação. Não determina "
                "intenção, culpa, traição ou mentira."
            )


with aba_orientacoes:

    st.subheader("Como utilizar o Veritas IA")

    st.markdown("""
    **1. Registre o objetivo**

    Defina qual fato precisa ser esclarecido.

    **2. Insira os relatos**

    Registre as versões sem alterar suas palavras
    para favorecer uma conclusão.

    **3. Informe as evidências**

    Descreva a origem de cada documento ou registro
    e o que ele efetivamente demonstra.

    **4. Investigue as diferenças**

    Pergunte se os relatos se referem ao mesmo evento,
    se houve correção ou se existe outra explicação.

    **5. Registre o que ficou esclarecido**

    Diferencie fatos confirmados, divergências ainda
    não resolvidas e informações sem evidência suficiente.

    **Limitação técnica**

    Esta versão utiliza expressões regulares para localizar
    datas, horários e números. Ainda não possui compreensão
    semântica, análise automática de documentos ou um modelo
    de Machine Learning treinado.
    """)


# --------------------------------------------------
# HISTORICO DE CASOS
# --------------------------------------------------

st.divider()
st.header("Histórico de casos")


def listar_casos():
    """Retorna os 20 casos mais recentes."""

    with conectar_banco() as conexao:
        conexao.row_factory = sqlite3.Row

        return conexao.execute("""
            SELECT *
            FROM casos
            ORDER BY id DESC
            LIMIT 20
        """).fetchall()


casos = listar_casos()

if not casos:

    st.info(
        "Nenhum caso foi salvo ainda. "
        "Faça uma análise para começar o histórico."
    )

else:

    st.caption(
        f"Exibindo os {len(casos)} casos mais recentes."
    )

    for caso in casos:

        titulo = (
            caso["identificacao"]
            or f"Caso {caso['id']}"
        )

        with st.expander(
            f"#{caso['id']} — {titulo} "
            f"({caso['criado_em']})"
        ):

            st.write("**Contexto:**")
            st.write(caso["contexto"] or "Não informado")

            st.write("**Declaração A:**")
            st.write(caso["relato_a"])

            st.write("**Declaração B:**")
            st.write(caso["relato_b"])

            st.write("**Evidências registradas:**")
            st.write(
                caso["evidencias"] or "Nenhuma informada"
            )

            st.write("**Divergências registradas:**")

            divergencias_salvas = json.loads(
                caso["divergencias"]
            )

            if divergencias_salvas:
                st.json(divergencias_salvas)
            else:
                st.write("Nenhuma divergência detectada.")

            st.write("**Perguntas sugeridas:**")

            perguntas_salvas = json.loads(
                caso["perguntas"]
            )

            if perguntas_salvas:
                for pergunta in perguntas_salvas:
                    st.markdown(f"- {pergunta}")
            else:
                st.write("Nenhuma pergunta foi gerada.")



# --------------------------------------------------
# LABORATORIO DE CONTRADICOES
# --------------------------------------------------

st.divider()
st.header("Laboratório de inferência textual")

st.write(
    "Compare duas afirmações e observe como o modelo "
    "classifica a relação entre elas."
)

premissa = st.text_area(
    "Afirmação de referência",
    key="nli_premissa",
    placeholder="Ex.: O pagamento foi realizado."
)

hipotese = st.text_area(
    "Afirmação para comparar",
    key="nli_hipotese",
    placeholder="Ex.: O pagamento não foi realizado."
)

if st.button("Analisar relação entre afirmações"):

    if not premissa.strip() or not hipotese.strip():

        st.error("Preencha as duas afirmações.")

    else:

        try:
            with st.spinner(
                "Carregando o modelo e analisando as frases..."
            ):

                modelo_nli = carregar_modelo_nli()

                resultados_nli = analisar_relacao_textual(
                    premissa,
                    hipotese,
                    modelo_nli
                )

            st.subheader("Classificações do modelo")

            for resultado in resultados_nli:

                rotulo = resultado["label"]
                pontuacao = resultado["score"]

                st.write(
                    f"**{rotulo}** — "
                    f"{pontuacao:.1%} de pontuação do modelo"
                )

            st.caption(
                "Essas pontuações são escores do classificador, "
                "não probabilidades de mentira ou de veracidade. "
                "O modelo pode errar, especialmente em português."
            )

            st.info(
                "Use o resultado como uma hipótese para revisão. "
                "Confirme o contexto e consulte evidências "
                "independentes antes de tirar conclusões."
            )

        except Exception as erro:

            st.error(
                "Não foi possível executar o modelo. "
                "Verifique a conexão, o download e as dependências."
            )

            st.code(str(erro))
