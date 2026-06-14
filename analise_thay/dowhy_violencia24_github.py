"""

Objetivo do script

1. Carregar a base VIOLBR24 filtrada.
2. Preparar variáveis no formato exigido pelo DoWhy.
3. Selecionar uma DAG candidata.
4. Construir o modelo causal com CausalModel.
5. Identificar o efeito causal com identify_effect().
6. Estimar o ATE com Propensity Score Weighting.
7. Rodar refutadores para avaliar robustez.
8. Exportar resultados em CSV.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from dowhy import CausalModel

warnings.filterwarnings("ignore")


# =============================================================================
# 1. CONFIGURAÇÕES GERAIS
# =============================================================================

PATH_BASE = "VIOLBR24_filtrado.csv"
OUTCOME = "LES_AUTOP"
METODO_ESTIMACAO = "backdoor.propensity_score_weighting"


# =============================================================================
# 2. DAGS CANDIDATAS
# =============================================================================

# DAG candidata 1: proposta por Matheus.
# Essa DAG inclui TRAN_MENT como fator central para explicar LES_AUTOP.
DAG_MATHEUS = """
digraph {

    CS_SEXO -> ORIENT_SEX;
    CS_SEXO -> IDENT_GEN;
    CS_SEXO -> TRAN_MENT;
    CS_SEXO -> OUT_VEZES;
    CS_SEXO -> LES_AUTOP;

    NU_IDADE_N -> CS_ESCOL_N;
    NU_IDADE_N -> SIT_CONJUG;
    NU_IDADE_N -> TRAN_MENT;
    NU_IDADE_N -> OUT_VEZES;
    NU_IDADE_N -> LES_AUTOP;

    CS_ESCOL_N -> SIT_CONJUG;
    CS_ESCOL_N -> TRAN_MENT;
    CS_ESCOL_N -> OUT_VEZES;
    CS_ESCOL_N -> LES_AUTOP;

    SIT_CONJUG -> TRAN_MENT;
    SIT_CONJUG -> OUT_VEZES;
    SIT_CONJUG -> LES_AUTOP;

    ORIENT_SEX -> TRAN_MENT;
    ORIENT_SEX -> LES_AUTOP;

    IDENT_GEN -> TRAN_MENT;
    IDENT_GEN -> LES_AUTOP;

    OUT_VEZES -> TRAN_MENT;
    OUT_VEZES -> LES_AUTOP;

    DEF_MENTAL -> TRAN_MENT;
    DEF_MENTAL -> LES_AUTOP;

    TRAN_MENT -> LES_AUTOP;
    TRAN_COMP -> LES_AUTOP;
}
"""


# DAG candidata 2: proposta por Haniel.
# Essa DAG enfatiza variáveis contextuais, como UF, local, horário e ocupação.
DAG_HANIEL = """
digraph {

    SG_UF -> deficiencia;
    SG_UF -> IDENT_GEN;
    SG_UF -> LES_AUTOP;

    SIT_CONJUG -> deficiencia;
    SIT_CONJUG -> HORA_OCOR;
    SIT_CONJUG -> OUT_VEZES;
    SIT_CONJUG -> LES_AUTOP;

    CS_SEXO -> metodo;
    CS_SEXO -> OUT_VEZES;
    CS_SEXO -> LES_AUTOP;

    CS_RACA -> LOCAL_OCOR;
    CS_RACA -> ID_OCUPA_N;

    ORIENT_SEX -> LOCAL_OCOR;
    ORIENT_SEX -> ID_OCUPA_N;
    ORIENT_SEX -> LES_AUTOP;

    HORA_OCOR -> ID_OCUPA_N;
    HORA_OCOR -> LES_AUTOP;

    CS_ESCOL_N -> NU_IDADE_N;
    CS_ESCOL_N -> ID_OCUPA_N;
    CS_ESCOL_N -> LES_AUTOP;

    NU_IDADE_N -> ID_OCUPA_N;
    NU_IDADE_N -> LES_AUTOP;

    deficiencia -> IDENT_GEN;
    deficiencia -> OUT_VEZES;
    deficiencia -> LES_AUTOP;

    IDENT_GEN -> LES_AUTOP;

    metodo -> OUT_VEZES;
    metodo -> LES_AUTOP;

    LOCAL_OCOR -> LES_AUTOP;

    OUT_VEZES -> LES_AUTOP;

    ID_OCUPA_N -> LES_AUTOP;
}
"""


@dataclass
class ConfigDAG:

    nome: str
    graph: str
    colunas_modelo: List[str]
    tratamentos_ate: List[str]
    tratamentos_refutadores: List[str]
    out_base: str
    out_ate: str
    out_refutadores: str


CONFIG_MATHEUS = ConfigDAG(
    nome="dag_mateus",
    graph=DAG_MATHEUS,
    colunas_modelo=[
        "LES_AUTOP",
        "TRAN_MENT",
        "TRAN_COMP",
        "DEF_MENTAL",
        "OUT_VEZES",
        "ORIENT_SEX",
        "IDENT_GEN",
        "CS_SEXO",
        "NU_IDADE_N",
        "CS_ESCOL_N",
        "SIT_CONJUG",
    ],
    tratamentos_ate=["TRAN_MENT", "OUT_VEZES", "ORIENT_SEX", "IDENT_GEN"],
    tratamentos_refutadores=["TRAN_MENT", "OUT_VEZES"],
    out_base="base_modelagem_dag_mateus.csv",
    out_ate="resultados_ate_dag_mateus.csv",
    out_refutadores="resultados_refutadores_dag_mateus.csv",
)

CONFIG_HANIEL = ConfigDAG(
    nome="dag_haniel",
    graph=DAG_HANIEL,
    colunas_modelo=[
        "LES_AUTOP",
        "OUT_VEZES",
        "ORIENT_SEX",
        "IDENT_GEN",
        "CS_SEXO",
        "NU_IDADE_N",
        "CS_ESCOL_N",
        "SIT_CONJUG",
        "SG_UF",
        "CS_RACA",
        "HORA_OCOR",
        "LOCAL_OCOR",
        "ID_OCUPA_N",
        "deficiencia",
        "metodo",
    ],
    tratamentos_ate=["OUT_VEZES", "ORIENT_SEX", "IDENT_GEN", "deficiencia", "metodo"],
    tratamentos_refutadores=["OUT_VEZES", "IDENT_GEN", "deficiencia"],
    out_base="base_modelagem_dag_haniel.csv",
    out_ate="resultados_ate_dag_haniel.csv",
    out_refutadores="resultados_refutadores_dag_haniel.csv",
)


# =============================================================================
# 3. FUNÇÕES DE PRÉ-PROCESSAMENTO
# =============================================================================

def mapear_binaria_sim_nao(serie: pd.Series) -> pd.Series:
    """
    Converte variáveis do SINAN codificadas como 1=Sim e 2=Não para 1/0.

    Códigos diferentes de 1 e 2 viram NaN. Isso é importante para evitar que
    categorias como 8=Não se aplica ou 9=Ignorado sejam interpretadas como
    informação válida no modelo causal.
    """

    return serie.map({1: 1, 2: 0})


def converter_idade_sinan(valor) -> float:
    """
    Converte a idade do formato SINAN para anos.

    No SINAN, idades em anos geralmente aparecem como 4xxx.
    Exemplo: 4033 significa 33 anos.
    """

    if pd.notna(valor) and 4000 <= valor < 5000:
        return valor - 4000
    return np.nan


def preparar_base(path: str = PATH_BASE) -> pd.DataFrame:

    df = pd.read_csv(path)

    df = df[df[OUTCOME].isin([1, 2])].copy()
    df[OUTCOME] = df[OUTCOME].map({1: 1, 2: 0})

    # Conversão da idade para anos.
    if "NU_IDADE_N" in df.columns:
        df["NU_IDADE_N"] = df["NU_IDADE_N"].apply(converter_idade_sinan)

    # Variáveis do SINAN codificadas como 1=Sim e 2=Não.
    colunas_sim_nao = [
        "TRAN_MENT",
        "TRAN_COMP",
        "DEF_MENTAL",
        "DEF_FISICA",
        "DEF_VISUAL",
        "DEF_AUDITI",
        "OUT_VEZES",
        "AG_ENVEN",
        "AG_FOGO",
        "AG_CORTE",
        "AG_OBJETO",
        "AG_ENFOR",
    ]

    for col in colunas_sim_nao:
        if col in df.columns:
            df[col] = mapear_binaria_sim_nao(df[col])

    # Sexo: F=0 e M=1. Outros valores ficam como NaN.
    if "CS_SEXO" in df.columns:
        df["CS_SEXO"] = df["CS_SEXO"].map({"F": 0, "M": 1})

    # Orientação sexual: 0=heterossexual; 1=minoria sexual.
    # Codificação usada: 1=heterossexual, 2=homossexual, 3=bissexual.
    if "ORIENT_SEX" in df.columns:
        df["ORIENT_SEX"] = df["ORIENT_SEX"].map({1: 0, 2: 1, 3: 1})

    # Identidade de gênero: 0=cis presumido/não se aplica; 1=minoria de gênero.
    # Codificação usada: 8=não se aplica; 1,2,3=identidades trans/travesti.
    if "IDENT_GEN" in df.columns:
        df["IDENT_GEN"] = df["IDENT_GEN"].map({8: 0, 1: 1, 2: 1, 3: 1})

    # Variável agregada de deficiência.
    # Recebe 1 se qualquer deficiência considerada tiver sido marcada como Sim.
    df["deficiencia"] = (
        (df.get("DEF_FISICA", 0) == 1)
        | (df.get("DEF_MENTAL", 0) == 1)
        | (df.get("DEF_VISUAL", 0) == 1)
        | (df.get("DEF_AUDITI", 0) == 1)
    ).astype(int)

    # Variável agregada de método de agressão.
    # Recebe 1 se qualquer meio de agressão tiver sido marcado como Sim.
    df["metodo"] = (
        (df.get("AG_ENVEN", 0) == 1)
        | (df.get("AG_FOGO", 0) == 1)
        | (df.get("AG_CORTE", 0) == 1)
        | (df.get("AG_OBJETO", 0) == 1)
        | (df.get("AG_ENFOR", 0) == 1)
    ).astype(int)


    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes

    return df


def selecionar_base_modelagem(df: pd.DataFrame, config: ConfigDAG) -> pd.DataFrame:

    colunas_disponiveis = [col for col in config.colunas_modelo if col in df.columns]
    df_modelo = df[colunas_disponiveis].dropna().copy()
    return df_modelo


# =============================================================================
# 4. FUNÇÕES DE DIAGNÓSTICO
# =============================================================================

def diagnosticar_base(df: pd.DataFrame, nome_dag: str) -> None:

    print("\n" + "=" * 70)
    print(f"DIAGNÓSTICO DA BASE - {nome_dag}")
    print("=" * 70)
    print("Shape:", df.shape)

    print("\nDistribuição do outcome LES_AUTOP:")
    print(df[OUTCOME].value_counts())
    print(df[OUTCOME].value_counts(normalize=True))

    print("\nMédias das variáveis numéricas:")
    print(df.mean(numeric_only=True).sort_values(ascending=False))


def validar_tratamento(df: pd.DataFrame, treatment: str) -> Optional[str]:

    if treatment not in df.columns:
        return "Tratamento não encontrado na base de modelagem."

    if df[treatment].nunique(dropna=True) != 2:
        return "Tratamento não binário ou sem variação suficiente."

    return None


# =============================================================================
# 5. ESTIMAÇÃO CAUSAL COM DOWHY
# =============================================================================

def estimar_ate(
    df: pd.DataFrame,
    treatment: str,
    graph: str,
    outcome: str = OUTCOME,
    metodo: str = METODO_ESTIMACAO,
) -> Dict[str, object]:

    erro_validacao = validar_tratamento(df, treatment)
    if erro_validacao is not None:
        return {
            "tratamento": treatment,
            "outcome": outcome,
            "metodo": metodo,
            "ate": np.nan,
            "status": "erro",
            "erro": erro_validacao,
        }

    print("\n" + "-" * 70)
    print(f"Estimando ATE: {treatment} -> {outcome}")
    print(f"Método: {metodo}")

    try:
        model = CausalModel(
            data=df,
            treatment=treatment,
            outcome=outcome,
            graph=graph,
        )

        estimand = model.identify_effect(proceed_when_unidentifiable=True)

        estimate = model.estimate_effect(
            estimand,
            method_name=metodo,
        )

        ate = float(estimate.value)
        print(f"ATE estimado: {ate:.6f}")

        return {
            "tratamento": treatment,
            "outcome": outcome,
            "metodo": metodo,
            "ate": ate,
            "status": "ok",
            "erro": None,
        }

    except Exception as exc:
        print(f"Erro ao estimar {treatment}: {exc}")
        return {
            "tratamento": treatment,
            "outcome": outcome,
            "metodo": metodo,
            "ate": np.nan,
            "status": "erro",
            "erro": str(exc),
        }


def rodar_ates(df: pd.DataFrame, config: ConfigDAG) -> pd.DataFrame:

    resultados = []
    for treatment in config.tratamentos_ate:
        resultados.append(estimar_ate(df=df, treatment=treatment, graph=config.graph))
    return pd.DataFrame(resultados)


# =============================================================================
# 6. TESTES DE ROBUSTEZ / REFUTADORES
# =============================================================================

def rodar_refutadores(
    df: pd.DataFrame,
    treatment: str,
    graph: str,
    outcome: str = OUTCOME,
    metodo: str = METODO_ESTIMACAO,
) -> List[Dict[str, object]]:
    """
    Roda refutadores do DoWhy para verificar robustez do efeito estimado.

    Refutadores usados:
    - random_common_cause: adiciona um confundidor aleatório e reestima o efeito.
    - data_subset_refuter: reestima o efeito em subconjuntos aleatórios dos dados.

    Se o ATE refutado for muito parecido com o ATE original, temos evidência de
    estabilidade da estimativa.
    """

    erro_validacao = validar_tratamento(df, treatment)
    if erro_validacao is not None:
        return [
            {
                "tratamento": treatment,
                "ate_original": np.nan,
                "refutador": None,
                "resultado": None,
                "status": "erro",
                "erro": erro_validacao,
            }
        ]

    print("\n" + "-" * 70)
    print(f"Rodando refutadores: {treatment} -> {outcome}")

    try:
        model = CausalModel(
            data=df,
            treatment=treatment,
            outcome=outcome,
            graph=graph,
        )

        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(estimand, method_name=metodo)
        ate_original = float(estimate.value)

        print(f"ATE original: {ate_original:.6f}")

        resultados = []
        refutadores = ["random_common_cause", "data_subset_refuter"]

        for refutador in refutadores:
            try:
                refute = model.refute_estimate(
                    estimand,
                    estimate,
                    method_name=refutador,
                )

                print(f"Refutador {refutador}: OK")

                resultados.append(
                    {
                        "tratamento": treatment,
                        "ate_original": ate_original,
                        "refutador": refutador,
                        "resultado": str(refute),
                        "status": "ok",
                        "erro": None,
                    }
                )

            except Exception as exc:
                print(f"Refutador {refutador} falhou: {exc}")
                resultados.append(
                    {
                        "tratamento": treatment,
                        "ate_original": ate_original,
                        "refutador": refutador,
                        "resultado": None,
                        "status": "erro",
                        "erro": str(exc),
                    }
                )

        return resultados

    except Exception as exc:
        print(f"Erro geral nos refutadores de {treatment}: {exc}")
        return [
            {
                "tratamento": treatment,
                "ate_original": np.nan,
                "refutador": None,
                "resultado": None,
                "status": "erro",
                "erro": str(exc),
            }
        ]


def rodar_todos_refutadores(df: pd.DataFrame, config: ConfigDAG) -> pd.DataFrame:

    todos_resultados = []
    for treatment in config.tratamentos_refutadores:
        todos_resultados.extend(
            rodar_refutadores(df=df, treatment=treatment, graph=config.graph)
        )
    return pd.DataFrame(todos_resultados)


# =============================================================================
# 7. EXECUÇÃO DO PIPELINE COMPLETO
# =============================================================================

def executar_pipeline(df_original: pd.DataFrame, config: ConfigDAG) -> None:

    print("\n" + "#" * 70)
    print(f"EXECUTANDO PIPELINE PARA: {config.nome.upper()}")
    print("#" * 70)

    df_modelo = selecionar_base_modelagem(df_original, config)
    diagnosticar_base(df_modelo, config.nome)

    df_modelo.to_csv(config.out_base, index=False)
    print(f"\nBase de modelagem salva em: {config.out_base}")

    # Estima os efeitos causais médios.
    resultados_ate = rodar_ates(df_modelo, config)
    resultados_ate.to_csv(config.out_ate, index=False)
    print(f"Resultados de ATE salvos em: {config.out_ate}")

    # Roda refutadores para os efeitos principais.
    resultados_refutadores = rodar_todos_refutadores(df_modelo, config)
    resultados_refutadores.to_csv(config.out_refutadores, index=False)
    print(f"Resultados dos refutadores salvos em: {config.out_refutadores}")

    print("\nResumo dos ATEs:")
    print(resultados_ate[["tratamento", "outcome", "ate", "status", "erro"]])


# =============================================================================
# 8. MAIN
# =============================================================================

def main() -> None:

    print("Carregando e preparando base...")
    df_original = preparar_base(PATH_BASE)

    executar_pipeline(df_original, CONFIG_MATHEUS)
    executar_pipeline(df_original, CONFIG_HANIEL)

    print("\nProcesso finalizado.")


if __name__ == "__main__":
    main()
