
import numpy as np
import pandas as pd

from causallearn.search.ConstraintBased.PC import pc


from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode

df = pd.read_csv("VIOLBR24_filtrado.csv")
df.columns

map_lesao = {1: 1, 2: 0, 9: pd.NA}
def mapear_idade(code):
    if code > 4000:
        return code - 4000
    elif code > 3000:
        return (code - 3000)/12
    else:
        return 0
map_raca = {
    1: "Branca",
    2: "Preta",
    3: "Amarela",
    4: "Parda",
    5: "Indígena",
    9: pd.NA
}
map_escola = {
    1: "F1 incompleto",
    2: "F1 completo",
    3: "F2 incompleto",
    4: "F2 completo",
    5: "EM incompleto",
    6: "EM completo",
    7: "SP incompleto",
    8: "SP completo",
    9: pd.NA,
    10: "Nao se aplica"
}
map_conjugal = {
    1: "Solteira",
    2: "Casada",
    3: "Viúva",
    4: "Divorciada",
    8: "Não se aplica",
    9: pd.NA
}
map_orientacao = {
    1: "Heterossexual",
    2: "Homossexual",
    3: "Bissexual",
    8: "Não se aplica",
    9: pd.NA
}
map_identidade = {
    1: "Travesti",
    2: "Mulher trans",
    3: "Homem trans",
    8: "Não se aplica",
    9: pd.NA
}
map_deficiencias = {
    1: 1,
    2: 0,
    8: 0,
    9: pd.NA
}
def hora(code):
    try:
        codeli = code.split(":")
        return int(codeli[0]) + int(codeli[1])/60
    except:
        print(f"Erro ao processar hora {code}")
        return pd.NA
    
def aplicar_mapas(df):
    df['HORA_OCOR'] = df['HORA_OCOR'].apply(hora)
    df['NU_IDADE_N'] = df['NU_IDADE_N'].apply(mapear_idade)
    df['LES_AUTOP'] = df['LES_AUTOP'].map(map_lesao)
    df['CS_RACA'] = df['CS_RACA'].map(map_raca)
    df['CS_ESCOL_N'] = df['CS_ESCOL_N'].map(map_escola)
    df['SIT_CONJUG'] = df['SIT_CONJUG'].map(map_conjugal)
    df['ORIENT_SEX'] = df['ORIENT_SEX'].map(map_orientacao)
    df['IDENT_GEN'] = df['IDENT_GEN'].map(map_identidade)
    df['DEF_FISICA'] = df['DEF_FISICA'].map(map_deficiencias)
    df['DEF_MENTAL'] = df['DEF_MENTAL'].map(map_deficiencias)
    df['DEF_VISUAL'] = df['DEF_VISUAL'].map(map_deficiencias)
    df['DEF_AUDITI'] = df['DEF_AUDITI'].map(map_deficiencias)
    df['TRAN_MENT'] = df['TRAN_MENT'].map(map_deficiencias)
    df['TRAN_COMP'] = df['TRAN_COMP'].map(map_deficiencias)
    df['OUT_VEZES'] = df['OUT_VEZES'].map(map_lesao)
    df['AG_ENVEN'] = df['AG_ENVEN'].map(map_lesao)
    df['AG_FOGO'] = df['AG_FOGO'].map(map_lesao)
    df['AG_CORTE'] = df['AG_CORTE'].map(map_lesao)
    df['AG_OBJETO'] = df['AG_OBJETO'].map(map_lesao)
    df['AG_ENFOR'] = df['AG_ENFOR'].map(map_lesao)
    return df
df = aplicar_mapas(df)

df_sem_nulos = df.copy()

df_sem_nulos.dropna(inplace=True)

colunas_metodo = ['AG_ENVEN', 'AG_FOGO', 'AG_CORTE', 'AG_OBJETO', 'AG_ENFOR']

df_sem_nulos['metodo'] = df_sem_nulos[colunas_metodo].idxmax(axis=1)


df_sem_nulos.loc[df_sem_nulos[colunas_metodo].sum(axis=1) == 0, 'metodo'] = pd.NA
mapa = {
    'AG_ENVEN': 'ENVEN',
    'AG_FOGO': 'FOGO',
    'AG_CORTE': 'CORTE',
    'AG_OBJETO': 'OBJETO',
    'AG_ENFOR': 'ENFOR'
}

df_sem_nulos['metodo'] = df_sem_nulos[colunas_metodo].idxmax(axis=1).map(mapa)

colunas_deficiencia = [
    'DEF_FISICA', 'DEF_MENTAL', 'DEF_VISUAL',
    'DEF_AUDITI', 'TRAN_MENT', 'TRAN_COMP'
]

df_sem_nulos['deficiencia'] = df_sem_nulos[colunas_deficiencia].apply(
    lambda row: ', '.join([col for col in colunas_deficiencia if row[col] == 1]),
    axis=1
)

df_sem_nulos.to_csv("VIOLBR24_processado.csv", index=False)

df_num = df_sem_nulos.copy()
df_num.drop(columns=['AG_ENVEN','AG_FOGO','AG_CORTE','AG_OBJETO','AG_ENFOR','DEF_FISICA', 'DEF_MENTAL', 'DEF_VISUAL','DEF_AUDITI', 'TRAN_MENT', 'TRAN_COMP'], inplace=True)

for col in df_num.columns:
    if pd.api.types.is_numeric_dtype(df_num[col]):
        continue

    df_num[col] = df_num[col].astype('category').cat.codes

df_num = df_num.replace([np.inf, -np.inf], np.nan).dropna()

def definir_background(df):
    var_exogena = [
        'CS_SEXO', 'NU_IDADE_N', 'CS_RACA', 'SG_UF', 'ORIENT_SEX', 'INDENT_GEN', 'SIT_CONJUG', 'CS_ESCOL_N'
    ]
    var_target = 'LES_AUTOP'

    num_vars = df.shape[1]

    nodes = [GraphNode(f"X{i+1}") for i in range(num_vars)]

    # Inicializa o objeto de restrições
    bk = BackgroundKnowledge()

    for i, col in enumerate(df.columns):
        if col in var_exogena:
            for j, col2 in enumerate(df.columns):
                if (col == 'NU_IDADE_N' and col2 == 'CS_ESCOL_N'):
                    continue
                bk.add_forbidden_by_node(nodes[j], nodes[i])
        elif col == var_target:
            for j, col2 in enumerate(df.columns):
                if col2 != var_target:
                    bk.add_forbidden_by_node(nodes[i], nodes[j])
    return bk

bk = definir_background(df_num)
cg = pc(df_num.to_numpy(dtype=float), bk=bk, alpha=0.001, stable=True)
import pickle

with open("dag_pc.pkl", "wb") as f:
    pickle.dump(cg, f)
    