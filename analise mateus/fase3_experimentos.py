import pandas as pd
from dowhy import CausalModel
import statsmodels.api as sm
import numpy as np

def pre_process_data(data_path: str) -> pd.DataFrame:
    print("Carregando e limpando os dados (Limpeza Nuclear)...")
    df = pd.read_csv(data_path, low_memory=False)
    
    # 1. Definir estritamente as colunas que importam para esse estimador
    vars_modelo = ['TRAN_MENT', 'LES_AUTOP', 'SIT_CONJUG', 'OUT_VEZES', 'NU_IDADE_N']
    
    # 2. Cortar fora todo o resto do dataframe (Isso mata 99% dos problemas do statsmodels)
    colunas_presentes = [c for c in vars_modelo if c in df.columns]
    df = df[colunas_presentes].copy()
    
    # 3. Filtro e mapeamento do desfecho e tratamento
    df = df[df['LES_AUTOP'].isin([1, 2, 1.0, 2.0])]
    df['LES_AUTOP'] = df['LES_AUTOP'].map({1: 1, 2: 0, 1.0: 1, 2.0: 0})
    df['TRAN_MENT'] = df['TRAN_MENT'].map({1: 1, 2: 0, 1.0: 1, 2.0: 0})
    
    # 4. Forçar conversão para numérico
    for col in colunas_presentes:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # 5. A Bala de Prata: Trocar qualquer 'Infinito' por NaN e dropar TODOS os NaNs
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    # 6. Reset do índice e forçar float em tudo para a matriz exog
    df = df.reset_index(drop=True)
    df = df.astype(float)
            
    print(f"Linhas mantidas (Matriz 100% limpa): {df.shape[0]}")
    return df

def main():
    csv_path = "dados_sinan_2024.csv"
    gml_path = "grafo_causal_sinan.gml"
    
    df = pre_process_data(csv_path)
    
    # 1. Recriando o modelo e identificação
    model = CausalModel(
        data=df,
        treatment="TRAN_MENT",
        outcome="LES_AUTOP",
        graph=gml_path
    )
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    
    # ==========================================
    # TAREFA 3.1: ESTIMAÇÃO DO EFEITO CAUSAL
    # ==========================================
    print("\n[Tarefa 3.1] Estimando o efeito causal via Regressão Logística...")
    
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.generalized_linear_model",
        confidence_intervals=True,
        method_params={
            "glm_family": sm.families.Binomial()  # <--- CORRIGIDO AQUI (Instanciando o objeto correto)
        }
    )

    print("\n" + "="*40)
    print("RESULTADO DA ESTIMAÇÃO (EFEITO CAUSAL)")
    print("="*40)
    print(estimate)
    print(f"Valor do Efeito Causal Estimado: {estimate.value:.6f}")
    
    # ==========================================
    # TAREFA 3.2: TESTES DE REFUTAÇÃO (ROBUSTEZ)
    # ==========================================
    print("\n[Tarefa 3.2] Iniciando os testes de refutação...")
    
    # Teste 1: Adicionar um Confundidor Aleatório (Não deve mudar muito o efeito)
    print("\nRoteando Teste 1: Random Common Cause...")
    refute_random = model.refute_estimate(
        identified_estimand, 
        estimate,
        method_name="random_common_cause"
    )
    print(refute_random)
    
    # Teste 2: Tratamento Placebo
    print("\nRoteando Teste 2: Placebo Treatment...")
    refute_placebo = model.refute_estimate(
        identified_estimand, 
        estimate,
        method_name="placebo_treatment_refuter" # <-- MUDANÇA AQUI
    )
    print(refute_placebo)
    
    print("\nExperimentos concluídos com sucesso!")

if __name__ == "__main__":
    main()
