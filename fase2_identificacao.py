import pandas as pd
from dowhy import CausalModel

def pre_process_data(data_path: str) -> pd.DataFrame:
    """Carrega e prepara os dados do SINAN para a análise causal."""
    print("Carregando os dados...")
    df = pd.read_csv(data_path)
    
    # 1. Filtrar respostas inválidas/ignoradas no desfecho (8 e 9)
    df = df[df['LES_AUTOP'].isin([1, 2])]
    
    # 2. Mapear para binário (1 = Sim -> 1, 2 = Não -> 0)
    df['LES_AUTOP'] = df['LES_AUTOP'].map({1: 1, 2: 0})
    
    # 3. Garantir que a variável de tratamento (TRAN_MENT) também seja binária (0 ou 1)
    # Se no seu dataset original for 1 (Sim) e 2 (Não), ajuste como abaixo:
    if df['TRAN_MENT'].isin([1, 2]).all():
        df['TRAN_MENT'] = df['TRAN_MENT'].map({1: 1, 2: 0})
        
    print(f"Dados carregados e filtrados. Total de registros: {df.shape[0]}")
    return df

def main():
    # Substitua pelo nome real do arquivo CSV convertido do Datasus
    csv_path = "dados_sinan_2024.csv" 
    gml_path = "grafo_causal_sinan.gml"
    
    # Passo 1: Pré-processamento
    try:
        df = pre_process_data(csv_path)
    except FileNotFoundError:
        print(f"Erro: Certifique-se de que o arquivo '{csv_path}' está na mesma pasta.")
        return

    # Passo 2: Definição do Objeto Causal no DoWhy
    print("\n[Tarefa 2.1] Inicializando o Modelo Causal no DoWhy...")
    model = CausalModel(
        data=df,
        treatment="TRAN_MENT",
        outcome="LES_AUTOP",
        graph=gml_path
    )
    
    # Passo 3: Identificação do Efeito Causal (Critério Backdoor/Frontdoor)
    print("\n[Tarefa 2.2] Executando a Identificação do Efeito...")
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    
    print("\n" + "="*40)
    print("RESULTADO DA IDENTIFICAÇÃO CAUSAL")
    print("="*40)
    print(identified_estimand)
    
    # Salvar a expressão matemática identificada para usar na Fase 3
    return model, identified_estimand

if __name__ == "__main__":
    main()
