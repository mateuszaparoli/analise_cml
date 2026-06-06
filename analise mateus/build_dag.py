import networkx as nx
import matplotlib.pyplot as plt

def main():
    print("Iniciando a construção do DAG Causal...")
    
    # Inicializa um Grafo Direcionado (DAG)
    G = nx.DiGraph()

    # 1. Definição dos Nós (Variáveis do dataset)
    nodes = [
        "NU_IDADE_N", "CS_SEXO", "CS_RACA", "DEF_MENTAL", "DEF_FISICA", "DEF_VISUAL", "DEF_AUDITI", # Camada 1: Inerentes
        "IDENT_GEN", "ORIENT_SEX", "CS_ESCOL_N", "OUT_VEZES",                                       # Camada 2: Construtos/Contexto
        "SIT_CONJUG",                                                                               # Camada 3: Suporte
        "TRAN_MENT", "TRAN_COMP",                                                                   # Camada 4: Mediadores
        "AG_ENVEN", "AG_FOGO", "AG_CORTE", "AG_OBJETO", "AG_ENFOR",                                 # Camada 5: Meio
        "LES_AUTOP"                                                                                 # Camada 6: Desfecho (Sink Node)
    ]
    G.add_nodes_from(nodes)

    # 2. Definição das Arestas (Direção Causal: Causa -> Efeito)
    # Baseado nas hipóteses do relatório parcial
    edges = [
        # Hipótese 5.6: Sexo e Idade -> Lesão Autoprovocada
        ("CS_SEXO", "LES_AUTOP"),
        ("NU_IDADE_N", "LES_AUTOP"),
        ("NU_IDADE_N", "CS_ESCOL_N"), # Idade geralmente afeta nível de escolaridade
        
        # Hipótese 5.2: Orientação/Identidade -> Saúde Mental
        ("ORIENT_SEX", "TRAN_MENT"),
        ("IDENT_GEN", "TRAN_MENT"),
        
        # Hipótese 5.4: Escolaridade -> Saúde Mental
        ("CS_ESCOL_N", "TRAN_MENT"),
        
        # Hipótese 5.3: Violência Recorrente -> Desfecho
        ("OUT_VEZES", "LES_AUTOP"),
        ("OUT_VEZES", "TRAN_MENT"), # Trauma repetido também afeta saúde mental
        
        # Hipótese 5.5: Situação Conjugal -> Desfecho
        ("SIT_CONJUG", "LES_AUTOP"),
        ("SIT_CONJUG", "TRAN_MENT"),
        
        # Hipótese 5.1: Transtornos -> Lesão Autoprovocada
        ("TRAN_MENT", "LES_AUTOP"),
        ("TRAN_COMP", "LES_AUTOP"),
        ("DEF_MENTAL", "LES_AUTOP"),
        
        # Hipótese 5.7: Meios de Agressão -> Desfecho
        ("AG_ENVEN", "LES_AUTOP"),
        ("AG_CORTE", "LES_AUTOP"),
        ("AG_ENFOR", "LES_AUTOP"),
        ("AG_FOGO", "LES_AUTOP"),
        ("AG_OBJETO", "LES_AUTOP")
    ]
    G.add_edges_from(edges)

    # 3. Verificação de Ciclos (Um DAG não pode ter ciclos)
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Erro: O grafo possui ciclos e não é um DAG válido!")
    else:
        print("Sucesso: O grafo é um DAG válido (sem ciclos).")

    # 4. Exportação do Grafo
    # O formato GML é excelente para carregar na biblioteca DoWhy na Fase 2
    nx.write_gml(G, "grafo_causal_sinan.gml")
    print("-> Grafo exportado para: 'grafo_causal_sinan.gml'")

    # 5. Gerar uma visualização rápida para o relatório
    plt.figure(figsize=(14, 10))
    # Layout hierárquico simplificado
    pos = nx.spring_layout(G, k=0.9, seed=42) 
    nx.draw(G, pos, with_labels=True, node_color='#A0CBE2', 
            node_size=2500, font_size=8, font_weight='bold', 
            edge_color='gray', arrows=True, arrowsize=15)
    
    plt.title("Modelo Estrutural Causal - Violência Autoprovocada", fontsize=16)
    plt.savefig("dag_visualizacao.png", format="PNG", dpi=300)
    print("-> Visualização salva como: 'dag_visualizacao.png'")

if __name__ == "__main__":
    main()
