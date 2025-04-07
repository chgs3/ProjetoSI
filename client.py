import numpy as np
import random
import connection
import time
import os

# Definições do Q-Learning
ALPHA = 0.3  # Vai ser a taxa de aprendizado dele
GAMMA = 0.95  # Vai ser o fator de desconto de aprendizado
EPSILON = 1.0  # Vai ser a taxa de exploração dele
EPSILON_DECAY = 0.995  # Redução gradual de EPSILON
EPSILON_MIN = 0.1  # Limite mínimo para exploração
ACTIONS = ["left", "right", "jump"]
TOTAL_EPISODES = 10000

# Inicializa a Q-Table com zeros (96 estados x 3 ações)
q_table = np.zeros((96, 3))

# Atualização para carregar a QTable
if os.path.exists("q_table.txt"):
    q_table = np.loadtxt("q_table.txt")
    print("Q-Table existente carregada. Continuando o treinamento...")

def escolher_acao(estado):
    if random.uniform(0, 1) < EPSILON:
        return random.choice(ACTIONS)  # Exploração
    else:
        return ACTIONS[np.argmax(q_table[estado])]  # Exploração

def salvar_q_table():
    np.savetxt("q_table.txt", q_table, fmt="%.6f")

def obter_indice_estado(estado_bin):
    return int(estado_bin, 2)  # Converte binário para índice inteiro

def main():
    global EPSILON
    porta = 2037
    socket_jogo = connection.connect(porta)
    
    if socket_jogo == 0:
        print("Falha na conexão com o jogo.")
        return
    
    for episodio in range(TOTAL_EPISODES):
        estado, _ = connection.get_state_reward(socket_jogo, "jump")
        estado = obter_indice_estado(estado)
        done = False
        
        while not done:
            acao = escolher_acao(estado)
            novo_estado, recompensa = connection.get_state_reward(socket_jogo, acao)
            novo_estado = obter_indice_estado(novo_estado)
            
            # atualiza a Q-Table
            a_index = ACTIONS.index(acao)
            q_table[estado, a_index] = q_table[estado, a_index] + ALPHA * (recompensa + GAMMA * np.max(q_table[novo_estado]) - q_table[estado, a_index])
            
            estado = novo_estado
            
            if recompensa == -1:  # condição de parada
                done = True
        
        # Decaimento do EPSILON
        EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)
        
        # Atualiza a Q-Table a cada episódio
        salvar_q_table()
        print(f"Episódio {episodio} concluído. EPSILON: {EPSILON:.4f}")
    
    print("Treinamento concluído! Q-Table salva.")
    
if __name__ == "__main__":
    main()
