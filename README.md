# Ambiente de Aprendizado por Reforço

## Descrição
Este projeto consiste no desenvolvimento de um ambiente para treinamento de agentes inteligentes utilizando aprendizado por reforço. Inspirado na biblioteca OpenAI Gym, o ambiente permite que programadores testem e validem diferentes algoritmos de aprendizado por reforço de maneira simples e intuitiva. O projeto é especialmente útil para estudantes e pesquisadores que desejam explorar esse campo de estudo.

O ambiente foi construído utilizando a linguagem Python e a biblioteca Ursina para renderização gráfica.

## Funcionalidades Principais
- **Ambiente Personalizável:** O tamanho do grid pode ser ajustado entre 4x4 e 8x8.
- **Interação do Agente:** O agente pode se mover em quatro direções (cima, baixo, esquerda, direita) e interagir com objetos.
- **Objetos no Ambiente:** Folhas e gravetos devem ser coletados e levados ao formigueiro, enquanto pedras e lixos devem ser evitados.
- **Sistema de Recompensas:** O agente recebe recompensas positivas e negativas de acordo com suas ações, incentivando comportamentos ótimos.
- **Interface Gráfica:** Utiliza a biblioteca Ursina para visualização em tempo real das ações do agente.
- **Modelo de Aprendizado por Reforço:** O ambiente pode ser modelado como um Processo de Decisão de Markov (MDP), permitindo a implementação de diversos algoritmos de aprendizado por reforço.

## Requisitos
- Python 3.8+
- Ursina Engine
- NumPy

## Como Usar
```python
from ant_env import AntEnvironment

# Criando um ambiente 4x4
env = AntEnvironment(render=True ,grid_size=4)

# Resetando o ambiente
env.reset()

# Tomando uma ação (exemplo: mover para a direita)
estado, recompensa, done = env.step(2)

print(f"Estado: {estado}, Recompensa: {recompensa}, Episódio terminado: {done}")
```

## Estrutura do Projeto
```
/AnthilAdventureEnv
├── ant_env.py  # Implementação do ambiente
├── /Models    # Modelos treinados para cada grid_size
├── /TestAndTrainExample     # Exemplos de algoritmos para treino e teste do ambiente
└── README.md    # Documentação do projeto
```

## Modelo de Recompensas
- **-0.4**: Quando o agente não está carregando nada.
- **-0.2**: Quando o agente está carregando um objeto útil.
- **-1**: Quando o agente carrega um objeto indesejado, tenta colidir com limites da malha, deposita um objeto útil fora do formigueiro ou deposita um objeto inútil no formigueiro.
- **-0.6**: Ao depositar um objeto inútil fora do formigueiro.
- **+0.3**: Ao coletar um objeto útil.
- **+1**: Ao depositar um objeto útil no formigueiro.
- **+10**: Quando todos os objetos úteis são levados ao formigueiro.

## Contribuição
Sinta-se à vontade para contribuir com melhorias, relatando problemas ou sugerindo novas funcionalidades. Basta abrir um pull request ou uma issue no repositório.
