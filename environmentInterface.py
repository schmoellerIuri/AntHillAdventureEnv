import numpy as np
from ursina import *

class Ambiente:
  """
  Interface para ambientes de aprendizado por reforço em Python.

  Atributos:
    transicoes: Array das transições do ambiente.
    estados: Array contendo o espaço de estados do ambiente.
    acoes: Array das ações que um agente pode tomar.
    recompensas: Array contendo as recompensas para cada estado.
    estados_finais: Array contendo os estados finais.
    estado_atual: Estado atual do ambiente.
    descricao: Descrição textual do ambiente.
  """

  def __init__(self, transicoes, estados, acoes, recompensas, estados_finais, 
               estado_inicial=None, descricao=""):
    self.transicoes = transicoes
    self.estados = estados
    self.acoes = acoes
    self.recompensas = recompensas
    self.estados_finais = estados_finais
    self.estado_atual = estado_inicial or self.estados[0]
    self.descricao = descricao

  def step(self, acao):
    """
    Realiza uma ação no ambiente e retorna a nova observação, 
    recompensa e se o episódio terminou.

    Argumentos:
      acao: Ação a ser tomada no ambiente (índice do array de ações).

    Retorna:
      nova_observacao: Novo estado do ambiente.
      recompensa: Recompensa recebida por realizar a ação.
      feito: Indica se o episódio terminou.
    """

    #TODO: Atualizar descrição do ambiente a cada passo
    probabilidades = self.transicoes[self.estado_atual, acao, :, :]
    novo_estado, recompensa, feito = np.random.choice(
        self.estados, self.recompensas, self.estados_finais, 
        p=probabilidades.flatten()
    )
    self.estado_atual = novo_estado
    return novo_estado, recompensa, feito

  def reset(self):
    """
    Reseta o ambiente para o estado inicial.

    Retorna:
      estado_inicial: Estado inicial do ambiente.
    """

    self.estado_atual = self.estados[0]
    return self.estado_atual

  def render(self):
    """
    Renderiza o ambiente graficamente com Ursina.

    Este método deve ser implementado de acordo
    com a representação visual desejada para o ambiente.
    """
    pass