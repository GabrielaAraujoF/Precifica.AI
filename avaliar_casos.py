# Avalia o engine contra o gabarito "Preço Justo - Casos de teste (5 cenários)".
#
# Uso:
#   python avaliar_casos.py                  modo offline: extração ideal simulada, sem gastar cota
#   python avaliar_casos.py --ia             modo IA: conversa real com o Gemini (~35 chamadas)
#   python avaliar_casos.py --ia --cenarios 1,3   roda só os cenários escolhidos
#
# Saída: resumo no terminal + relatorio_casos.md (só o que não bateu).
#
# Modo offline: a "IA" devolve o JSON que um extrator obediente ao schema atual devolveria
# (sem fazer contas). Testa a matemática e as regras do motor.
# Modo IA: testa o produto de verdade, incluindo como o Gemini interpreta cada frase.
# Verificações marcadas como "só IA" ficam como pendentes no modo offline.

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import time

import engine
from engine import formatar_moeda, processar_mensagem_usuario

TOLERANCIA = 0.015  # R$ (gabarito arredonda em centavos; "cerca de" tolera 1 centavo)
PAUSA_IA_SEGUNDOS = 4.5  # respeita 15 requisições/min do plano gratuito
NOVAS_TENTATIVAS_IA = 3  # se a API estiver fora, tenta o mesmo turno de novo
ESPERA_API_FORA_SEGUNDOS = 30
ERROS_DE_API = ("Nenhum modelo respondeu", "Falha na chamada ao Gemini", "GEMINI_API_KEY", "não instalado")

# Silencia os tracebacks do engine; o avaliador mostra mensagens curtas
logging.getLogger("preco_justo.engine").setLevel(logging.CRITICAL)


class IAIndisponivel(Exception):
    """A API do Gemini não respondeu mesmo após novas tentativas."""


class CotaEsgotada(IAIndisponivel):
    """Todos os modelos responderam 429: insistir só gasta mais cota."""


# Infraestrutura

class Sessao:
    """Conversa com o engine, guardando o estado entre mensagens."""

    def __init__(self, modo_ia: bool):
        self.modo_ia = modo_ia
        self.estado = None
        self.respostas: list[str] = []
        self._primeira = True

    def enviar(self, mensagem: str, extracao_ideal: dict) -> dict:
        if self.modo_ia:
            if not self._primeira:
                time.sleep(PAUSA_IA_SEGUNDOS)
            for tentativa in range(1, NOVAS_TENTATIVAS_IA + 1):
                r = processar_mensagem_usuario(mensagem, self.estado)
                erro = r.get("erro") or ""
                if not any(e in erro for e in ERROS_DE_API):
                    break
                if "GEMINI_API_KEY" in erro or "não instalado" in erro:
                    raise IAIndisponivel(erro)
                codigos = [t.split(":")[-1].strip() for t in erro.split("Tentativas:")[-1].split("|")]
                if codigos and all(c == "429" for c in codigos):
                    raise CotaEsgotada(erro)
                detalhe = erro.split("Tentativas:")[-1].strip()
                print(f"   API indisponível ({detalhe}) - tentativa {tentativa}/{NOVAS_TENTATIVAS_IA}; "
                      f"aguardando {ESPERA_API_FORA_SEGUNDOS}s...")
                if tentativa == NOVAS_TENTATIVAS_IA:
                    raise IAIndisponivel(erro)
                time.sleep(ESPERA_API_FORA_SEGUNDOS)
        else:
            base = {"intencao": "informar_dados", "materiais": [], "custos_fixos": [], "tempo_partes_horas": []}
            r = processar_mensagem_usuario(mensagem, self.estado, extrator=lambda m, e: {**base, **extracao_ideal})
        self._primeira = False
        self.estado = r["estado"]
        self.respostas.append(r["resposta"])
        return r


class Cenario:
    def __init__(self, nome: str, modo_ia: bool):
        self.nome = nome
        self.modo_ia = modo_ia
        self.itens: list[dict] = []
        self.indisponivel: str | None = None

    def checar(self, verificacao, esperado, obtido, ok, causa="", so_ia=False):
        if so_ia and not self.modo_ia:
            ok, obtido = None, "verificar com --ia"
        self.itens.append({
            "verificacao": verificacao, "esperado": esperado,
            "obtido": obtido, "ok": ok, "causa": causa,
        })

    def numero(self, verificacao, esperado, obtido, causa="", so_ia=False):
        ok = obtido is not None and abs(obtido - esperado) <= TOLERANCIA
        txt = "não calculou" if obtido is None else formatar_moeda(obtido)
        self.checar(verificacao, formatar_moeda(esperado), txt, ok, causa, so_ia)

    def pergunta(self, verificacao, resposta, termos, causa="", so_ia=False):
        ok = contem(resposta, *termos)
        self.checar(verificacao, "pergunta/menciona", "sim" if ok else "não", ok, causa, so_ia)


def contem(texto: str, *termos: str) -> bool:
    t = engine._chave(texto or "")
    return any(engine._chave(x) in t for x in termos)


def valor_no_texto(resposta: str, valor: float) -> bool:
    return formatar_moeda(valor) in (resposta or "")


# Conceitos do gabarito a partir da saída do engine
#   Custos = materiais + deslocamento + outros + taxa + custos mensais rateados
#   Sobra  = preço - custos ;  Valor por hora = sobra / horas totais

def gabarito_servico(m: dict | None, horas: float | None) -> dict:
    if not m:
        return {"custos": None, "sobra": None, "valor_hora": None}
    custos = (m["total_materiais"] + m["deslocamento"] + m["outros_gastos"]
              + m["valor_taxa"] + m.get("custos_fixos_rateados", 0))
    sobra = m["preco_cobrado"] - custos
    vh = sobra / horas if horas else None
    return {"custos": round(custos, 2), "sobra": round(sobra, 2), "valor_hora": round(vh, 2) if vh is not None else None}


def materiais_do_estado(estado: dict) -> float:
    """Custo de materiais por atendimento (considera rendimento das embalagens)."""
    itens = [m for m in (estado or {}).get("materiais") or [] if m.get("valor") is not None]
    return round(sum(float(engine.custo_material(m)) for m in itens), 2)


def pediu_confirmacao(resposta: str) -> bool:
    return contem(resposta, "confira", "confirm", "esta tudo certo")


CAUSA_META_HORA = ("O motor exige 'quanto quer ganhar por hora' para calcular; o gabarito calcula "
                   "sem meta e mostra o valor por hora REAL (sobra ÷ horas).")
CAUSA_SEM_VALOR_HORA = "O motor não calcula nem mostra o valor por hora real (sobra ÷ horas)."
CAUSA_SEM_CONFIRMACAO = "O motor calcula direto, sem mostrar os dados e pedir confirmação antes."
CAUSA_SEM_ESTIMATIVA = "A marcação 'estimativa sua' não aparece na resposta."
CAUSA_PERGUNTA = "A pergunta esperada não aparece na resposta do assistente."
CAUSA_EXTRACAO = "A IA não preencheu o campo como o schema pede (conferir o prompt)."


# Cenário 1: caminho feliz

def cenario_1(modo_ia):
    c = Cenario("1. Caminho feliz (domicílio)", modo_ia)
    s = Sessao(modo_ia)
    r = s.enviar("Faço maquiagem social e cobro R$ 150.",
                 {"nome_servico": "Maquiagem social", "preco_cobrado": 150, "modelo_cobranca": "por_servico"})
    c.checar("Não calcula com dados faltando", "sem resultado", "calculou" if r["metricas"] else "sem resultado",
             r["metricas"] is None)
    c.pergunta("Pergunta materiais", r["resposta"], ["materia"], CAUSA_PERGUNTA)
    c.pergunta("Pergunta deslocamento", r["resposta"], ["desloc", "transporte"], CAUSA_PERGUNTA)
    c.pergunta("Pergunta forma de pagamento", r["resposta"], ["pag", "cartao", "taxa"], CAUSA_PERGUNTA)
    c.pergunta("Pergunta tempo contando preparo e limpeza", r["resposta"], ["preparo"], CAUSA_PERGUNTA)

    s.enviar("Descartáveis R$ 3,50 + produtos R$ 10,00 + cílios R$ 8,00",
             {"materiais": [{"item": "Descartáveis", "valor": 3.5}, {"item": "Produtos", "valor": 10},
                            {"item": "Cílios", "valor": 8}]})
    s.enviar("1h30 de maquiagem + 30 min de preparo + 30 min de limpeza", {"tempo_partes_horas": [1.5, 0.5, 0.5]})
    s.enviar("App ida e volta: R$ 24,00", {"deslocamento": 24})
    r = s.enviar("Cartão, taxa de 3,5%", {"taxa_maquininha_pct": 0.035})

    c.numero("Materiais somados", 21.50, materiais_do_estado(s.estado))
    c.checar("Tempo total = 2h30 (soma de maquiagem + preparo + limpeza)", "2,5h",
             f"{s.estado.get('tempo_horas')}h", s.estado.get("tempo_horas") == 2.5, CAUSA_EXTRACAO, so_ia=True)
    c.checar("Mostra os valores e pede confirmação antes de calcular", "pede confirmação",
             "sim" if pediu_confirmacao(r["resposta"]) else "não", pediu_confirmacao(r["resposta"]),
             CAUSA_SEM_CONFIRMACAO)
    c.checar("Não calcula antes da confirmação", "sem resultado", "calculou" if r["metricas"] else "sem resultado",
             r["metricas"] is None)

    r = s.enviar("Sim", {"intencao": "confirmar"})
    m = r["metricas"] or {}
    g = gabarito_servico(r["metricas"], s.estado.get("tempo_horas"))
    c.checar("Calcula após a confirmação", "resultado", "calculou" if r["metricas"] else "não calculou",
             r["metricas"] is not None, CAUSA_META_HORA)
    c.numero("Taxa do cartão (3,5% de 150)", 5.25, m.get("valor_taxa"))
    c.numero("Total de custos", 50.75, m.get("custos_totais"))
    c.numero("Sobra", 99.25, m.get("sobra_real"))
    c.numero("Valor por hora (2h30)", 39.70, m.get("valor_hora_real"))
    c.numero("Conferência independente da sobra (preço - custos)", 99.25, g["sobra"])
    for valor, nome in ((50.75, "total de custos"), (99.25, "sobra"), (39.70, "valor por hora")):
        c.checar(f"Texto mostra {nome} ({formatar_moeda(valor)})", formatar_moeda(valor),
                 "sim" if valor_no_texto(r["resposta"], valor) else "não", valor_no_texto(r["resposta"], valor),
                 "Valor calculado, mas não exibido na resposta.")

    r = s.enviar("Quero ganhar R$ 50 por hora", {"valor_hora_desejado": 50})
    m = r["metricas"] or {}
    c.numero("[meta] Preço mínimo para R$ 50/h (gabarito: cerca de R$ 176,69)", 176.69, m.get("preco_minimo"))
    c.checar("[meta] Texto mostra o preço mínimo para a meta", "R$ 176,68 ou R$ 176,69",
             "sim" if valor_no_texto(r["resposta"], 176.68) or valor_no_texto(r["resposta"], 176.69) else "não",
             valor_no_texto(r["resposta"], 176.68) or valor_no_texto(r["resposta"], 176.69))
    c.numero("[meta] Sobra continua sendo preço - custos", 99.25, m.get("sobra_real"))
    return c


# Cenário 2: "E se eu cobrar R$ 60?"

def cenario_2(modo_ia):
    c = Cenario("2. Comparação de preços (R$ 80 x R$ 60)", modo_ia)
    s = Sessao(modo_ia)
    s.enviar("Faço maquiagem básica e cobro R$ 80. Gasto R$ 15 de materiais, R$ 10 de deslocamento, "
             "levo 2 horas no total e recebo no cartão com taxa de 3%.",
             {"nome_servico": "Maquiagem básica", "preco_cobrado": 80, "tempo_horas": 2, "deslocamento": 10,
              "materiais": [{"item": "Materiais", "valor": 15}], "taxa_maquininha_pct": 0.03,
              "modelo_cobranca": "por_servico"})
    r = s.enviar("E se eu cobrar R$ 60?", {"intencao": "simular_preco", "novo_preco_simulado": 60})
    sim = r["simulacao"]
    c.checar("Faz a simulação", "simulação feita",
             "feita" if sim else "pediu: " + ", ".join(r["informacoes_faltantes"]), sim is not None, CAUSA_META_HORA)
    o, n = (sim or {}).get("metricas_original") or {}, (sim or {}).get("metricas_simuladas") or {}
    c.numero("Taxa a R$ 80", 2.40, o.get("valor_taxa"))
    c.numero("Taxa a R$ 60 (recalculada)", 1.80, n.get("valor_taxa"))
    c.checar("Custos fixos não mudam", "R$ 25,00 nos dois",
             f"{formatar_moeda(o.get('custo_direto', 0))} / {formatar_moeda(n.get('custo_direto', 0))}",
             bool(sim) and o["custo_direto"] == n["custo_direto"] == 25.0)
    c.numero("Total de custos a R$ 80", 27.40, o.get("custos_totais"))
    c.numero("Total de custos a R$ 60", 26.80, n.get("custos_totais"))
    c.numero("Sobra a R$ 80", 52.60, o.get("sobra_real"))
    c.numero("Sobra a R$ 60", 33.20, n.get("sobra_real"))
    c.numero("Diferença de sobra", -19.40, (sim or {}).get("diferenca_sobra_real"))
    c.numero("Valor por hora a R$ 80", 26.30, o.get("valor_hora_real"))
    c.numero("Valor por hora a R$ 60", 16.60, n.get("valor_hora_real"))

    texto = r["resposta"]
    c.checar("Mostra os dois cenários lado a lado", "R$ 80 e R$ 60 no texto",
             "sim" if valor_no_texto(texto, 80) and valor_no_texto(texto, 60) else "não",
             valor_no_texto(texto, 80) and valor_no_texto(texto, 60))
    c.checar("Mostra taxa R$ 2,40 → R$ 1,80 no texto", "R$ 2,40 e R$ 1,80",
             "sim" if valor_no_texto(texto, 2.40) and valor_no_texto(texto, 1.80) else "não",
             valor_no_texto(texto, 2.40) and valor_no_texto(texto, 1.80))
    c.checar("Explica por que a sobra cai mais que o preço (-25% x -36,9%)", "explicação",
             "sim" if contem(texto, "36,9") and contem(texto, "25,0") else "não",
             contem(texto, "36,9") and contem(texto, "25,0"))
    return c


# Cenário 3: pacote x unidade

def cenario_3(modo_ia):
    c = Cenario("3. Pacote x unidade", modo_ia)
    s = Sessao(modo_ia)
    s.enviar("Cobro R$ 120 pela maquiagem.", {"preco_cobrado": 120, "modelo_cobranca": "por_servico"})
    turnos = [
        ("Uma base de 30 ml me custou R$ 120, rende uns 40 atendimentos",
         {"item": "Base 30 ml", "valor": 120, "rendimento": 40, "embalagem": True, "estimativa": True}),
        ("Caixa de cílios com 10 pares por R$ 35, uso 1 par por cliente",
         {"item": "Cílios", "valor": 35, "rendimento": 10, "quantidade_usada": 1, "embalagem": True}),
        ("Pacote de 100 esponjinhas descartáveis por R$ 25, uso 2",
         {"item": "Esponjinhas", "valor": 25, "rendimento": 100, "quantidade_usada": 2, "embalagem": True}),
        ("Caixa de cotonetes por R$ 12, rende uns 60 atendimentos",
         {"item": "Cotonetes", "valor": 12, "rendimento": 60, "embalagem": True, "estimativa": True}),
    ]
    for msg, material in turnos:
        s.enviar(msg, {"materiais": [material]})
    r = s.enviar("Levo 1h30. Sem deslocamento, pagamento por Pix.",
                 {"tempo_horas": 1.5, "deslocamento": 0, "taxa_maquininha_pct": 0})

    c.checar("Schema tem campo para rendimento da embalagem", "sim",
             "sim" if "rendimento" in engine.SYSTEM_PROMPT else "não", "rendimento" in engine.SYSTEM_PROMPT)
    mat = materiais_do_estado(s.estado)
    c.checar("Nunca soma as embalagens (R$ 192) como custo de um atendimento", "≠ R$ 192,00",
             formatar_moeda(mat), abs(mat - 192) > TOLERANCIA, CAUSA_EXTRACAO, so_ia=True)
    c.numero("Materiais por atendimento", 7.20, mat, CAUSA_EXTRACAO, so_ia=True)
    c.checar("Marca rendimentos chutados como 'estimativa sua' (confirmação)", "estimativa sua",
             "sim" if contem(r["resposta"], "estimativa sua") else "não", contem(r["resposta"], "estimativa sua"),
             CAUSA_SEM_ESTIMATIVA)
    r = s.enviar("Sim", {"intencao": "confirmar"})
    m = r["metricas"] or {}
    c.checar("Calcula após confirmação", "resultado", "calculou" if r["metricas"] else "não calculou",
             r["metricas"] is not None)
    c.numero("Sobra", 112.80, m.get("sobra_real"), so_ia=True)
    c.numero("Valor por hora (1h30)", 75.20, m.get("valor_hora_real"), so_ia=True)
    c.checar("Resultado marca 'estimativa sua'", "estimativa sua",
             "sim" if contem(r["resposta"], "estimativa sua") else "não", contem(r["resposta"], "estimativa sua"),
             CAUSA_SEM_ESTIMATIVA)

    r = s.enviar("Na verdade a base rende só 30 atendimentos.",
                 {"intencao": "corrigir_dado", "materiais": [{"item": "Base 30 ml", "rendimento": 30}]})
    m = r["metricas"] or {}
    c.numero("Materiais após correção (base rende 30)", 8.20, materiais_do_estado(s.estado), CAUSA_EXTRACAO)
    c.checar("Recalcula direto após a correção", "resultado", "calculou" if r["metricas"] else "não calculou",
             r["metricas"] is not None)
    c.numero("Sobra após correção", 111.80, m.get("sobra_real"))
    c.numero("Valor por hora após correção", 74.53, m.get("valor_hora_real"))

    s2 = Sessao(modo_ia)
    r = s2.enviar("Cobro R$ 120. Comprei um pó compacto de R$ 80.",
                  {"preco_cobrado": 120, "materiais": [{"item": "Pó compacto", "valor": 80, "embalagem": True}]})
    c.pergunta("Pergunta quanto rende uma embalagem sem rendimento informado", r["resposta"], ["rende"],
               CAUSA_PERGUNTA)
    c.checar("Não usa o preço da embalagem como custo do atendimento", "sem resultado",
             "calculou" if r["metricas"] else "sem resultado", r["metricas"] is None)
    return c


# Cenário 4: incompleto e inválido

def cenario_4(modo_ia):
    c = Cenario("4. Dados incompletos e inválidos", modo_ia)
    s = Sessao(modo_ia)
    r = s.enviar("Faço maquiagem de noiva e cobro R$ 400.",
                 {"nome_servico": "Maquiagem de noiva", "preco_cobrado": 400, "modelo_cobranca": "por_servico"})
    c.checar("Não calcula com dados faltando", "sem resultado", "calculou" if r["metricas"] else "sem resultado",
             r["metricas"] is None)
    c.pergunta("Pergunta materiais", r["resposta"], ["materia"], CAUSA_PERGUNTA)
    c.pergunta("Pergunta tempo", r["resposta"], ["tempo"], CAUSA_PERGUNTA)
    c.pergunta("Pergunta deslocamento", r["resposta"], ["desloc", "transporte"], CAUSA_PERGUNTA)

    r = s.enviar("Não sei quanto gasto de material", {"nao_sabe": ["materiais"]})
    c.pergunta("Oferece ajuda para estimar material", r["resposta"], ["estim"], CAUSA_PERGUNTA)
    c.checar("Não inventa valor de material", "nenhum material", str(s.estado.get("materiais")),
             not s.estado.get("materiais"))

    r = s.enviar("Deslocamento: -30", {"deslocamento": -30})
    c.checar("Não aceita deslocamento negativo", "não grava -30", str(s.estado.get("deslocamento")),
             (s.estado.get("deslocamento") or 0) >= 0)
    c.pergunta("Pergunta se foi erro de digitação", r["resposta"], ["digitacao"], CAUSA_PERGUNTA)

    r = s.enviar("Levo 0 horas", {"tempo_horas": 0})
    c.checar("Não aceita tempo 0", "não grava 0", f"{s.estado.get('tempo_horas')}",
             s.estado.get("tempo_horas") not in (0, 0.0))
    c.pergunta("Pede o tempo real", r["resposta"], ["tempo real"], CAUSA_PERGUNTA)

    r = s.enviar("Uns 40 e pouco", {"materiais": [{"item": "Material", "valor": 40, "estimativa": True}]})
    c.pergunta("Aceita número aproximado e marca como estimativa", r["resposta"], ["estimativa sua"],
               CAUSA_SEM_ESTIMATIVA)

    for msg, esperado in (("Cobro mil e duzentos", 1200.0), ("Cobro 1.200,50", 1200.5), ("Cobro 1200", 1200.0)):
        r = s.enviar(msg, {"intencao": "corrigir_dado", "preco_cobrado": esperado, "campo_corrigido": "preco_cobrado"})
        c.checar(f"Interpreta '{msg[6:]}'", formatar_moeda(esperado), formatar_moeda(s.estado.get("preco_cobrado") or 0),
                 s.estado.get("preco_cobrado") == esperado, CAUSA_EXTRACAO, so_ia=True)
        c.checar(f"Mostra o valor entendido para '{msg[6:]}'", formatar_moeda(esperado),
                 "sim" if valor_no_texto(r["resposta"], esperado) else "não", valor_no_texto(r["resposta"], esperado),
                 "O valor interpretado não é repetido para a pessoa conferir.")

    r = s.enviar("Cobro R$ 0", {"intencao": "corrigir_dado", "preco_cobrado": 0, "campo_corrigido": "preco_cobrado"})
    c.pergunta("Pergunta se preço R$ 0 é erro ou serviço gratuito", r["resposta"], ["gratuito"], CAUSA_PERGUNTA)
    c.checar("Não grava preço R$ 0", "preço anterior mantido", formatar_moeda(s.estado.get("preco_cobrado") or 0),
             (s.estado.get("preco_cobrado") or 0) > 0)

    r = s.enviar("Na verdade cobro R$ 400. Materiais uns R$ 45 (estimativa), deslocamento R$ 40, levo 4 horas "
                 "e recebo por Pix.",
                 {"intencao": "corrigir_dado", "preco_cobrado": 400, "deslocamento": 40, "tempo_horas": 4,
                  "taxa_maquininha_pct": 0, "materiais": [{"item": "Material", "valor": 45, "estimativa": True}]})
    c.checar("Mostra os dados e pede confirmação", "pede confirmação",
             "sim" if pediu_confirmacao(r["resposta"]) else "não", pediu_confirmacao(r["resposta"]),
             CAUSA_SEM_CONFIRMACAO)
    r = s.enviar("Sim", {"intencao": "confirmar"})
    m = r["metricas"] or {}
    c.checar("Calcula com os dados finais", "resultado", "calculou" if r["metricas"] else "não calculou",
             r["metricas"] is not None)
    c.numero("Total de custos", 85.00, m.get("custos_totais"))
    c.numero("Sobra", 315.00, m.get("sobra_real"))
    c.numero("Valor por hora (4h)", 78.75, m.get("valor_hora_real"))
    alvo = "Materiais: R$ 45,00 (estimativa sua)"
    c.checar(f"Exibe '{alvo}'", alvo, "sim" if alvo in r["resposta"] else "não", alvo in r["resposta"],
             CAUSA_SEM_ESTIMATIVA)
    return c


# Cenário 5: custos mensais e plano mensal

def cenario_5(modo_ia):
    c = Cenario("5. Custos mensais e plano mensal", modo_ia)
    s = Sessao(modo_ia)
    r = s.enviar("Faço acompanhamento de automaquiagem e cobro R$ 400 por mês. São 4 encontros de 2 horas.",
                 {"nome_servico": "Acompanhamento de automaquiagem", "modelo_cobranca": "recorrente",
                  "preco_cobrado": 400, "periodicidade_preco": "mensal", "encontros_por_mes": 4,
                  "horas_por_encontro": 2})
    c.checar("Aceita preço por mês (modo recorrente)", "recorrente / mensal",
             f"{s.estado.get('modelo_cobranca')} / {s.estado.get('periodicidade_preco')}",
             s.estado.get("modelo_cobranca") == "recorrente" and s.estado.get("periodicidade_preco") == "mensal",
             CAUSA_EXTRACAO, so_ia=True)
    c.checar("Não confunde R$ 400/mês com preço por encontro", "R$ 400,00",
             formatar_moeda(s.estado.get("preco_cobrado") or 0), s.estado.get("preco_cobrado") == 400,
             CAUSA_EXTRACAO, so_ia=True)
    horas = (s.estado.get("encontros_por_mes") or 0) * (s.estado.get("horas_por_encontro") or 0)
    c.checar("Soma as horas do mês (4 × 2h = 8h)", "8h", f"{_num(horas)}h", horas == 8, CAUSA_EXTRACAO, so_ia=True)
    c.pergunta("Pergunta material por encontro", r["resposta"], ["materia"], CAUSA_PERGUNTA)
    c.pergunta("Pergunta gastos que se repetem todo mês", r["resposta"], ["todo mes"], CAUSA_PERGUNTA)
    c.pergunta("Pergunta quantos clientes dividem os gastos", r["resposta"], ["client"], CAUSA_PERGUNTA)

    s.enviar("R$ 20,00 por encontro",
             {"custos_fixos": [{"item": "Material", "valor": 20, "por_encontro": True}]})
    s.enviar("Depreciação do kit R$ 60 + imposto mensal R$ 75 + agenda R$ 65 = R$ 200",
             {"custos_fixos": [{"item": "Depreciação do kit", "valor": 60, "periodicidade": "mensal"},
                               {"item": "Imposto", "valor": 75, "periodicidade": "mensal"},
                               {"item": "Agenda", "valor": 65, "periodicidade": "mensal"}]})
    s.enviar("Uns 5 clientes de plano (estimativa)",
             {"quantidade_clientes": 5, "campos_estimados": ["quantidade_clientes"]})
    r = s.enviar("Pix, sem taxa", {"taxa_maquininha_pct": 0})
    c.checar("Mostra os dados e pede confirmação", "pede confirmação",
             "sim" if pediu_confirmacao(r["resposta"]) else "não", pediu_confirmacao(r["resposta"]),
             CAUSA_SEM_CONFIRMACAO)
    r = s.enviar("Sim", {"intencao": "confirmar"})
    m = r["metricas"] or {}
    c.checar("Calcula sem exigir meta de renda", "resultado",
             "calculou" if r["metricas"] else "pediu: " + ", ".join(r["informacoes_faltantes"]),
             r["metricas"] is not None)
    c.numero("Custos mensais registrados", 200.00, m.get("custos_fixos_mensais"))
    c.numero("Material por cliente no mês (4 × R$ 20)", 80.00, m.get("custo_por_cliente_mensal"), CAUSA_EXTRACAO)
    c.numero("Custos mensais rateados por cliente (200 ÷ 5)", 40.00, m.get("rateio_por_cliente"))
    c.numero("Total de custos por cliente", 120.00, m.get("custos_por_cliente"))
    c.numero("Sobra por cliente", 280.00, m.get("sobra_por_cliente"))
    c.numero("Valor por hora (280 ÷ 8h)", 35.00, m.get("valor_hora_real"))
    c.checar("Texto mostra sobra por cliente (R$ 280,00) e valor por hora (R$ 35,00)", "os dois valores",
             "sim" if valor_no_texto(r["resposta"], 280) and valor_no_texto(r["resposta"], 35) else "não",
             valor_no_texto(r["resposta"], 280) and valor_no_texto(r["resposta"], 35))
    c.checar("Marca a divisão por 5 clientes como 'estimativa sua'", "estimativa sua",
             "sim" if contem(r["resposta"], "estimativa sua") else "não", contem(r["resposta"], "estimativa sua"),
             CAUSA_SEM_ESTIMATIVA)
    return c


def _num(v) -> str:
    return engine._fmt_num(v or 0)


# Relatório

def gerar_relatorio(cenarios: list[Cenario], modo_ia: bool) -> str:
    cenarios_validos = [c for c in cenarios if not c.indisponivel]
    nao_avaliados = [c.nome.split(".")[0] for c in cenarios if c.indisponivel]
    total = sum(len(c.itens) for c in cenarios_validos)
    ok = sum(1 for c in cenarios_validos for i in c.itens if i["ok"] is True)
    falhas = sum(1 for c in cenarios_validos for i in c.itens if i["ok"] is False)
    pend = total - ok - falhas
    linhas = [
        "# Relatório: casos de teste do Preço Justo",
        "",
        f"Execução: {dt.datetime.now():%d/%m/%Y %H:%M} | Modo: {'IA real (Gemini)' if modo_ia else 'offline (extração simulada)'}",
        "",
        f"**{ok} de {total} verificações bateram com o gabarito.** {falhas} não bateram"
        + (f" e {pend} dependem do modo `--ia`." if pend else "."),
        "",
        "",
    ]
    if nao_avaliados:
        linhas += [f"⚠️ **Cenários não avaliados por instabilidade da API:** {', '.join(nao_avaliados)}. "
                   "O total acima considera só os cenários que rodaram.", ""]
    causas = {}
    for c in cenarios:
        for i in c.itens:
            if i["ok"] is False and i["causa"]:
                causas.setdefault(i["causa"], set()).add(c.nome.split(".")[0])
    if causas:
        linhas += ["## Causas principais", ""]
        for causa, cen in sorted(causas.items(), key=lambda x: -len(x[1]))[:5]:
            rotulo = "Cenários" if len(cen) > 1 else "Cenário"
            linhas.append(f"- **{rotulo} {', '.join(sorted(cen))}:** {causa}")
        linhas.append("")

    linhas += ["## Detalhe por cenário", ""]
    for c in cenarios:
        if c.indisponivel:
            linhas += [f"### {c.nome}: NÃO AVALIADO", "",
                       "A API do Gemini não respondeu (instabilidade do Google). Rode de novo mais tarde.", ""]
            continue
        erros = [i for i in c.itens if i["ok"] is False]
        acertos = sum(1 for i in c.itens if i["ok"] is True)
        linhas.append(f"### {c.nome}: {acertos}/{len(c.itens)}")
        linhas.append("")
        if not erros:
            linhas += ["Tudo bateu com o gabarito.", ""]
            continue
        linhas += ["| Não bateu | Esperado | Obtido | Provável causa |", "|---|---|---|---|"]
        for i in erros:
            linhas.append(f"| {i['verificacao']} | {i['esperado']} | {i['obtido']} | {i['causa']} |")
        linhas.append("")
    return "\n".join(linhas)


def main():
    parser = argparse.ArgumentParser(description="Avalia o engine contra os 5 cenários do gabarito.")
    parser.add_argument("--ia", action="store_true", help="usa o Gemini real (gasta ~35 requisições)")
    parser.add_argument("--saida", default="relatorio_casos.md")
    parser.add_argument("--cenarios", default="1,2,3,4,5", help="ex: --cenarios 1,3 (roda só esses)")
    args = parser.parse_args()

    cenarios = []
    nomes = {cenario_1: "1. Caminho feliz (domicílio)", cenario_2: "2. Comparação de preços (R$ 80 x R$ 60)",
             cenario_3: "3. Pacote x unidade", cenario_4: "4. Dados incompletos e inválidos",
             cenario_5: "5. Custos mensais e plano mensal"}
    escolhidos = {int(x) for x in args.cenarios.split(",") if x.strip()}
    todas = (cenario_1, cenario_2, cenario_3, cenario_4, cenario_5)
    for funcao in [f for n, f in enumerate(todas, 1) if n in escolhidos]:
        print(f"Rodando {nomes[funcao]}...")
        try:
            c = funcao(args.ia)
        except CotaEsgotada:
            print("\nCota da API esgotada (erro 429 em todos os modelos). Avaliação interrompida.")
            print("Espere a cota reiniciar ou use uma chave de outra conta Google no .env.")
            print("Os cenários que já rodaram estão no relatório.")
            break
        except IAIndisponivel as exc:
            c = Cenario(nomes[funcao], args.ia)
            c.indisponivel = str(exc)
            print(f"{c.nome:45} NÃO AVALIADO (API indisponível)")
            cenarios.append(c)
            if "GEMINI_API_KEY" in str(exc) or "não instalado" in str(exc):
                print(f"Erro de configuração: {exc}")
                sys.exit(2)
            continue
        cenarios.append(c)
        ok = sum(1 for i in c.itens if i["ok"] is True)
        falhas = sum(1 for i in c.itens if i["ok"] is False)
        print(f"{c.nome:45} {ok:>3} ok  {falhas:>3} falhas")

    with open(args.saida, "w", encoding="utf-8") as f:
        f.write(gerar_relatorio(cenarios, args.ia))
    if args.ia:
        print(f"Configuração usada: modelo {engine.MODELO_GEMINI}, reserva {engine.MODELOS_RESERVA}, "
              f"timeout {engine.TIMEOUT_LLM_SEGUNDOS:g}s, prazo total {engine.PRAZO_TOTAL_SEGUNDOS:g}s, "
              f"raciocínio {engine.NIVEL_RACIOCINIO}")
    print(f"\nRelatório salvo em {args.saida}")
    if any(c.indisponivel for c in cenarios):
        sys.exit(2)
    sys.exit(1 if any(i["ok"] is False for c in cenarios for i in c.itens) else 0)


if __name__ == "__main__":
    main()