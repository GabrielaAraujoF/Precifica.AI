"""
test_engine.py — Testes do módulo Preço Justo.

Rodar:  python -m unittest -v test_engine.py      (ou: pytest -v)

Os testes principais NÃO dependem da API: a extração é substituída por um
"FakeExtrator" que devolve o JSON que a LLM deveria produzir. Assim o motor
financeiro e o fluxo de estado são validados de forma determinística.

Testes de integração com o Gemini real rodam apenas com:
    GEMINI_API_KEY=... RUN_INTEGRATION=1 python -m unittest -v test_engine.py
"""

import copy
import os
import logging
import unittest
from unittest import mock

import engine

logging.getLogger("preco_justo.engine").setLevel(logging.CRITICAL)  # saída limpa
from engine import (
    atualizar_estado,
    calcular_metricas_financeiras,
    formatar_explicacao_humana,
    normalizar_extracao,
    para_float,
    processar_mensagem_usuario,
    simular_novo_preco,
)


def resposta_llm(**campos):
    """Monta um JSON no formato de saída da LLM, com defaults vazios."""
    base = {
        "intencao": "informar_dados",
        "categoria_servico": None,
        "nome_servico": None,
        "preco_cobrado": None,
        "novo_preco_simulado": None,
        "tempo_horas": None,
        "valor_hora_desejado": None,
        "materiais": [],
        "deslocamento": None,
        "outros_gastos": None,
        "taxa_maquininha_pct": None,
        "campo_corrigido": None,
        "informacoes_faltantes": [],
    }
    base.update(campos)
    return base


class FakeExtrator:
    """Substitui a LLM: devolve um JSON fixo e registra o que recebeu."""

    def __init__(self, saida):
        self.saida = saida
        self.chamadas = []

    def __call__(self, mensagem, estado):
        self.chamadas.append((mensagem, copy.deepcopy(estado)))
        if isinstance(self.saida, Exception):
            raise self.saida
        return copy.deepcopy(self.saida)


# --------------------------------------------------------------------------- #
# Cenários obrigatórios
# --------------------------------------------------------------------------- #

class TestCenario1ManicureInicio(unittest.TestCase):
    MENSAGEM = (
        "Cobro 60 reais para fazer unha em gel. Demoro 2 horas. Gasto 15 de gel "
        "e lixas, e 5 de transporte. Quero ganhar 15 reais por hora."
    )
    SAIDA_LLM = resposta_llm(
        categoria_servico="Beleza",
        nome_servico="Unha em gel",
        preco_cobrado=60.0,
        tempo_horas=2.0,
        valor_hora_desejado=15.0,
        materiais=[{"item": "gel e lixas", "valor": 15.0}],
        deslocamento=5.0,
    )

    def setUp(self):
        self.r = processar_mensagem_usuario(self.MENSAGEM, extrator=FakeExtrator(self.SAIDA_LLM))

    def test_intencao_e_sucesso(self):
        self.assertTrue(self.r["sucesso"])
        self.assertEqual(self.r["intencao"], "informar_dados")
        self.assertEqual(self.r["informacoes_faltantes"], [])

    def test_custo_total_50(self):
        m = self.r["metricas"]
        self.assertEqual(m["custo_direto"], 20.00)
        self.assertEqual(m["custo_mao_obra"], 30.00)
        self.assertEqual(m["custo_total"], 50.00)

    def test_sobra_7_com_taxa_5pct(self):
        m = self.r["metricas"]
        self.assertEqual(m["taxa_maquininha_pct"], 0.05)  # taxa padrão aplicada
        self.assertEqual(m["valor_liquido"], 57.00)
        self.assertEqual(m["sobra_liquida"], 7.00)
        self.assertEqual(m["preco_minimo"], 52.63)

    def test_resposta_legivel(self):
        self.assertIn("**Unha em gel**", self.r["resposta"])
        self.assertIn("R$ 7,00", self.r["resposta"])
        self.assertIn("R$ 52,63", self.r["resposta"])


class TestCenario2CosturaCorrecao(unittest.TestCase):
    ESTADO_INICIAL = {
        "categoria_servico": "Costura",
        "nome_servico": "Ajuste de vestido",
        "preco_cobrado": 150.0,
        "tempo_horas": 3.0,
        "valor_hora_desejado": 25.0,
        "materiais": [{"item": "Tecido", "valor": 20.0}, {"item": "Linha e zíper", "valor": 8.0}],
        "deslocamento": None,
        "outros_gastos": None,
        "taxa_maquininha_pct": 0.05,
    }
    SAIDA_LLM = resposta_llm(
        intencao="corrigir_dado",
        materiais=[{"item": "tecido", "valor": 35.0}],
        campo_corrigido="materiais",
    )

    def setUp(self):
        self.estado_original = copy.deepcopy(self.ESTADO_INICIAL)
        self.r = processar_mensagem_usuario(
            "Ajusta aí, na verdade o tecido custou 35 reais.",
            estado_atual=self.ESTADO_INICIAL,
            extrator=FakeExtrator(self.SAIDA_LLM),
        )

    def test_intencao_corrigir(self):
        self.assertTrue(self.r["sucesso"])
        self.assertEqual(self.r["intencao"], "corrigir_dado")

    def test_tecido_atualizado_para_35(self):
        materiais = self.r["estado"]["materiais"]
        self.assertEqual(len(materiais), 2)  # não duplicou o tecido
        self.assertEqual(materiais[0], {"item": "Tecido", "valor": 35.0})

    def test_restante_preservado(self):
        e = self.r["estado"]
        self.assertEqual(e["materiais"][1], {"item": "Linha e zíper", "valor": 8.0})
        self.assertEqual(e["preco_cobrado"], 150.0)
        self.assertEqual(e["tempo_horas"], 3.0)
        self.assertEqual(e["valor_hora_desejado"], 25.0)
        self.assertEqual(e["nome_servico"], "Ajuste de vestido")

    def test_metricas_recalculadas(self):
        m = self.r["metricas"]
        self.assertEqual(m["custo_direto"], 43.00)   # 35 + 8
        self.assertEqual(m["custo_total"], 118.00)   # 43 + 3*25
        self.assertEqual(m["sobra_liquida"], 24.50)  # 142.50 - 118
        self.assertIn("dado corrigido", self.r["resposta"])

    def test_estado_de_entrada_nao_mutado(self):
        self.assertEqual(self.ESTADO_INICIAL, self.estado_original)


class TestCenario3ArtesanatoSimulacao(unittest.TestCase):
    ESTADO_INICIAL = {
        "categoria_servico": "Artesanato",
        "nome_servico": "Caixa decorada em MDF",
        "preco_cobrado": 60.0,
        "tempo_horas": 2.0,
        "valor_hora_desejado": 10.0,
        "materiais": [{"item": "MDF", "valor": 12.0}, {"item": "Tinta", "valor": 8.0}],
        "deslocamento": None,
        "outros_gastos": None,
        "taxa_maquininha_pct": 0.05,
    }
    SAIDA_LLM = resposta_llm(intencao="simular_preco", novo_preco_simulado=90.0, preco_cobrado=60.0)

    def setUp(self):
        self.r = processar_mensagem_usuario(
            "E se eu cobrar R$ 90 em vez de R$ 60?",
            estado_atual=self.ESTADO_INICIAL,
            extrator=FakeExtrator(self.SAIDA_LLM),
        )

    def test_intencao_simular(self):
        self.assertTrue(self.r["sucesso"])
        self.assertEqual(self.r["intencao"], "simular_preco")

    def test_comparacao_sobras(self):
        s = self.r["simulacao"]
        # custo total = 12 + 8 + 2*10 = 40
        self.assertEqual(s["sobra_original"], 17.00)   # 57.00 - 40
        self.assertEqual(s["sobra_simulada"], 45.50)   # 85.50 - 40
        self.assertEqual(s["diferenca_sobra"], 28.50)
        self.assertEqual(s["preco_original"], 60.00)
        self.assertEqual(s["preco_simulado"], 90.00)

    def test_simulacao_nao_altera_preco_da_sessao(self):
        self.assertEqual(self.r["estado"]["preco_cobrado"], 60.0)

    def test_resposta_comparativa(self):
        self.assertIn("R$ 28,50 a mais", self.r["resposta"])
        self.assertIn("antes R$ 17,00", self.r["resposta"])


# --------------------------------------------------------------------------- #
# Motor financeiro (funções puras)
# --------------------------------------------------------------------------- #

class TestMotorFinanceiro(unittest.TestCase):
    def test_exemplo_maquiagem_do_briefing(self):
        m = calcular_metricas_financeiras(
            materiais=[{"item": "produtos", "valor": 25.0}],
            deslocamento=10.0, outros_gastos=0, tempo_horas=1.5,
            valor_hora_desejado=20.0, preco_cobrado=120.0, taxa_maquininha_pct=0.05,
        )
        self.assertEqual(m["custo_direto"], 35.00)
        self.assertEqual(m["custo_mao_obra"], 30.00)
        self.assertEqual(m["custo_total"], 65.00)
        self.assertEqual(m["valor_liquido"], 114.00)
        self.assertEqual(m["sobra_liquida"], 49.00)
        self.assertEqual(m["margem_pct"], 40.83)
        self.assertEqual(m["preco_minimo"], 68.42)

        texto = formatar_explicacao_humana({**m, "nome_servico": "Maquiagem Social"}, "informar_dados")
        self.assertIn("1,5h a R$ 20/h", texto)
        self.assertIn("margem de 40,8%", texto)
        self.assertIn("R$ 68,42", texto)

    def test_sem_taxa(self):
        m = calcular_metricas_financeiras([], 0, 0, 1, 50, 100, taxa_maquininha_pct=0)
        self.assertEqual(m["valor_liquido"], 100.00)
        self.assertEqual(m["preco_minimo"], 50.00)

    def test_none_vira_zero_e_taxa_padrao(self):
        m = calcular_metricas_financeiras(None, None, None, 1, 10, 20, taxa_maquininha_pct=None)
        self.assertEqual(m["custo_total"], 10.00)
        self.assertEqual(m["taxa_maquininha_pct"], 0.05)

    def test_preco_zero_sem_divisao_por_zero(self):
        m = calcular_metricas_financeiras([], 0, 0, 1, 10, 0)
        self.assertIsNone(m["margem_pct"])
        self.assertFalse(m["cobre_custos"])

    def test_prejuizo_gera_alerta(self):
        m = calcular_metricas_financeiras([{"item": "x", "valor": 50}], 0, 0, 2, 20, 60)
        self.assertEqual(m["sobra_liquida"], -33.00)
        self.assertFalse(m["cobre_custos"])
        self.assertIn("Faltam R$ 33,00", formatar_explicacao_humana(m, "informar_dados"))

    def test_valores_invalidos(self):
        with self.assertRaises(ValueError):
            calcular_metricas_financeiras([], -1, 0, 1, 10, 20)
        with self.assertRaises(ValueError):
            calcular_metricas_financeiras([], 0, 0, 1, 10, 20, taxa_maquininha_pct=1.0)

    def test_arredondamento_em_centavos(self):
        m = calcular_metricas_financeiras([{"item": "a", "valor": 0.1}, {"item": "b", "valor": 0.2}], 0, 0, 0, 0, 1)
        self.assertEqual(m["total_materiais"], 0.30)  # sem 0.30000000000000004

    def test_simular_sem_preco_original(self):
        estado = {"materiais": [], "tempo_horas": 1, "valor_hora_desejado": 10}
        s = simular_novo_preco(estado, 50)
        self.assertIsNone(s["diferenca_sobra"])
        self.assertEqual(s["sobra_simulada"], 37.50)


# --------------------------------------------------------------------------- #
# Normalização da saída da LLM e mescla de estado
# --------------------------------------------------------------------------- #

class TestNormalizacao(unittest.TestCase):
    def test_numeros_em_formato_brasileiro(self):
        self.assertEqual(para_float("R$ 1.234,56"), 1234.56)
        self.assertEqual(para_float("35,5"), 35.5)
        self.assertEqual(para_float("1.500"), 1500.0)
        self.assertEqual(para_float(12), 12.0)
        self.assertIsNone(para_float("abc"))
        self.assertIsNone(para_float(-3))
        self.assertIsNone(para_float(True))

    def test_taxa_em_pontos_percentuais(self):
        self.assertEqual(normalizar_extracao(resposta_llm(taxa_maquininha_pct=5))["taxa_maquininha_pct"], 0.05)
        self.assertEqual(normalizar_extracao(resposta_llm(taxa_maquininha_pct="3,5%"))["taxa_maquininha_pct"], 0.035)

    def test_intencao_invalida_inferida(self):
        d = normalizar_extracao({"intencao": "???", "novo_preco_simulado": 80})
        self.assertEqual(d["intencao"], "simular_preco")
        self.assertEqual(normalizar_extracao({})["intencao"], "informar_dados")

    def test_simulacao_com_valor_no_campo_errado(self):
        d = normalizar_extracao(resposta_llm(intencao="simular_preco", preco_cobrado=80))
        self.assertEqual(d["novo_preco_simulado"], 80.0)
        self.assertIsNone(d["preco_cobrado"])

    def test_materiais_invalidos_descartados(self):
        d = normalizar_extracao(resposta_llm(materiais=[{"item": "gel", "valor": "15"}, {"item": "", "valor": 3}, "x"]))
        self.assertEqual(d["materiais"], [{"item": "gel", "valor": 15.0}])

    def test_json_com_cercas_markdown(self):
        self.assertEqual(engine._extrair_json('```json\n{"intencao": "informar_dados"}\n```'),
                         {"intencao": "informar_dados"})
        with self.assertRaises(engine.ErroExtracao):
            engine._extrair_json("não sei")


class TestAtualizarEstado(unittest.TestCase):
    def test_informar_adiciona_e_substitui_mesmo_nome(self):
        e = atualizar_estado(
            {"materiais": [{"item": "Gel", "valor": 10}]},
            normalizar_extracao(resposta_llm(materiais=[{"item": "gel", "valor": 12}, {"item": "Lixa", "valor": 3}])),
        )
        self.assertEqual(e["materiais"], [{"item": "Gel", "valor": 12.0}, {"item": "Lixa", "valor": 3.0}])

    def test_correcao_por_nome_aproximado(self):
        e = atualizar_estado(
            {"materiais": [{"item": "Tecido de algodão", "valor": 20}]},
            normalizar_extracao(resposta_llm(intencao="corrigir_dado", materiais=[{"item": "tecido", "valor": 35}])),
        )
        self.assertEqual(e["materiais"], [{"item": "Tecido de algodão", "valor": 35.0}])

    def test_none_nao_apaga_dados(self):
        e = atualizar_estado({"tempo_horas": 2.0}, normalizar_extracao(resposta_llm(preco_cobrado=50)))
        self.assertEqual(e["tempo_horas"], 2.0)
        self.assertEqual(e["preco_cobrado"], 50.0)

    def test_correcao_de_campo_simples(self):
        e = atualizar_estado({"tempo_horas": 2.0, "preco_cobrado": 60.0},
                             normalizar_extracao(resposta_llm(intencao="corrigir_dado", tempo_horas=3,
                                                              campo_corrigido="tempo_horas")))
        self.assertEqual(e["tempo_horas"], 3.0)
        self.assertEqual(e["preco_cobrado"], 60.0)


# --------------------------------------------------------------------------- #
# Robustez do contrato com o front-end

class TestRobustez(unittest.TestCase):
    CHAVES = {"sucesso", "intencao", "dados_extraidos", "estado", "metricas",
              "simulacao", "informacoes_faltantes", "resposta", "erro"}

    def test_falha_da_ia_nao_trava_e_preserva_estado(self):
        estado = {"preco_cobrado": 60.0, "materiais": []}
        r = processar_mensagem_usuario("oi", estado, extrator=FakeExtrator(engine.ErroExtracao("timeout")))
        self.assertEqual(set(r), self.CHAVES)
        self.assertFalse(r["sucesso"])
        self.assertEqual(r["estado"]["preco_cobrado"], 60.0)
        self.assertTrue(r["resposta"])

    def test_estado_malformado_do_navegador_nao_gera_erro_500(self):
        ruins = [
            {"preco_cobrado": 60, "tempo_horas": "abc", "valor_hora_desejado": 10},
            {"preco_cobrado": 60, "tempo_horas": 1, "valor_hora_desejado": 10, "materiais": "tecido"},
            {"preco_cobrado": 60, "tempo_horas": 1, "valor_hora_desejado": 10, "materiais": [{"item": "x", "valor": "abc"}]},
        ]
        for estado in ruins:
            r = processar_mensagem_usuario("Cobro 60", estado,
                                           extrator=FakeExtrator(resposta_llm(preco_cobrado=60)))
            self.assertEqual(set(r), self.CHAVES)
            self.assertFalse(r["sucesso"])
            self.assertEqual(r["estado"], estado)  # devolve o estado recebido, sem mexer
            self.assertTrue(r["resposta"])

    def test_mensagem_vazia(self):
        extrator = FakeExtrator(resposta_llm())
        r = processar_mensagem_usuario("   ", extrator=extrator)
        self.assertEqual(set(r), self.CHAVES)
        self.assertEqual(extrator.chamadas, [])  # nem chama a IA

    def test_dados_incompletos_pede_o_que_falta(self):
        r = processar_mensagem_usuario("Cobro 80 na escova",
                                       extrator=FakeExtrator(resposta_llm(preco_cobrado=80)))
        self.assertTrue(r["sucesso"])
        self.assertEqual(r["informacoes_faltantes"], ["tempo_horas", "valor_hora_desejado"])
        self.assertIsNone(r["metricas"])
        self.assertIn("quanto tempo leva", r["resposta"])

    def test_simulacao_sem_custos_pede_dados(self):
        r = processar_mensagem_usuario("E se eu cobrar 90?",
                                       extrator=FakeExtrator(resposta_llm(intencao="simular_preco",
                                                                          novo_preco_simulado=90)))
        self.assertIsNone(r["simulacao"])
        self.assertIn("tempo_horas", r["informacoes_faltantes"])

    def test_fluxo_completo_em_varias_mensagens(self):
        r1 = processar_mensagem_usuario("Cobro 100 no corte", extrator=FakeExtrator(
            resposta_llm(preco_cobrado=100, nome_servico="Corte")))
        r2 = processar_mensagem_usuario("Levo 1h e quero 40/h", r1["estado"], extrator=FakeExtrator(
            resposta_llm(tempo_horas=1, valor_hora_desejado=40)))
        self.assertEqual(r2["metricas"]["sobra_liquida"], 55.00)  # 95 - 40
        r3 = processar_mensagem_usuario("E se eu cobrar 80?", r2["estado"], extrator=FakeExtrator(
            resposta_llm(intencao="simular_preco", novo_preco_simulado=80)))
        self.assertEqual(r3["simulacao"]["diferenca_sobra"], -19.00)
        self.assertIn("a menos", r3["resposta"])

    def test_contexto_do_estado_enviado_para_ia(self):
        conteudo = engine._montar_conteudo("o tecido foi 35", {"materiais": [{"item": "Tecido", "valor": 20}]})
        self.assertIn("ESTADO ATUAL", conteudo)
        self.assertIn("Tecido", conteudo)
        self.assertEqual(engine._montar_conteudo("oi", None), "oi")


# --------------------------------------------------------------------------- #
# Resiliência da chamada à IA (cliente Gemini simulado, sem rede)
# --------------------------------------------------------------------------- #

class ErroAPIFalso(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.code = code


class ClienteFalso:
    """Imita cliente.models.generate_content; cada item do roteiro é um erro ou um JSON."""

    def __init__(self, roteiro):
        self.roteiro = list(roteiro)
        self.modelos_chamados = []
        self.models = self

    def generate_content(self, model, contents, config):
        self.modelos_chamados.append(model)
        item = self.roteiro.pop(0)
        if isinstance(item, Exception):
            raise item
        return type("Resp", (), {"text": item})()


class TestResilienciaGemini(unittest.TestCase):
    JSON_OK = '{"intencao": "simular_preco", "novo_preco_simulado": 90}'

    def rodar(self, roteiro):
        cliente = ClienteFalso(roteiro)
        with mock.patch.object(engine, "_obter_cliente", return_value=cliente), \
             mock.patch.object(engine, "_config_geracao", return_value=None), \
             mock.patch.object(engine, "MODELO_GEMINI", "principal"), \
             mock.patch.object(engine, "MODELOS_RESERVA", ["reserva"]), \
             mock.patch.object(engine, "TENTATIVAS_POR_MODELO", 2), \
             mock.patch.object(engine.time, "sleep"):
            try:
                return engine.extrair_dados_com_ia("E se eu cobrar 90?"), cliente
            except engine.ErroExtracao as exc:
                return exc, cliente

    def test_503_tenta_de_novo_no_mesmo_modelo(self):
        r, c = self.rodar([ErroAPIFalso(503), self.JSON_OK])
        self.assertEqual(r["novo_preco_simulado"], 90.0)
        self.assertEqual(c.modelos_chamados, ["principal", "principal"])

    def test_503_persistente_usa_modelo_reserva(self):
        r, c = self.rodar([ErroAPIFalso(503), ErroAPIFalso(503), self.JSON_OK])
        self.assertEqual(r["intencao"], "simular_preco")
        self.assertEqual(c.modelos_chamados, ["principal", "principal", "reserva"])

    def test_404_pula_direto_para_reserva(self):
        r, c = self.rodar([ErroAPIFalso(404), self.JSON_OK])
        self.assertEqual(c.modelos_chamados, ["principal", "reserva"])

    def test_429_cota_esgotada_pula_direto_para_reserva(self):
        r, c = self.rodar([ErroAPIFalso(429), self.JSON_OK])
        self.assertEqual(r["intencao"], "simular_preco")
        self.assertEqual(c.modelos_chamados, ["principal", "reserva"])

    def test_504_pula_direto_para_reserva(self):
        r, c = self.rodar([ErroAPIFalso(504), self.JSON_OK])
        self.assertEqual(r["intencao"], "simular_preco")
        self.assertEqual(c.modelos_chamados, ["principal", "reserva"])

    def test_deteccao_familia_gemini3(self):
        self.assertTrue(engine._eh_gemini3("gemini-3.8-flash"))
        self.assertTrue(engine._eh_gemini3("gemini-3.5-flash-lite"))
        self.assertTrue(engine._eh_gemini3("gemini-flash-latest"))
        self.assertFalse(engine._eh_gemini3("gemini-2.5-flash"))
        self.assertFalse(engine._eh_gemini3("gemini-1.5-flash"))

    def test_chave_invalida_falha_na_hora(self):
        r, c = self.rodar([ErroAPIFalso(401)])
        self.assertIsInstance(r, engine.ErroExtracao)
        self.assertEqual(c.modelos_chamados, ["principal"])

    def test_prazo_total_interrompe_tentativas(self):
        with mock.patch.object(engine, "PRAZO_TOTAL_SEGUNDOS", 0):
            r, c = self.rodar([self.JSON_OK])
        self.assertIsInstance(r, engine.ErroExtracao)
        self.assertEqual(c.modelos_chamados, [])

    def test_tudo_falha_vira_erro_amigavel(self):
        r, c = self.rodar([ErroAPIFalso(503)] * 4)
        self.assertIsInstance(r, engine.ErroExtracao)
        self.assertIn("Nenhum modelo respondeu", str(r))


# --------------------------------------------------------------------------- #
# Modelo recorrente (assinaturas, mensalidades, aulas...)
# --------------------------------------------------------------------------- #

class TestCenarioRecorrenteAssinatura(unittest.TestCase):
    MENSAGEM = (
        "Tenho um produto digital que funciona por assinatura, tenho 12 assinantes que pagam R$150 por mês, "
        "meus custos são 370 anuais de domínio, 120 mensais de assinaturas de agentes de ia, 400 com "
        "desenvolvimento, e passo cerca de 3 horas por dia trabalhando para este produto. Quero ganhar "
        "R$5000 por mês, quanto devo cobrar"
    )
    SAIDA_LLM = resposta_llm(
        modelo_cobranca="recorrente",
        nome_servico="Produto digital por assinatura",
        preco_cobrado=150.0,
        periodicidade_preco="mensal",
        quantidade_clientes=12.0,
        custos_fixos=[
            {"item": "Domínio", "valor": 370.0, "periodicidade": "anual", "por_cliente": False},
            {"item": "Assinaturas de agentes de IA", "valor": 120.0, "periodicidade": "mensal", "por_cliente": False},
            {"item": "Desenvolvimento", "valor": 400.0, "periodicidade": None, "por_cliente": False},
        ],
        horas_trabalhadas=3.0,
        periodicidade_horas="diario",
        renda_desejada=5000.0,
        periodicidade_renda="mensal",
    )

    def setUp(self):
        self.r = processar_mensagem_usuario(self.MENSAGEM, extrator=FakeExtrator(self.SAIDA_LLM))
        self.m = self.r["metricas"]

    def test_contrato_de_saida_inalterado(self):
        self.assertEqual(set(self.r), TestRobustez.CHAVES)
        self.assertTrue(self.r["sucesso"])
        self.assertEqual(self.r["estado"]["modelo_cobranca"], "recorrente")
        self.assertEqual(self.m["modelo_cobranca"], "recorrente")

    def test_custos_convertidos_para_mes(self):
        self.assertEqual(self.m["custos_fixos"][0]["valor_mensal"], 30.83)  # 370 / 12
        self.assertEqual(self.m["custos_fixos_mensais"], 550.83)            # 30,83 + 120 + 400
        self.assertEqual(self.m["custo_total"], 5550.83)                    # + renda de 5.000

    def test_horas_e_valor_hora_implicito(self):
        self.assertEqual(self.m["horas_mensais"], 66.0)          # 3h x 22 dias
        self.assertEqual(self.m["valor_hora_implicito"], 75.76)  # 5000 / 66

    def test_situacao_atual_e_preco_recomendado(self):
        self.assertEqual(self.m["receita_bruta"], 1800.00)
        self.assertEqual(self.m["valor_liquido"], 1710.00)
        self.assertEqual(self.m["sobra_liquida"], -3840.83)
        self.assertFalse(self.m["cobre_custos"])
        self.assertEqual(self.m["preco_minimo"], 486.92)        # 5550,83 / (12 x 0,95)
        self.assertEqual(self.m["clientes_necessarios"], 39)    # 5550,83 / 142,50

    def test_premissas_explicitadas(self):
        self.assertIn('"Desenvolvimento" como gasto mensal', self.r["resposta"])
        self.assertIn("22 dias de trabalho por mês", self.r["resposta"])
        self.assertIn("R$ 486,92 por mês por cliente", self.r["resposta"])
        self.assertIn("**39 clientes**", self.r["resposta"])

    def test_simular_novo_preco(self):
        r = processar_mensagem_usuario("E se eu cobrar 500?", self.r["estado"], extrator=FakeExtrator(
            resposta_llm(intencao="simular_preco", novo_preco_simulado=500)))
        s = r["simulacao"]
        self.assertEqual(s["sobra_original"], -3840.83)
        self.assertEqual(s["sobra_simulada"], 149.17)
        self.assertEqual(s["diferenca_sobra"], 3990.00)
        self.assertEqual(r["estado"]["preco_cobrado"], 150.0)  # sessão preservada

    def test_simular_mais_clientes(self):
        r = processar_mensagem_usuario("E se eu tiver 40 assinantes?", self.r["estado"], extrator=FakeExtrator(
            resposta_llm(intencao="simular_preco", novo_quantidade_clientes=40)))
        s = r["simulacao"]
        self.assertEqual(s["clientes_simulado"], 40.0)
        self.assertEqual(s["metricas_simuladas"]["preco_minimo"], 146.07)
        self.assertEqual(r["estado"]["quantidade_clientes"], 12.0)
        self.assertIn("com 12: R$ 486,92", r["resposta"])

    def test_corrigir_custo_para_gasto_unico(self):
        r = processar_mensagem_usuario("O desenvolvimento foi um gasto único", self.r["estado"],
                                       extrator=FakeExtrator(resposta_llm(
                                           intencao="corrigir_dado",
                                           custos_fixos=[{"item": "desenvolvimento", "valor": 400, "periodicidade": "unico"}])))
        self.assertEqual(len(r["estado"]["custos_fixos"]), 3)
        self.assertEqual(r["metricas"]["custos_fixos_mensais"], 184.17)  # 30,83 + 120 + 33,33
        self.assertIn("dividido em 12 meses", r["resposta"])

    def test_preco_anual(self):
        r = processar_mensagem_usuario("x", extrator=FakeExtrator(resposta_llm(
            modelo_cobranca="recorrente", quantidade_clientes=10, preco_cobrado=1200,
            periodicidade_preco="anual", renda_desejada=2000, periodicidade_renda="mensal")))
        m = r["metricas"]
        self.assertEqual(m["receita_bruta"], 1000.00)          # 10 x 1200 / 12
        self.assertEqual(m["preco_minimo"], 210.53)            # por mês
        self.assertEqual(m["preco_minimo_periodo"], 2526.32)   # por ano
        self.assertIn("R$ 2.526,32 por ano", r["resposta"])

    def test_sem_preco_calcula_so_recomendacao(self):
        r = processar_mensagem_usuario("x", extrator=FakeExtrator(resposta_llm(
            modelo_cobranca="recorrente", quantidade_clientes=20, renda_desejada=3000,
            periodicidade_renda="mensal",
            custos_fixos=[{"item": "Servidor", "valor": 5, "periodicidade": "mensal", "por_cliente": True}])))
        m = r["metricas"]
        self.assertIsNone(m["sobra_liquida"])
        self.assertEqual(m["custo_total"], 3100.00)   # 5 x 20 + 3000
        self.assertEqual(m["preco_minimo"], 163.16)   # 3100 / (20 x 0,95)

    def test_renda_por_horas_e_valor_hora(self):
        r = processar_mensagem_usuario("x", extrator=FakeExtrator(resposta_llm(
            modelo_cobranca="recorrente", quantidade_clientes=8, horas_trabalhadas=20,
            periodicidade_horas="semanal", valor_hora_desejado=50)))
        self.assertEqual(r["metricas"]["origem_renda"], "horas_x_valor_hora")
        self.assertEqual(r["metricas"]["renda_desejada_mensal"], 4333.33)  # 20h x 52/12 x 50

    def test_pede_o_que_falta_no_recorrente(self):
        r = processar_mensagem_usuario("Tenho alunos de inglês", extrator=FakeExtrator(resposta_llm(
            modelo_cobranca="recorrente", preco_cobrado=300, periodicidade_preco="mensal")))
        self.assertEqual(r["informacoes_faltantes"], ["quantidade_clientes", "renda_desejada"])
        self.assertIn("quantos clientes ou assinantes", r["resposta"])

    def test_modelo_inferido_sem_classificacao_da_ia(self):
        d = normalizar_extracao(resposta_llm(quantidade_clientes=5))
        self.assertEqual(d["modelo_cobranca"], "recorrente")
        self.assertIsNone(normalizar_extracao(resposta_llm(preco_cobrado=50))["modelo_cobranca"])

    def test_normalizacao_de_periodos(self):
        d = normalizar_extracao(resposta_llm(
            periodicidade_preco="por mês", periodicidade_horas="Diário",
            custos_fixos=[{"item": "a", "valor": "1.200,00", "periodicidade": "ao ano"},
                          {"item": "b", "valor": 10, "periodicidade": "uma vez"},
                          {"item": "c", "valor": 10, "periodicidade": "???"}]))
        self.assertEqual(d["periodicidade_preco"], "mensal")
        self.assertEqual(d["periodicidade_horas"], "diario")
        self.assertEqual([c["periodicidade"] for c in d["custos_fixos"]], ["anual", "unico", None])
        self.assertEqual(d["custos_fixos"][0]["valor"], 1200.0)

    def test_sessao_por_servico_nao_vira_recorrente_por_engano(self):
        estado = {"modelo_cobranca": "recorrente", "quantidade_clientes": 3, "renda_desejada": 900}
        e = atualizar_estado(estado, normalizar_extracao(resposta_llm(modelo_cobranca="por_servico")))
        self.assertEqual(e["modelo_cobranca"], "recorrente")  # não volta sozinho


class TestPorServicoComCustosFixos(unittest.TestCase):
    BASE = dict(nome_servico="Unha em gel", preco_cobrado=60, tempo_horas=2, valor_hora_desejado=15,
                materiais=[{"item": "gel", "valor": 15}], deslocamento=5,
                custos_fixos=[{"item": "Aluguel do salão", "valor": 800, "periodicidade": "mensal"}])

    def test_custos_fixos_rateados_por_atendimento(self):
        r = processar_mensagem_usuario("x", extrator=FakeExtrator(resposta_llm(**self.BASE, atendimentos_por_mes=40)))
        m = r["metricas"]
        self.assertEqual(m["modelo_cobranca"], "por_servico")
        self.assertEqual(m["custo_direto"], 20.00)           # fórmula original intacta
        self.assertEqual(m["custos_fixos_rateados"], 20.00)  # 800 / 40
        self.assertEqual(m["custo_total"], 70.00)
        self.assertEqual(m["sobra_liquida"], -13.00)
        self.assertIn("R$ 800,00/mês ÷ 40 atendimentos", r["resposta"])

    def test_custos_fixos_sem_atendimentos_avisa(self):
        r = processar_mensagem_usuario("x", extrator=FakeExtrator(resposta_llm(**self.BASE)))
        self.assertEqual(r["metricas"]["custo_total"], 50.00)  # não inventa rateio
        self.assertIn("quantos atendimentos você faz por mês", r["resposta"])



# Integração real com o Gemini

@unittest.skipUnless(os.getenv("GEMINI_API_KEY") and os.getenv("RUN_INTEGRATION") == "1",
                     "Defina GEMINI_API_KEY e RUN_INTEGRATION=1 para testar com o Gemini real.")
class TestIntegracaoGemini(unittest.TestCase):
    def test_cenario1_real(self):
        r = processar_mensagem_usuario(TestCenario1ManicureInicio.MENSAGEM)
        self.assertTrue(r["sucesso"], r["erro"])
        self.assertEqual(r["intencao"], "informar_dados")
        self.assertEqual(r["metricas"]["custo_total"], 50.00)
        self.assertEqual(r["metricas"]["sobra_liquida"], 7.00)

    def test_cenario2_real(self):
        r = processar_mensagem_usuario("Ajusta aí, na verdade o tecido custou 35 reais.",
                                       TestCenario2CosturaCorrecao.ESTADO_INICIAL)
        self.assertTrue(r["sucesso"], r["erro"])
        self.assertEqual(r["intencao"], "corrigir_dado")
        tecido = [m for m in r["estado"]["materiais"] if "tecido" in m["item"].lower()]
        self.assertEqual(tecido[0]["valor"], 35.0)
        self.assertEqual(len(r["estado"]["materiais"]), 2)

    def test_cenario3_real(self):
        r = processar_mensagem_usuario("E se eu cobrar R$ 90 em vez de R$ 60?",
                                       TestCenario3ArtesanatoSimulacao.ESTADO_INICIAL)
        self.assertTrue(r["sucesso"], r["erro"])  # mostra o motivo se a API falhar
        self.assertEqual(r["intencao"], "simular_preco")
        self.assertEqual(r["simulacao"]["diferenca_sobra"], 28.50)

    def test_recorrente_real(self):
        r = processar_mensagem_usuario(TestCenarioRecorrenteAssinatura.MENSAGEM)
        self.assertTrue(r["sucesso"], r["erro"])
        self.assertEqual(r["estado"]["modelo_cobranca"], "recorrente")
        m = r["metricas"]
        self.assertEqual(m["quantidade_clientes"], 12.0)
        self.assertEqual(m["renda_desejada_mensal"], 5000.00)
        self.assertEqual(m["horas_mensais"], 66.0)
        self.assertEqual(m["preco_cobrado"], 150.00)
        self.assertEqual(len(m["custos_fixos"]), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)