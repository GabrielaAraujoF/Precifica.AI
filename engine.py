# mensagem do usuário
# extrair_dados_com_ia()        (LLM: SÓ converte texto em JSON)
# normalizar_extracao()         (valida e corrige tipos vindos da LLM)
# atualizar_estado()            (mescla com a sessão)
# confirmação dos dados         (mostra o que entendeu antes do primeiro cálculo)
# calcular_metricas_financeiras() / calcular_metricas_recorrentes() / simular_novo_preco()   (matemática)
# formatar_explicacao_humana()  (texto final para o chat)
#
# Dois modelos de cobrança:
#   por_servico: preço por atendimento/peça (maquiagem, manicure, costura, artesanato...)
#   recorrente:  preço por período (plano mensal, assinatura, hospedagem, aulas...)
#
# Resultado principal: sobra = preço - custos (materiais, deslocamento, taxa, custos mensais rateados)
# e valor por hora real = sobra / horas. A meta por hora/renda é opcional.


from __future__ import annotations

import copy
import json
import logging
import math
import os
import re
import time
import unicodedata
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Callable, Optional

try:  # python-dotenv é opcional em tempo de execução (útil em dev)
    from dotenv import load_dotenv

    load_dotenv(encoding="utf-8-sig")  # tolera BOM (Set-Content do PowerShell 5)
except ImportError:  # pragma: no cover
    pass

logger = logging.getLogger("preco_justo.engine")

# Configuração

TAXA_PADRAO = 0.05  # usada (e avisada) quando a forma de pagamento não foi informada
# Mostra os dados entendidos e espera "sim" antes do primeiro cálculo
PEDIR_CONFIRMACAO = os.getenv("PEDIR_CONFIRMACAO", "1") != "0"

MODELO_GEMINI = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
# Modelos reserva, tentados em ordem quando o principal falha. Separados por vírgula.
MODELOS_RESERVA = [
    m.strip()
    for m in os.getenv("GEMINI_MODELOS_RESERVA", "gemini-3.8-flash").split(",")
    if m.strip()
]
TENTATIVAS_POR_MODELO = int(os.getenv("GEMINI_TENTATIVAS", "2"))
ESPERA_INICIAL_SEGUNDOS = float(os.getenv("GEMINI_ESPERA", "1.5"))
CODIGOS_TEMPORARIOS = {500, 502, 503}  # sobrecarga momentânea: tentar de novo
# 404 inexistente | 429 cota esgotada | 504 lento demais: vai direto ao próximo modelo
CODIGOS_TROCAR_MODELO = {404, 429, 504}
# Gemini 3+: temperatura padrão (recomendação do Google) e raciocínio mínimo (extração simples, mais rápida)
NIVEL_RACIOCINIO = os.getenv("GEMINI_THINKING", "minimal")  # minimal | low | medium | high
TIMEOUT_LLM_SEGUNDOS = float(os.getenv("GEMINI_TIMEOUT", "25"))  # por chamada
# Tempo máximo somando todas as tentativas. Depois disso o usuário recebe a mensagem amigável.
PRAZO_TOTAL_SEGUNDOS = float(os.getenv("GEMINI_PRAZO_TOTAL", "40"))

# Conversão de períodos: tudo vira base MENSAL antes das contas
DIAS_TRABALHO_MES = int(os.getenv("DIAS_TRABALHO_MES", "22"))  # para "horas por dia"
DIAS_MES = 30  # para custos diários
MESES_AMORTIZACAO = 12  # gasto único é diluído em 12 meses
PERIODOS = ("diario", "semanal", "mensal", "trimestral", "semestral", "anual", "unico")
FATOR_MENSAL = {
    "diario": Decimal(DIAS_MES),
    "semanal": Decimal(52) / Decimal(12),
    "mensal": Decimal(1),
    "trimestral": Decimal(1) / Decimal(3),
    "semestral": Decimal(1) / Decimal(6),
    "anual": Decimal(1) / Decimal(12),
    "unico": Decimal(1) / Decimal(MESES_AMORTIZACAO),
}
# Horas e renda "por dia" contam só dias de trabalho
FATOR_TRABALHO = {**FATOR_MENSAL, "diario": Decimal(DIAS_TRABALHO_MES)}
NOME_PERIODO = {
    "diario": "por dia", "semanal": "por semana", "mensal": "por mês",
    "trimestral": "por trimestre", "semestral": "por semestre", "anual": "por ano",
    "unico": "gasto único",
}

SYSTEM_PROMPT = """Você é o extrator de dados estruturados do "Preço Justo", um assistente de precificação para trabalhadores autônomos.
Sua ÚNICA função é converter a mensagem do usuário em um JSON estruturado.
NUNCA faça cálculos (nem somas, nem divisões, nem multiplicações) e não escreva nada fora do JSON.

### Intenções:
- "informar_dados": o usuário passa custos, tempos, preços ou outras informações.
- "corrigir_dado": o usuário altera algo dito antes ("na verdade...").
- "simular_preco": o usuário pergunta por um cenário alternativo ("E se eu cobrar X?", "E se eu tiver 20 clientes?").
- "confirmar": o usuário confirma que os dados estão certos ("sim", "pode calcular", "está certo").

### Formato obrigatório da saída (JSON):
{
  "intencao": "informar_dados" | "corrigir_dado" | "simular_preco" | "confirmar",
  "categoria_servico": string ou null,
  "nome_servico": string ou null,
  "modelo_cobranca": "por_servico" | "recorrente" | null,
  "preco_cobrado": float ou null,
  "periodicidade_preco": "semanal" | "mensal" | "trimestral" | "semestral" | "anual" | null,
  "novo_preco_simulado": float ou null,
  "novo_quantidade_clientes": float ou null,
  "tempo_horas": float ou null,
  "tempo_partes_horas": [float],
  "valor_hora_desejado": float ou null,
  "materiais": [
    {"item": string, "valor": float ou null, "rendimento": float ou null, "quantidade_usada": float ou null, "embalagem": boolean, "estimativa": boolean}
  ],
  "deslocamento": float ou null,
  "outros_gastos": float ou null,
  "taxa_maquininha_pct": float ou null,
  "atendimentos_por_mes": float ou null,
  "quantidade_clientes": float ou null,
  "encontros_por_mes": float ou null,
  "horas_por_encontro": float ou null,
  "custos_fixos": [
    {"item": string, "valor": float ou null, "periodicidade": "diario" | "semanal" | "mensal" | "trimestral" | "semestral" | "anual" | "unico" | null, "por_cliente": boolean, "por_encontro": boolean, "estimativa": boolean}
  ],
  "horas_trabalhadas": float ou null,
  "periodicidade_horas": "diario" | "semanal" | "mensal" | null,
  "renda_desejada": float ou null,
  "periodicidade_renda": "semanal" | "mensal" | "anual" | null,
  "campos_estimados": [strings],
  "nao_sabe": [strings],
  "campo_corrigido": string ou null
}

### Regras:
1. Converta números escritos em texto ("cinquenta reais" -> 50.0; "mil e duzentos" -> 1200.0; "1.200,50" -> 1200.5; "uma hora e meia" -> 1.5; "30 minutos" -> 0.5).
2. Copie valores negativos ou zero EXATAMENTE como o usuário escreveu ("-30" -> -30.0; "0 horas" -> 0.0). Não corrija nada.
3. Tempo em partes ("1h30 de maquiagem + 30 min de preparo + 30 min de limpeza"): liste cada parte em horas em "tempo_partes_horas" ([1.5, 0.5, 0.5]) e deixe "tempo_horas" null.
4. Materiais comprados em embalagem/pacote/caixa/frasco: "valor" é o preço da embalagem; "rendimento" é quantos atendimentos ou unidades ela rende; "quantidade_usada" é quantas unidades vão em cada atendimento. Ex: "caixa de cílios com 10 pares por R$ 35, uso 1 par" -> valor 35, rendimento 10, quantidade_usada 1; "base de R$ 120 rende uns 40 atendimentos" -> valor 120, rendimento 40, estimativa true. Se for embalagem e o rendimento não foi dito, use "embalagem": true e "rendimento": null.
5. Material gasto por inteiro em cada atendimento: só "valor". Se a pessoa disser que não gasta material, use [{"item": "Nenhum", "valor": 0}].
6. Transporte, Uber, app, gasolina ou passagem vão em "deslocamento". "Sem deslocamento" -> 0.0.
7. Pagamento: Pix, dinheiro ou "sem taxa" -> "taxa_maquininha_pct": 0.0. Cartão com taxa X% -> fração (3,5% -> 0.035). Cartão sem taxa informada -> null.
8. "estimativa": true (e nome do campo em "campos_estimados", para campos simples) quando o usuário chuta ou aproxima: "uns", "mais ou menos", "acho que", "cerca de", "e pouco", "(estimativa)".
9. "nao_sabe": nomes dos campos que o usuário diz não saber (ex: "não sei quanto gasto de material" -> ["materiais"]). Não invente valor.
10. Em "corrigir_dado", preencha SOMENTE o que mudou. Para um material ou custo, repita o nome do item como aparece no ESTADO ATUAL e só o atributo alterado (ex: "a base rende só 30" -> materiais [{"item": "<nome da base>", "rendimento": 30}]).
11. Em "simular_preco", o novo valor vai em "novo_preco_simulado"; se o usuário citar o preço atual ("em vez de R$ 60"), coloque-o em "preco_cobrado".
12. "recorrente": o cliente paga por período (plano mensal, assinatura, mensalidade, hospedagem, aulas mensais). "por_servico": cobra por atendimento ou peça. Na dúvida, null.
13. Em "recorrente": número de clientes/assinantes/alunos em "quantidade_clientes"; "4 encontros de 2 horas" -> "encontros_por_mes": 4, "horas_por_encontro": 2; tempo total trabalhado ("3 horas por dia") em "horas_trabalhadas" + "periodicidade_horas"; meta de ganho ("quero ganhar 5000 por mês") em "renda_desejada" + "periodicidade_renda". Material gasto por encontro vai em "custos_fixos" com "por_encontro": true; custo de cada cliente por mês com "por_cliente": true.
14. Gastos que se repetem todo mês mesmo sem atender (aluguel, imposto, agenda, depreciação de equipamento, domínio, softwares) vão em "custos_fixos" com a periodicidade dita; se o período não foi dito, null. Se a pessoa disser que não tem, use [{"item": "Nenhum", "valor": 0, "periodicidade": "mensal"}].
15. Em "por_servico", "atendo 40 clientes por mês" -> "atendimentos_por_mes": 40.
16. "quero ganhar R$ 50 por hora" -> "valor_hora_desejado": 50.
17. Listas sem informação ficam vazias ([]). O ESTADO ATUAL enviado junto é só contexto: não repita dados dele que o usuário não mencionou agora.
"""

INTENCOES_VALIDAS = ("informar_dados", "corrigir_dado", "simular_preco", "confirmar")
MODELOS_COBRANCA = ("por_servico", "recorrente")

# Campos numéricos simples; os da direita não podem ser zero
CAMPOS_NUMERICOS = (
    "preco_cobrado", "novo_preco_simulado", "novo_quantidade_clientes", "tempo_horas",
    "valor_hora_desejado", "deslocamento", "outros_gastos", "atendimentos_por_mes",
    "quantidade_clientes", "encontros_por_mes", "horas_por_encontro", "horas_trabalhadas", "renda_desejada",
)
NAO_PODE_SER_ZERO = {
    "preco_cobrado", "novo_preco_simulado", "novo_quantidade_clientes", "tempo_horas", "atendimentos_por_mes",
    "quantidade_clientes", "encontros_por_mes", "horas_por_encontro", "horas_trabalhadas",
}
CAMPOS_TEXTO = ("categoria_servico", "nome_servico", "campo_corrigido")
CAMPOS_PERIODO = ("periodicidade_preco", "periodicidade_horas", "periodicidade_renda")

ROTULOS = {
    "preco_cobrado": "preço", "novo_preco_simulado": "preço simulado", "novo_quantidade_clientes": "número de clientes",
    "tempo_horas": "tempo", "valor_hora_desejado": "valor da hora", "deslocamento": "deslocamento",
    "outros_gastos": "outros gastos", "atendimentos_por_mes": "atendimentos por mês",
    "quantidade_clientes": "número de clientes", "encontros_por_mes": "encontros por mês",
    "horas_por_encontro": "horas por encontro", "horas_trabalhadas": "horas trabalhadas",
    "renda_desejada": "renda desejada", "taxa_maquininha_pct": "taxa de pagamento", "materiais": "materiais",
    "custos_fixos": "custos mensais",
}

# Campos que ficam guardados na sessão (o "estado").
ESTADO_VAZIO: dict = {
    "categoria_servico": None,
    "nome_servico": None,
    "modelo_cobranca": None,
    "preco_cobrado": None,
    "periodicidade_preco": None,
    "tempo_horas": None,
    "valor_hora_desejado": None,
    "materiais": [],
    "deslocamento": None,
    "outros_gastos": None,
    "taxa_maquininha_pct": None,
    "atendimentos_por_mes": None,
    "quantidade_clientes": None,
    "encontros_por_mes": None,
    "horas_por_encontro": None,
    "custos_fixos": [],
    "horas_trabalhadas": None,
    "periodicidade_horas": None,
    "renda_desejada": None,
    "periodicidade_renda": None,
    "estimados": [],                  # campos simples que são estimativa do usuário
    "aguardando_confirmacao": False,  # mostrou o resumo e espera "sim"
    "confirmado": False,              # já calculou uma vez (correções recalculam direto)
}
CAMPOS_CONTROLE = ("estimados", "aguardando_confirmacao", "confirmado")
CAMPOS_CONTEXTO = tuple(k for k in ESTADO_VAZIO if k not in CAMPOS_CONTROLE)

# Perguntas feitas quando falta algo (as chaves "rendimento:<item>" são montadas na hora)
PERGUNTAS = {
    "preco_cobrado": "quanto você cobra pelo serviço (ex: 'cobro R$ 80')",
    "preco_recorrente": "quanto cada cliente paga por mês (ou, se ainda não tem preço, quanto quer ganhar por mês)",
    "tempo_horas": "quanto tempo leva cada atendimento, contando preparo e limpeza (ex: '1h30 de serviço + 30 min de preparo')",
    "materiais": "quanto gasta de material em cada atendimento (se compra em pacote, diga o preço e quantos atendimentos rende)",
    "deslocamento": "se tem gasto de deslocamento (ex: 'Uber ida e volta R$ 24' ou 'não tenho')",
    "taxa_maquininha_pct": "como o cliente costuma pagar (Pix, dinheiro ou cartão com taxa de X%)",
    "quantidade_clientes": "quantos clientes ou assinantes dividem esses custos no mês (ex: 'tenho 12 assinantes')",
    "custos_fixos": ("se tem gastos que se repetem todo mês, mesmo sem atender (equipamento, imposto, agenda, "
                     "aluguel, sistemas). Se não tiver, diga 'nenhum'"),
    "encontros_por_mes": "quantos encontros ou atendimentos cada cliente tem por mês",
    "material_por_cliente": "quanto gasta de material por cliente ou por encontro (se não gastar, diga 'nenhum')",
}


class ErroExtracao(Exception):
    """Falha ao obter/interpretar a resposta da LLM."""


# Conversão e normalização

_CENTAVO = Decimal("0.01")


def _dec(valor: Any) -> Decimal:
    """Converte para Decimal via str (evita herdar imprecisão do float). None -> 0."""
    if valor is None:
        return Decimal("0")
    if isinstance(valor, Decimal):
        return valor
    if isinstance(valor, bool):
        raise ValueError("Valor booleano não é numérico.")
    return Decimal(str(valor))


def _arred(valor: Decimal) -> float:
    """Arredonda para centavos (meio para cima) e devolve float para JSON."""
    return float(valor.quantize(_CENTAVO, rounding=ROUND_HALF_UP))


def _numero_bruto(valor: Any) -> Optional[float]:
    """
    Converte números vindos da LLM em float, aceitando formatos brasileiros e negativos.
    Exemplos: 35 -> 35.0 | "35,50" -> 35.5 | "R$ 1.234,56" -> 1234.56 | "-30" -> -30.0
    """
    if valor is None or isinstance(valor, bool):
        return None
    if isinstance(valor, (int, float)):
        numero = float(valor)
    elif isinstance(valor, str):
        bruto = valor.strip()
        negativo = bruto.startswith("-")
        texto = re.sub(r"[^\d,.]", "", bruto)
        if not texto or texto in {",", "."}:
            return None
        if "," in texto and "." in texto:
            if texto.rfind(",") > texto.rfind("."):  # 1.234,56
                texto = texto.replace(".", "").replace(",", ".")
            else:  # 1,234.56
                texto = texto.replace(",", "")
        elif "," in texto:
            texto = texto.replace(",", ".")
        elif re.fullmatch(r"\d{1,3}(\.\d{3})+", texto):  # 1.500 = mil e quinhentos
            texto = texto.replace(".", "")
        try:
            numero = float(texto)
        except ValueError:
            return None
        if negativo:
            numero = -numero
    else:
        return None
    if math.isnan(numero) or math.isinf(numero):
        return None
    return numero


def para_float(valor: Any) -> Optional[float]:
    """Como _numero_bruto, mas devolve None para negativos (valores de custo/preço válidos)."""
    numero = _numero_bruto(valor)
    return numero if numero is not None and numero >= 0 else None


def _normalizar_periodo(valor: Any) -> Optional[str]:
    """'Mensal', 'por mês', 'ano'... -> chave de PERIODOS (None se não reconhecer)."""
    if not isinstance(valor, str) or not valor.strip():
        return None
    texto = _chave(valor)
    sinonimos = (
        ("unico", ("unic", "uma vez", "pontual", "avulso")),
        ("trimestral", ("trimest",)),
        ("semestral", ("semest",)),
        ("semanal", ("seman",)),
        ("diario", ("diari", "dia")),
        ("mensal", ("mensal", "mes", "month")),
        ("anual", ("anual", "ano", "year")),
    )
    for chave, termos in sinonimos:
        if any(t in texto for t in termos):
            return chave
    return None


def _chave(texto: str) -> str:
    """Normaliza nomes para comparação: sem acento, minúsculo, espaços simples."""
    sem_acento = unicodedata.normalize("NFKD", str(texto))
    sem_acento = "".join(c for c in sem_acento if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", sem_acento).strip().lower()


def normalizar_extracao(bruto: Any) -> dict:
    """
    Garante que a saída da LLM tenha exatamente o schema esperado, com tipos corretos.
    Valores negativos (ou zero onde não faz sentido) NÃO são guardados: vão para
    "invalidos" para o assistente perguntar se foi erro de digitação.
    É idempotente (pode ser chamada mais de uma vez sem efeito colateral).
    """
    if not isinstance(bruto, dict):
        raise ErroExtracao("A resposta da IA não é um objeto JSON.")

    invalidos: list[dict] = list(bruto.get("invalidos") or []) if isinstance(bruto.get("invalidos"), list) else []

    def numero(campo: str, valor: Any, zero_invalido: bool) -> Optional[float]:
        n = _numero_bruto(valor)
        if n is None:
            return None
        if n < 0 or (zero_invalido and n == 0):
            invalidos.append({"campo": campo, "valor": n, "motivo": "negativo" if n < 0 else "zero"})
            return None
        return n

    dados: dict = {}
    for campo in CAMPOS_NUMERICOS:
        dados[campo] = numero(campo, bruto.get(campo), campo in NAO_PODE_SER_ZERO)

    taxa = _numero_bruto(bruto.get("taxa_maquininha_pct"))
    if taxa is not None and taxa >= 1:
        taxa = taxa / 100  # a IA devolveu em pontos percentuais
    if taxa is not None and not (0 <= taxa < 1):
        invalidos.append({"campo": "taxa_maquininha_pct", "valor": taxa, "motivo": "negativo"})
        taxa = None
    dados["taxa_maquininha_pct"] = taxa

    for campo in CAMPOS_TEXTO:
        valor = bruto.get(campo)
        dados[campo] = valor.strip() if isinstance(valor, str) and valor.strip() else None
    for campo in CAMPOS_PERIODO:
        dados[campo] = _normalizar_periodo(bruto.get(campo))
    if dados["periodicidade_preco"] == "unico":
        dados["periodicidade_preco"] = None

    # Tempo em partes: a soma é feita aqui, em Python
    partes = []
    for p in bruto.get("tempo_partes_horas") or []:
        n = numero("tempo_horas", p, False)
        if n is not None:
            partes.append(n)
    dados["tempo_partes_horas"] = partes
    if dados["tempo_horas"] is None and partes:
        soma = sum((_dec(p) for p in partes), Decimal("0"))
        if soma > 0:
            dados["tempo_horas"] = float(soma)
        else:
            invalidos.append({"campo": "tempo_horas", "valor": 0.0, "motivo": "zero"})

    materiais = []
    for m in bruto.get("materiais") or []:
        if not isinstance(m, dict) or not isinstance(m.get("item"), str) or not m["item"].strip():
            continue
        item = m["item"].strip()
        novo = {
            "item": item,
            "valor": numero(f"material:{item}", m.get("valor"), False),
            "rendimento": numero(f"rendimento:{item}", m.get("rendimento"), True),
            "quantidade_usada": numero(f"quantidade:{item}", m.get("quantidade_usada"), True),
            "embalagem": m.get("embalagem") is True,
            "estimativa": m.get("estimativa") is True,
        }
        if any(novo[k] is not None for k in ("valor", "rendimento", "quantidade_usada")) or novo["embalagem"]:
            materiais.append(novo)
    dados["materiais"] = materiais

    custos_fixos = []
    for c in bruto.get("custos_fixos") or []:
        if not isinstance(c, dict) or not isinstance(c.get("item"), str) or not c["item"].strip():
            continue
        item = c["item"].strip()
        novo = {
            "item": item,
            "valor": numero(f"custo:{item}", c.get("valor"), False),
            "periodicidade": _normalizar_periodo(c.get("periodicidade")),
            "por_cliente": c.get("por_cliente") is True,
            "por_encontro": c.get("por_encontro") is True,
            "estimativa": c.get("estimativa") is True,
        }
        if novo["valor"] is not None or novo["periodicidade"] is not None:
            custos_fixos.append(novo)
    dados["custos_fixos"] = custos_fixos

    estimados = bruto.get("campos_estimados") or []
    dados["campos_estimados"] = [c for c in estimados if isinstance(c, str) and c in ESTADO_VAZIO]
    nao_sabe = bruto.get("nao_sabe") or []
    dados["nao_sabe"] = [str(c) for c in nao_sabe if c] if isinstance(nao_sabe, list) else []
    dados["invalidos"] = invalidos

    modelo = bruto.get("modelo_cobranca")
    if modelo not in MODELOS_COBRANCA:
        # Sinais fortes de cobrança recorrente quando a IA não classificou
        sinais = (dados["quantidade_clientes"], dados["novo_quantidade_clientes"], dados["periodicidade_preco"])
        modelo = "recorrente" if any(v is not None for v in sinais) else None
    dados["modelo_cobranca"] = modelo

    simulando = dados["novo_preco_simulado"] is not None or dados["novo_quantidade_clientes"] is not None
    intencao = bruto.get("intencao")
    if intencao not in INTENCOES_VALIDAS:
        intencao = "simular_preco" if simulando else "informar_dados"
    # Proteção: LLM marcou simulação mas colocou o valor no campo errado.
    if intencao == "simular_preco" and not simulando and dados["preco_cobrado"] is not None:
        dados["novo_preco_simulado"] = dados["preco_cobrado"]
        dados["preco_cobrado"] = None
    dados["intencao"] = intencao
    return dados


def _extrair_json(texto: str) -> dict:
    """Lê o JSON da resposta, tolerando cercas ```json e texto ao redor."""
    if not texto:
        raise ErroExtracao("Resposta vazia da IA.")
    limpo = re.sub(r"```(?:json)?", "", texto).strip()
    inicio, fim = limpo.find("{"), limpo.rfind("}")
    if inicio == -1 or fim == -1:
        raise ErroExtracao("Nenhum JSON encontrado na resposta da IA.")
    try:
        return json.loads(limpo[inicio : fim + 1])
    except json.JSONDecodeError as exc:
        raise ErroExtracao(f"JSON inválido retornado pela IA: {exc}") from exc

# Integração com a LLM (Gemini)

_cache_clientes: dict = {}
# Modelos que recusaram o nível de raciocínio configurado passam a usar este (ex: "low")
_NIVEL_POR_MODELO: dict = {}
NIVEL_RACIOCINIO_RESERVA = "low"


def _obter_cliente():
    """Cria (uma vez) o cliente do SDK google-genai."""
    chave = os.getenv("GEMINI_API_KEY")
    if not chave:
        raise ErroExtracao("GEMINI_API_KEY não configurada (crie o arquivo .env).")
    if chave in _cache_clientes:
        return _cache_clientes[chave]
    try:
        from google import genai  # import tardio: testes rodam sem o pacote
        from google.genai import types
    except ImportError as exc:
        raise ErroExtracao("Pacote 'google-genai' não instalado (pip install -r requirements.txt).") from exc

    cliente = genai.Client(
        api_key=chave,
        http_options=types.HttpOptions(timeout=int(TIMEOUT_LLM_SEGUNDOS * 1000)),  # em ms
    )
    _cache_clientes[chave] = cliente
    return cliente


def _eh_gemini3(modelo: str) -> bool:
    """Família Gemini 3+ (inclui aliases '-latest', que apontam para a geração atual)."""
    return bool(re.search(r"gemini-([3-9]|\d{2,})", modelo)) or modelo.endswith("-latest")


def _config_geracao(modelo: str):
    """
    system_instruction + saída estritamente JSON.
      - Gemini 1.x/2.x: temperature=0.0 (como pede o briefing).
      - Gemini 3+: temperatura padrão + raciocínio baixo (recomendação do Google).
    O determinismo dos NÚMEROS não depende disso: toda conta é feita em Python
    e a saída da IA é validada em normalizar_extracao().
    """
    from google.genai import types

    if _eh_gemini3(modelo):
        return types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_level=_NIVEL_POR_MODELO.get(modelo, NIVEL_RACIOCINIO)),
        )
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.0,
        response_mime_type="application/json",
    )


def _montar_conteudo(mensagem: str, estado_atual: Optional[dict]) -> str:
    """Envia o estado da sessão como contexto, para a IA saber o que pode ser corrigido."""
    if not estado_atual:
        return mensagem
    contexto = {k: estado_atual.get(k) for k in CAMPOS_CONTEXTO}
    if not any(v not in (None, [], False) for v in contexto.values()):
        return mensagem
    return (
        "ESTADO ATUAL DA SESSÃO (apenas contexto):\n"
        f"{json.dumps(contexto, ensure_ascii=False)}\n\n"
        f"MENSAGEM DO USUÁRIO:\n{mensagem}"
    )


def _codigo_http(exc: Exception) -> Optional[int]:
    """Extrai o status HTTP de um erro do SDK google-genai (None se não houver)."""
    codigo = getattr(exc, "code", None)
    return codigo if isinstance(codigo, int) else None


def extrair_dados_com_ia(mensagem: str, estado_atual: Optional[dict] = None) -> dict:
    """
    Chama o Gemini e devolve o JSON já normalizado. Lança ErroExtracao em falhas.

    Resiliência:
      - 500/502/503 (sobrecarga): espera com backoff e tenta de novo; se persistir,
        passa para o próximo modelo de MODELOS_RESERVA.
      - 404 (inexistente), 429 (cota esgotada) ou 504 (lento): pula direto ao próximo.
      - JSON inválido: tenta de novo no mesmo modelo.
      - 400/401/403 (chave ou requisição inválida): falha na hora, sem insistir.
    """
    cliente = _obter_cliente()
    conteudo = _montar_conteudo(mensagem, estado_atual)
    modelos = list(dict.fromkeys([MODELO_GEMINI, *MODELOS_RESERVA]))  # sem duplicados
    falhas: list[str] = []
    limite = time.monotonic() + PRAZO_TOTAL_SEGUNDOS

    for modelo in modelos:
        for tentativa in range(1, TENTATIVAS_POR_MODELO + 1):
            if time.monotonic() >= limite:
                falhas.append("prazo total esgotado")
                raise ErroExtracao("Nenhum modelo respondeu a tempo. Tentativas: " + " | ".join(falhas))
            try:
                resposta = cliente.models.generate_content(
                    model=modelo, contents=conteudo, config=_config_geracao(modelo)
                )
                dados = normalizar_extracao(_extrair_json(resposta.text))
                if modelo != MODELO_GEMINI:
                    logger.warning("Extração feita com o modelo reserva %s", modelo)
                return dados
            except ErroExtracao as exc:  # JSON ruim: tenta de novo
                falhas.append(f"{modelo}: {exc}")
                logger.warning("%s tentativa %s: resposta inválida (%s)", modelo, tentativa, exc)
                continue
            except Exception as exc:
                codigo = _codigo_http(exc)
                falhas.append(f"{modelo}: {codigo or type(exc).__name__}")
                if codigo in CODIGOS_TROCAR_MODELO:
                    logger.warning("Modelo %s falhou com %s; tentando o próximo", modelo, codigo)
                    break
                if codigo in CODIGOS_TEMPORARIOS or codigo is None:  # None = rede/timeout
                    logger.warning("%s tentativa %s: erro temporário %s", modelo, tentativa, codigo)
                    if tentativa < TENTATIVAS_POR_MODELO:
                        espera = ESPERA_INICIAL_SEGUNDOS * (2 ** (tentativa - 1))
                        time.sleep(max(0.0, min(espera, limite - time.monotonic())))
                    continue
                recusou_nivel = codigo == 400 and "thinking" in str(exc).lower()
                if recusou_nivel and _NIVEL_POR_MODELO.get(modelo) != NIVEL_RACIOCINIO_RESERVA:
                    # Nem todo modelo aceita todo nível de raciocínio: tenta de novo com o reserva
                    logger.warning("Modelo %s não aceita raciocínio '%s'; usando '%s'", modelo,
                                   NIVEL_RACIOCINIO, NIVEL_RACIOCINIO_RESERVA)
                    _NIVEL_POR_MODELO[modelo] = NIVEL_RACIOCINIO_RESERVA
                    continue
                anteriores = f" Tentativas anteriores: {' | '.join(falhas[:-1])}." if len(falhas) > 1 else ""
                raise ErroExtracao(f"Falha na chamada ao Gemini ({modelo}): {exc}.{anteriores}") from exc

    raise ErroExtracao("Nenhum modelo respondeu. Tentativas: " + " | ".join(falhas))



# Motor financeiro determinístico

def custo_material(m: dict) -> Decimal:
    """Custo de um material por atendimento: preço ÷ rendimento × quantidade usada."""
    custo = _dec(m.get("valor"))
    if m.get("rendimento"):
        custo = custo / _dec(m["rendimento"])
    if m.get("quantidade_usada"):
        custo = custo * _dec(m["quantidade_usada"])
    return custo


def calcular_metricas_financeiras(
    materiais: list[dict],
    deslocamento: float,
    outros_gastos: float,
    tempo_horas: float,
    valor_hora_desejado: Optional[float],
    preco_cobrado: float,
    taxa_maquininha_pct: float = TAXA_PADRAO,
    custos_fixos_rateados: float = 0,
) -> dict:
    """
    Cobrança por serviço. "materiais" já vem com o custo por atendimento em "valor".

        Custos   = materiais + deslocamento + outros + rateio + taxa (taxa = preço × %)
        Sobra    = preço - custos
        Valor/h  = sobra / tempo_horas
        Meta (opcional): mão de obra = tempo × valor_hora_desejado
        P_min    = (custos sem taxa + mão de obra) / (1 - taxa)    (sem meta: só cobre os custos)

    Também devolve "custo_total" e "sobra_liquida" no conceito antigo (descontando a meta),
    que continuam valendo para quem já usa esses campos.
    """
    taxa = _dec(TAXA_PADRAO if taxa_maquininha_pct is None else taxa_maquininha_pct)
    if not (Decimal("0") <= taxa < Decimal("1")):
        raise ValueError("taxa_maquininha_pct deve estar entre 0 e 1 (ex: 0.05 = 5%).")

    entradas = {
        "deslocamento": _dec(deslocamento),
        "outros_gastos": _dec(outros_gastos),
        "tempo_horas": _dec(tempo_horas),
        "valor_hora_desejado": _dec(valor_hora_desejado),
        "preco_cobrado": _dec(preco_cobrado),
        "custos_fixos_rateados": _dec(custos_fixos_rateados),
    }
    valores_materiais = [_dec(m.get("valor")) for m in (materiais or [])]
    if any(v < 0 for v in list(entradas.values()) + valores_materiais):
        raise ValueError("Valores não podem ser negativos.")

    total_materiais = sum(valores_materiais, Decimal("0"))
    custo_direto = total_materiais + entradas["deslocamento"] + entradas["outros_gastos"]
    custos_sem_taxa = custo_direto + entradas["custos_fixos_rateados"]
    custo_mao_obra = entradas["tempo_horas"] * entradas["valor_hora_desejado"]
    custo_total = custos_sem_taxa + custo_mao_obra

    preco = entradas["preco_cobrado"]
    valor_taxa = preco * taxa
    valor_liquido = preco - valor_taxa
    custos_totais = custos_sem_taxa + valor_taxa
    sobra_real = preco - custos_totais
    horas = entradas["tempo_horas"]
    valor_hora_real = sobra_real / horas if horas > 0 else None
    sobra = valor_liquido - custo_total  # descontando a meta (0 se não houver meta)
    margem = (sobra_real / preco * 100) if preco > 0 else None
    tem_meta = entradas["valor_hora_desejado"] > 0

    return {
        "modelo_cobranca": "por_servico",
        "total_materiais": _arred(total_materiais),
        "deslocamento": _arred(entradas["deslocamento"]),
        "outros_gastos": _arred(entradas["outros_gastos"]),
        "custo_direto": _arred(custo_direto),
        "custos_fixos_rateados": _arred(entradas["custos_fixos_rateados"]),
        "tempo_horas": float(horas),
        "preco_cobrado": _arred(preco),
        "taxa_maquininha_pct": float(taxa),
        "valor_taxa": _arred(valor_taxa),
        "valor_liquido": _arred(valor_liquido),
        "custos_totais": _arred(custos_totais),
        "sobra_real": _arred(sobra_real),
        "valor_hora_real": _arred(valor_hora_real) if valor_hora_real is not None else None,
        "margem_pct": _arred(margem) if margem is not None else None,
        "tem_meta": tem_meta,
        "valor_hora_desejado": _arred(entradas["valor_hora_desejado"]),
        "custo_mao_obra": _arred(custo_mao_obra),
        "custo_total": _arred(custo_total),
        "sobra_liquida": _arred(sobra),
        "preco_minimo": _arred(custo_total / (1 - taxa)),
        "cobre_custos": sobra >= 0,
    }


def _modo(estado: dict) -> str:
    return "recorrente" if estado.get("modelo_cobranca") == "recorrente" else "por_servico"


def _custos_fixos_mensais(custos_fixos: list[dict], encontros_por_mes: Optional[float] = None) -> list[dict]:
    """Converte cada custo para base mensal. Sem período informado = mensal (premissa)."""
    itens = []
    for c in custos_fixos or []:
        if c.get("valor") is None:
            continue
        valor = _dec(c["valor"])
        if valor < 0:
            raise ValueError("Valores não podem ser negativos.")
        periodo = c.get("periodicidade")
        if c.get("por_encontro"):
            if not encontros_por_mes:
                raise ValueError("Custo por encontro sem número de encontros por mês.")
            mensal = valor * _dec(encontros_por_mes)
        else:
            mensal = valor * FATOR_MENSAL[periodo or "mensal"]
        itens.append({
            "item": c["item"],
            "valor": valor,
            "periodicidade": periodo,
            "por_cliente": bool(c.get("por_cliente")),
            "por_encontro": bool(c.get("por_encontro")),
            "estimativa": bool(c.get("estimativa")),
            "valor_mensal": mensal,
        })
    return itens


def _materiais_detalhe(materiais: list[dict]) -> list[dict]:
    detalhe = []
    for m in materiais or []:
        if m.get("valor") is None:
            continue
        detalhe.append({
            "item": m["item"],
            "valor": _arred(_dec(m["valor"])),
            "rendimento": m.get("rendimento"),
            "quantidade_usada": m.get("quantidade_usada"),
            "estimativa": bool(m.get("estimativa")),
            "custo": _arred(custo_material(m)),
        })
    return detalhe


def _metricas_servico(estado: dict, preco: Optional[float] = None) -> dict:
    preco_usado = estado.get("preco_cobrado") if preco is None else preco
    materiais = estado.get("materiais") or []
    fixos = _custos_fixos_mensais([c for c in estado.get("custos_fixos") or [] if not c.get("por_encontro")])
    total_fixos = sum((c["valor_mensal"] for c in fixos), Decimal("0"))
    atendimentos = estado.get("atendimentos_por_mes")
    rateio = total_fixos / _dec(atendimentos) if fixos and atendimentos else Decimal("0")
    metricas = calcular_metricas_financeiras(
        materiais=[{"item": m["item"], "valor": custo_material(m)} for m in materiais if m.get("valor") is not None],
        deslocamento=estado.get("deslocamento"),
        outros_gastos=estado.get("outros_gastos"),
        tempo_horas=estado.get("tempo_horas"),
        valor_hora_desejado=estado.get("valor_hora_desejado"),
        preco_cobrado=preco_usado,
        taxa_maquininha_pct=estado.get("taxa_maquininha_pct"),
        custos_fixos_rateados=rateio,
    )
    metricas.update({
        "materiais_detalhe": _materiais_detalhe(materiais),
        "custos_fixos_mensais": _arred(total_fixos),
        "atendimentos_por_mes": atendimentos,
        "taxa_informada": estado.get("taxa_maquininha_pct") is not None,
        "estimados": list(estado.get("estimados") or []),
    })
    return metricas


def _metricas_do_estado(estado: dict, preco: Optional[float] = None, clientes: Optional[float] = None) -> dict:
    """Atalho: calcula as métricas a partir de um estado de sessão, conforme o modelo de cobrança."""
    if _modo(estado) == "recorrente":
        metricas = calcular_metricas_recorrentes(
            custos_fixos=estado.get("custos_fixos") or [],
            quantidade_clientes=estado.get("quantidade_clientes") if clientes is None else clientes,
            preco_cobrado=estado.get("preco_cobrado") if preco is None else preco,
            periodicidade_preco=estado.get("periodicidade_preco"),
            taxa_maquininha_pct=estado.get("taxa_maquininha_pct"),
            renda_desejada=estado.get("renda_desejada"),
            periodicidade_renda=estado.get("periodicidade_renda"),
            horas_trabalhadas=estado.get("horas_trabalhadas"),
            periodicidade_horas=estado.get("periodicidade_horas"),
            valor_hora_desejado=estado.get("valor_hora_desejado"),
            encontros_por_mes=estado.get("encontros_por_mes"),
            horas_por_encontro=estado.get("horas_por_encontro"),
        )
        metricas["taxa_informada"] = estado.get("taxa_maquininha_pct") is not None
        metricas["estimados"] = list(estado.get("estimados") or [])
        return metricas
    return _metricas_servico(estado, preco)


def calcular_metricas_recorrentes(
    custos_fixos: list[dict],
    quantidade_clientes: float,
    preco_cobrado: Optional[float] = None,
    periodicidade_preco: Optional[str] = None,
    taxa_maquininha_pct: Optional[float] = TAXA_PADRAO,
    renda_desejada: Optional[float] = None,
    periodicidade_renda: Optional[str] = None,
    horas_trabalhadas: Optional[float] = None,
    periodicidade_horas: Optional[str] = None,
    valor_hora_desejado: Optional[float] = None,
    encontros_por_mes: Optional[float] = None,
    horas_por_encontro: Optional[float] = None,
) -> dict:
    """
    Cobrança recorrente (plano mensal, assinatura, aulas...). Tudo em base MENSAL.

        F      = custos fixos do mês ;  rateio = F / clientes
        M      = material por cliente no mês (por cliente, ou por encontro × encontros)
        Por cliente: custos = M + rateio + taxa ;  sobra = preço/mês - custos
        Total:  sobra_real = sobra por cliente × clientes
        Valor/h = sobra por cliente / horas por cliente  (ou sobra total / horas totais)
        Meta (opcional, renda desejada): P_min = (F + M×clientes + renda) / (clientes × (1 - taxa))
        Sem meta: P_min só cobre os custos.
    """
    taxa = _dec(TAXA_PADRAO if taxa_maquininha_pct is None else taxa_maquininha_pct)
    if not (Decimal("0") <= taxa < Decimal("1")):
        raise ValueError("taxa_maquininha_pct deve estar entre 0 e 1 (ex: 0.05 = 5%).")
    clientes = _dec(quantidade_clientes)
    if clientes <= 0:
        raise ValueError("quantidade_clientes deve ser maior que zero.")

    premissas: list[str] = []
    itens = _custos_fixos_mensais(custos_fixos, encontros_por_mes)
    for c in itens:
        if c["valor"] == 0 or c["por_encontro"]:
            continue
        if c["periodicidade"] is None:
            premissas.append(f"\"{c['item']}\" como gasto mensal")
        elif c["periodicidade"] == "unico":
            premissas.append(f"o gasto único com \"{c['item']}\" dividido em {MESES_AMORTIZACAO} meses")
    fixos = sum((c["valor_mensal"] for c in itens if not c["por_cliente"] and not c["por_encontro"]), Decimal("0"))
    material_cliente = sum((c["valor_mensal"] for c in itens if c["por_cliente"] or c["por_encontro"]), Decimal("0"))
    rateio = fixos / clientes

    horas_cliente = None
    if encontros_por_mes and horas_por_encontro:
        horas_cliente = _dec(encontros_por_mes) * _dec(horas_por_encontro)
    horas_total = None
    if horas_trabalhadas is not None:
        periodo_h = periodicidade_horas or "mensal"
        if periodicidade_horas is None:
            premissas.append("as horas informadas como horas por mês")
        elif periodo_h == "diario":
            premissas.append(f"{DIAS_TRABALHO_MES} dias de trabalho por mês")
        horas_total = _dec(horas_trabalhadas) * FATOR_TRABALHO[periodo_h]
    elif horas_cliente is not None:
        horas_total = horas_cliente * clientes

    renda = None
    origem_renda = None
    if renda_desejada is not None and renda_desejada > 0:
        if periodicidade_renda is None:
            premissas.append("a renda desejada como mensal")
        renda = _dec(renda_desejada) * FATOR_TRABALHO[periodicidade_renda or "mensal"]
        origem_renda = "renda_desejada"
    elif horas_total is not None and valor_hora_desejado:
        renda = horas_total * _dec(valor_hora_desejado)
        origem_renda = "horas_x_valor_hora"

    custo_clientes = material_cliente * clientes
    custo_total = fixos + custo_clientes + (renda or Decimal("0"))
    preco_minimo_mensal = custo_total / (clientes * (1 - taxa))
    fator_preco = FATOR_MENSAL[periodicidade_preco or "mensal"]
    if preco_cobrado is not None and periodicidade_preco is None:
        premissas.append("o preço informado como mensal")

    resultado = {
        "modelo_cobranca": "recorrente",
        "custos_fixos": [
            {
                "item": c["item"], "valor": _arred(c["valor"]), "periodicidade": c["periodicidade"],
                "por_cliente": c["por_cliente"], "por_encontro": c["por_encontro"],
                "estimativa": c["estimativa"], "valor_mensal": _arred(c["valor_mensal"]),
            }
            for c in itens
        ],
        "custos_fixos_mensais": _arred(fixos),
        "rateio_por_cliente": _arred(rateio),
        "custo_por_cliente_mensal": _arred(material_cliente),
        "custo_clientes_mensal": _arred(custo_clientes),
        "encontros_por_mes": encontros_por_mes,
        "horas_por_encontro": horas_por_encontro,
        "horas_por_cliente": _arred(horas_cliente) if horas_cliente is not None else None,
        "horas_mensais": _arred(horas_total) if horas_total is not None else None,
        "tem_meta": renda is not None,
        "renda_desejada_mensal": _arred(renda) if renda is not None else None,
        "origem_renda": origem_renda,
        "valor_hora_implicito": _arred(renda / horas_total) if renda is not None and horas_total else None,
        "custo_total": _arred(custo_total),
        "quantidade_clientes": float(clientes),
        "taxa_maquininha_pct": float(taxa),
        "periodicidade_preco": periodicidade_preco or "mensal",
        "preco_minimo": _arred(preco_minimo_mensal),
        "preco_minimo_periodo": _arred(preco_minimo_mensal / fator_preco),
        "preco_cobrado": None, "preco_mensal_por_cliente": None, "receita_bruta": None, "valor_taxa": None,
        "valor_liquido": None, "taxa_por_cliente": None, "custos_por_cliente": None, "sobra_por_cliente": None,
        "sobra_real": None, "valor_hora_real": None, "sobra_liquida": None, "margem_pct": None,
        "falta_para_meta": None, "clientes_necessarios": None, "cobre_custos": None,
        "premissas": premissas,
    }

    if preco_cobrado is not None:
        preco = _dec(preco_cobrado)
        if preco < 0:
            raise ValueError("Valores não podem ser negativos.")
        preco_mensal = preco * fator_preco
        receita = preco_mensal * clientes
        valor_taxa = receita * taxa
        liquido = receita - valor_taxa
        taxa_cliente = preco_mensal * taxa
        custos_cliente = material_cliente + rateio + taxa_cliente
        sobra_cliente = preco_mensal - custos_cliente
        sobra_real = sobra_cliente * clientes
        if horas_cliente:
            valor_hora_real = sobra_cliente / horas_cliente
        elif horas_total:
            valor_hora_real = sobra_real / horas_total
        else:
            valor_hora_real = None
        sobra = liquido - custo_total
        contribuicao = preco_mensal * (1 - taxa) - material_cliente
        necessarios = (
            int(math.ceil((fixos + (renda or Decimal("0"))) / contribuicao)) if contribuicao > 0 else None
        )
        resultado.update({
            "preco_cobrado": _arred(preco),
            "preco_mensal_por_cliente": _arred(preco_mensal),
            "receita_bruta": _arred(receita),
            "valor_taxa": _arred(valor_taxa),
            "valor_liquido": _arred(liquido),
            "taxa_por_cliente": _arred(taxa_cliente),
            "custos_por_cliente": _arred(custos_cliente),
            "sobra_por_cliente": _arred(sobra_cliente),
            "sobra_real": _arred(sobra_real),
            "valor_hora_real": _arred(valor_hora_real) if valor_hora_real is not None else None,
            "sobra_liquida": _arred(sobra),
            "margem_pct": _arred(sobra_real / receita * 100) if receita > 0 else None,
            "falta_para_meta": _arred(renda - sobra_real) if renda is not None else None,
            "clientes_necessarios": necessarios,
            "cobre_custos": sobra >= 0,
        })
    return resultado


def simular_novo_preco(
    estado_atual: dict, novo_preco: Optional[float], novos_clientes: Optional[float] = None
) -> dict:
    """
    Recalcula as métricas mantendo os custos do estado atual, mudando o preço para 'novo_preco'
    (a taxa percentual é recalculada sobre o novo preço).
    No modelo recorrente também aceita 'novos_clientes' ("e se eu tiver 20 assinantes?").
    "sobra_*" e "diferenca_sobra" seguem o conceito antigo (descontando a meta);
    "sobra_real_*" e "diferenca_sobra_real" são preço - custos.
    """
    recorrente = _modo(estado_atual) == "recorrente"
    if novo_preco is not None and novo_preco <= 0:
        raise ValueError("novo_preco deve ser maior que zero.")
    if novo_preco is None and not (recorrente and novos_clientes):
        raise ValueError("novo_preco deve ser maior que zero.")

    simuladas = _metricas_do_estado(estado_atual, preco=novo_preco, clientes=novos_clientes)
    original = None
    if estado_atual.get("preco_cobrado") is not None or recorrente:
        original = _metricas_do_estado(estado_atual)

    def dif(campo: str) -> Optional[float]:
        if original and original.get(campo) is not None and simuladas.get(campo) is not None:
            return _arred(_dec(simuladas[campo]) - _dec(original[campo]))
        return None

    return {
        "modelo_cobranca": "recorrente" if recorrente else "por_servico",
        "preco_original": original["preco_cobrado"] if original else None,
        "preco_simulado": simuladas["preco_cobrado"],
        "clientes_original": original.get("quantidade_clientes") if original else None,
        "clientes_simulado": simuladas.get("quantidade_clientes"),
        "sobra_original": original["sobra_liquida"] if original else None,
        "sobra_simulada": simuladas["sobra_liquida"],
        "diferenca_sobra": dif("sobra_liquida"),
        "sobra_real_original": original["sobra_real"] if original else None,
        "sobra_real_simulada": simuladas["sobra_real"],
        "diferenca_sobra_real": dif("sobra_real"),
        "metricas_original": original,
        "metricas_simuladas": simuladas,
    }


def _encontrar_material(materiais: list[dict], nome: str, aproximado: bool) -> Optional[int]:
    """Índice do item com o mesmo nome (ou nome contido, se aproximado)."""
    alvo = _chave(nome)
    for i, m in enumerate(materiais):
        if _chave(m["item"]) == alvo:
            return i
    if aproximado and alvo:
        for i, m in enumerate(materiais):
            existente = _chave(m["item"])
            if existente and (alvo in existente or existente in alvo):
                return i
    return None


def _mesclar_itens(lista: list[dict], novos: list[dict], aproximado: bool, campos: tuple) -> list[dict]:
    """Substitui itens de mesmo nome (só os atributos informados) e adiciona os novos."""
    lista = [dict(x) for x in lista]
    for novo in novos:
        indice = _encontrar_material(lista, novo["item"], aproximado)
        if indice is None:
            if novo.get("valor") is not None or novo.get("embalagem"):
                lista.append(dict(novo))
            continue
        atual = lista[indice]
        for campo in campos:
            if novo.get(campo) is not None:
                atual[campo] = novo[campo]
        for flag in ("embalagem", "por_cliente", "por_encontro"):
            if flag in novo and novo[flag]:
                atual[flag] = True
        if "estimativa" in novo and any(novo.get(c) is not None for c in campos):
            atual["estimativa"] = bool(novo.get("estimativa"))
    # Ao informar um gasto real do mesmo tipo, o marcador "Nenhum" deixa de valer
    def tipo(x):
        return (bool(x.get("por_cliente")), bool(x.get("por_encontro")))

    def eh_nenhum(x):
        return _chave(x["item"]) == "nenhum" and not x.get("valor")

    reais = {tipo(x) for x in lista if not eh_nenhum(x)}
    return [x for x in lista if not (eh_nenhum(x) and tipo(x) in reais)]


def atualizar_estado(estado_anterior: dict, dados_novos: dict) -> dict:
    """
    Mescla o estado anterior da sessão com os novos dados extraídos da IA.
    Suporta atualização de itens específicos quando intencao == 'corrigir_dado'.

    Regras:
      - Nunca altera o dicionário recebido (trabalha sobre uma cópia).
      - Campos None nos dados novos NÃO apagam o que já existe.
      - Materiais e custos com o mesmo nome são atualizados (só o que veio); novos são adicionados.
        Em 'corrigir_dado' a busca por nome é aproximada ("base" encontra "Base 30 ml").
      - Em 'simular_preco' o preço original da sessão é preservado.
      - Campos marcados como estimativa ficam em "estimados".
    """
    estado = copy.deepcopy(ESTADO_VAZIO)
    estado.update(copy.deepcopy(estado_anterior or {}))
    estado["materiais"] = list(estado.get("materiais") or [])
    estado["custos_fixos"] = list(estado.get("custos_fixos") or [])
    estimados = set(estado.get("estimados") or [])
    intencao = dados_novos.get("intencao")

    # Modelo de cobrança: só passa a "recorrente"; "por_servico" só define se ainda não houver
    modelo = dados_novos.get("modelo_cobranca")
    if modelo == "recorrente" or (modelo == "por_servico" and estado.get("modelo_cobranca") is None):
        estado["modelo_cobranca"] = modelo

    marcados = set(dados_novos.get("campos_estimados") or [])
    for campo in CAMPOS_CONTEXTO:
        if campo in ("materiais", "custos_fixos", "modelo_cobranca"):
            continue
        valor = dados_novos.get(campo)
        if valor is None:
            continue
        if intencao == "simular_preco" and campo == "preco_cobrado" and estado.get("preco_cobrado") is not None:
            continue  # o preço "de referência" da simulação não sobrescreve a sessão
        estado[campo] = valor
        if campo in marcados:
            estimados.add(campo)
        else:
            estimados.discard(campo)
    estado["estimados"] = sorted(estimados)

    aproximado = intencao == "corrigir_dado"
    estado["materiais"] = _mesclar_itens(
        estado["materiais"], dados_novos.get("materiais") or [], aproximado,
        ("valor", "rendimento", "quantidade_usada"),
    )
    estado["custos_fixos"] = _mesclar_itens(
        estado["custos_fixos"], dados_novos.get("custos_fixos") or [], aproximado, ("valor", "periodicidade"),
    )
    return estado


def _custos_mensais_respondidos(estado: dict) -> bool:
    return any(not c.get("por_cliente") and not c.get("por_encontro") for c in estado.get("custos_fixos") or [])


def campos_faltantes(estado: dict) -> list[str]:
    """Dados essenciais ainda não informados: sem eles o assistente não calcula."""
    faltantes = []
    if _modo(estado) == "recorrente":
        if not estado.get("quantidade_clientes"):
            faltantes.append("quantidade_clientes")
        tem_meta = estado.get("renda_desejada") is not None or (
            estado.get("horas_trabalhadas") is not None and estado.get("valor_hora_desejado") is not None
        )
        if estado.get("preco_cobrado") is None and not tem_meta:
            faltantes.append("preco_recorrente")
        if not _custos_mensais_respondidos(estado):
            faltantes.append("custos_fixos")
        if any(c.get("por_encontro") for c in estado.get("custos_fixos") or []) and not estado.get("encontros_por_mes"):
            faltantes.append("encontros_por_mes")
        return faltantes
    if estado.get("preco_cobrado") is None:
        faltantes.append("preco_cobrado")
    if estado.get("tempo_horas") is None:
        faltantes.append("tempo_horas")
    if not estado.get("materiais"):
        faltantes.append("materiais")
    if estado.get("deslocamento") is None:
        faltantes.append("deslocamento")
    for m in estado.get("materiais") or []:
        if m.get("embalagem") and not m.get("rendimento"):
            faltantes.append(f"rendimento:{m['item']}")
    return faltantes


def campos_opcionais(estado: dict) -> list[str]:
    """Dados que o assistente pergunta, mas não bloqueiam o cálculo (viram premissa avisada)."""
    opcionais = []
    if estado.get("taxa_maquininha_pct") is None:
        opcionais.append("taxa_maquininha_pct")
    if _modo(estado) == "recorrente" and not any(
        c.get("por_cliente") or c.get("por_encontro") for c in estado.get("custos_fixos") or []
    ):
        opcionais.append("material_por_cliente")
    return opcionais



# Respostas em linguagem simples

def formatar_moeda(valor: float) -> str:
    """1234.5 -> 'R$ 1.234,50' | -5 -> '-R$ 5,00'"""
    texto = f"{abs(valor):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{'-' if valor < 0 else ''}R$ {texto}"


def _fmt_num(valor: float) -> str:
    """1.5 -> '1,5' | 2.0 -> '2' | 1.25 -> '1,25'"""
    return f"{float(valor):.2f}".rstrip("0").rstrip(".").replace(".", ",")


def _fmt_pct(valor: Optional[float]) -> str:
    return "—" if valor is None else f"{valor:.1f}%".replace(".", ",")


def _preco_periodo(valor: float, periodo: str) -> str:
    return f"{formatar_moeda(valor)} {NOME_PERIODO.get(periodo, 'por mês')}"


def _est(m: dict, campo: str) -> str:
    return " (estimativa sua)" if campo in (m.get("estimados") or []) else ""


def _texto_material(d: dict, rotulo: Optional[str] = None) -> str:
    detalhes = []
    if d.get("rendimento"):
        conta = f"{formatar_moeda(d['valor'])} ÷ {_fmt_num(d['rendimento'])}"
        if d.get("quantidade_usada") and d["quantidade_usada"] != 1:
            conta += f" × {_fmt_num(d['quantidade_usada'])}"
        detalhes.append(conta)
    elif d.get("quantidade_usada") and d["quantidade_usada"] != 1:
        detalhes.append(f"{formatar_moeda(d['valor'])} × {_fmt_num(d['quantidade_usada'])}")
    if d.get("estimativa"):
        detalhes.append("estimativa sua")
    texto = f"{rotulo or d['item']}: {formatar_moeda(d['custo'])}"
    return texto + (f" ({', '.join(detalhes)})" if detalhes else "")


def _linhas_materiais(m: dict) -> list[str]:
    detalhe = [d for d in m.get("materiais_detalhe") or [] if not (_chave(d["item"]) == "nenhum" and not d["custo"])]
    if not detalhe:
        if m.get("materiais_detalhe"):
            return ["- Materiais: sem custo"]
        return [f"- Materiais: {formatar_moeda(m['total_materiais'])}"]
    if len(detalhe) == 1:
        return ["- " + _texto_material(detalhe[0], "Materiais")]
    linhas = ["- " + _texto_material(d) for d in detalhe]
    total = f"- **Total de materiais:** {formatar_moeda(m['total_materiais'])}"
    if any(d["estimativa"] for d in detalhe):
        total += " (inclui estimativas suas)"
    return linhas + [total]


def _texto_taxa(m: dict, preco: Optional[float] = None) -> str:
    if m["taxa_maquininha_pct"] == 0:
        return "- Taxa de pagamento: sem taxa"
    base = f" de {formatar_moeda(preco)}" if preco is not None else ""
    return (f"- Taxa do cartão ({_fmt_num(m['taxa_maquininha_pct'] * 100)}%{base}): "
            f"{formatar_moeda(m['valor_taxa'] if preco is None else m.get('taxa_por_cliente') or m['valor_taxa'])}")


def _aviso_taxa_padrao(m: dict) -> list[str]:
    if m.get("taxa_informada", True):
        return []
    return ["", f"ℹ️ Considerei {_fmt_num(TAXA_PADRAO * 100)}% de taxa de cartão porque a forma de pagamento "
                "não foi informada. Se for Pix ou dinheiro, me avise que eu recalculo."]


def _preco_sugerido_para_teste(m: dict) -> int:
    """Sugestão de valor para o usuário simular (15% acima, múltiplo de 5)."""
    base = max(m["preco_cobrado"], m["preco_minimo"]) * 1.15
    return int(math.ceil(base / 5) * 5)


def _texto_resumo(m: dict) -> str:
    nome = m.get("nome_servico")
    linhas = [f"Resumo do seu serviço (**{nome}**):" if nome else "Resumo do seu serviço:"]
    linhas += _linhas_materiais(m)
    linhas.append(f"- Deslocamento: {formatar_moeda(m['deslocamento'])}{_est(m, 'deslocamento')}"
                  if m["deslocamento"] else "- Deslocamento: sem custo")
    if m.get("outros_gastos"):
        linhas.append(f"- Outros gastos: {formatar_moeda(m['outros_gastos'])}{_est(m, 'outros_gastos')}")
    if m.get("custos_fixos_rateados"):
        linhas.append(
            f"- Custos mensais rateados ({formatar_moeda(m['custos_fixos_mensais'])}/mês ÷ "
            f"{_fmt_num(m['atendimentos_por_mes'])} atendimentos{_est(m, 'atendimentos_por_mes')}): "
            f"{formatar_moeda(m['custos_fixos_rateados'])}"
        )
    linhas.append(_texto_taxa(m, m["preco_cobrado"]) if m["taxa_maquininha_pct"] else "- Taxa de pagamento: sem taxa")
    linhas += [f"- **Total de custos:** {formatar_moeda(m['custos_totais'])}", ""]

    linhas.append(f"Cobrando **{formatar_moeda(m['preco_cobrado'])}**:")
    linhas.append(f"- **Sobra para você:** {formatar_moeda(m['sobra_real'])}")
    if m.get("valor_hora_real") is not None:
        linhas.append(
            f"- **Valor por hora:** {formatar_moeda(m['valor_hora_real'])} "
            f"({_fmt_num(m['tempo_horas'])}h de trabalho, contando preparo e limpeza{_est(m, 'tempo_horas')})"
        )
    if m["sobra_real"] < 0:
        linhas.append(f"- ⚠️ Nesse preço você tem prejuízo: os custos passam do preço em "
                      f"{formatar_moeda(abs(m['sobra_real']))}.")
    linhas.append("")

    if m.get("tem_meta"):
        meta = f"R$ {_fmt_num(m['valor_hora_desejado'])}/h"
        if m["cobre_custos"]:
            linhas.append(
                f"✅ Acima da sua meta de {meta}: sobram {formatar_moeda(m['sobra_liquida'])} além da hora desejada. "
                f"O preço mínimo para essa meta é {formatar_moeda(m['preco_minimo'])}."
            )
        else:
            por_hora = (abs(m["sobra_liquida"]) / m["tempo_horas"]) if m["tempo_horas"] else 0
            linhas.append(
                f"🎯 Faltam {formatar_moeda(abs(m['sobra_liquida']))} neste serviço "
                f"({formatar_moeda(por_hora)} por hora) para chegar na sua meta de {meta}. "
                f"Para alcançá-la, cobre pelo menos **{formatar_moeda(m['preco_minimo'])}**."
            )
    else:
        linhas.append(
            f"💡 **Preço mínimo:** {formatar_moeda(m['preco_minimo'])}. Abaixo disso você não cobre nem os custos. "
            "Se quiser, me diga quanto quer ganhar por hora que eu calculo o preço para essa meta."
        )
    if m.get("custos_fixos_mensais") and not m.get("custos_fixos_rateados"):
        linhas += ["", (
            f"ℹ️ Anotei seus custos fixos de {formatar_moeda(m['custos_fixos_mensais'])} por mês, mas eles ainda "
            "não entraram na conta. Me diga quantos atendimentos você faz por mês para incluí-los."
        )]
    linhas += _aviso_taxa_padrao(m)
    linhas += ["", f"Quer testar outro valor? Pergunte: 'E se eu cobrar R$ {_preco_sugerido_para_teste(m)}?'"]
    return "\n".join(linhas)


def _variacao(antes: float, depois: float) -> Optional[float]:
    return (depois - antes) / abs(antes) * 100 if antes else None


def _texto_simulacao(s: dict) -> str:
    sim = s["metricas_simuladas"]
    orig = s["metricas_original"]
    if orig is None:
        linhas = [
            f"🔎 Simulação cobrando **{formatar_moeda(s['preco_simulado'])}**:",
            f"- Total de custos: {formatar_moeda(sim['custos_totais'])}",
            f"- Sobra: **{formatar_moeda(sim['sobra_real'])}**",
        ]
        if sim.get("valor_hora_real") is not None:
            linhas.append(f"- Valor por hora: **{formatar_moeda(sim['valor_hora_real'])}**")
        return "\n".join(linhas)

    def seta(campo: str) -> str:
        return f"{formatar_moeda(orig[campo])} → **{formatar_moeda(sim[campo])}**"

    linhas = [f"🔎 Comparando **{formatar_moeda(s['preco_original'])}** com **{formatar_moeda(s['preco_simulado'])}**:"]
    if sim["taxa_maquininha_pct"]:
        linhas.append(f"- Taxa do cartão ({_fmt_num(sim['taxa_maquininha_pct'] * 100)}%): {seta('valor_taxa')}")
    linhas.append(f"- Total de custos: {seta('custos_totais')}")
    linhas.append(f"- Sobra para você: {seta('sobra_real')}")
    if sim.get("valor_hora_real") is not None and orig.get("valor_hora_real") is not None:
        linhas.append(f"- Valor por hora: {seta('valor_hora_real')}")
    linhas.append("")

    var_preco = _variacao(orig["preco_cobrado"], sim["preco_cobrado"])
    var_sobra = _variacao(orig["sobra_real"], sim["sobra_real"])
    if var_preco and var_sobra is not None:
        if var_preco < 0:
            frase = f"O preço cai {_fmt_pct(abs(var_preco))}"
            if var_sobra < 0 and abs(var_sobra) > abs(var_preco):
                frase += (f", mas a sobra cai {_fmt_pct(abs(var_sobra))}: materiais e deslocamento continuam "
                          "custando o mesmo, então quase toda a redução sai do que sobra para você.")
            else:
                frase += f" e a sobra cai {_fmt_pct(abs(var_sobra))}."
        else:
            frase = f"O preço sobe {_fmt_pct(var_preco)}"
            if var_sobra > var_preco:
                frase += (f" e a sobra sobe {_fmt_pct(var_sobra)}: como os custos do serviço não mudam, "
                          "quase todo o aumento vira sobra (só a taxa do cartão acompanha o preço).")
            else:
                frase += f" e a sobra muda {_fmt_pct(var_sobra)}."
        linhas.append(frase)

    if sim.get("tem_meta"):
        meta = f"R$ {_fmt_num(sim['valor_hora_desejado'])}/h"
        if sim["cobre_custos"]:
            linhas.append(f"✅ Com {formatar_moeda(s['preco_simulado'])} você passa da sua meta de {meta}.")
        else:
            linhas.append(f"🎯 Ainda abaixo da sua meta de {meta}: o mínimo para ela é "
                          f"{formatar_moeda(sim['preco_minimo'])}.")
    elif sim["sobra_real"] < 0:
        linhas.append(f"⚠️ Esse valor não cobre os custos. O mínimo é {formatar_moeda(sim['preco_minimo'])}.")
    return "\n".join(linhas)


def _descricao_custo(c: dict) -> str:
    generico = _chave(c["item"]) in ("material", "materiais")
    texto = formatar_moeda(c["valor"]) if generico else f"{c['item']} {formatar_moeda(c['valor'])}"
    if c.get("por_encontro"):
        texto += " por encontro"
    elif c.get("por_cliente"):
        texto += " por cliente"
    if c.get("periodicidade") not in (None, "mensal") and not c.get("por_encontro"):
        texto += f" {NOME_PERIODO[c['periodicidade']]}"
    if c.get("estimativa"):
        texto += " (estimativa sua)"
    return texto


def _texto_resumo_recorrente(m: dict) -> str:
    nome = m.get("nome_servico")
    periodo = m["periodicidade_preco"]
    clientes = _fmt_num(m["quantidade_clientes"])
    est_clientes = _est(m, "quantidade_clientes")
    fixos = [c for c in m["custos_fixos"] if not c["por_cliente"] and not c["por_encontro"] and c["valor"]]
    linhas = [f"Resumo do seu plano (**{nome}**):" if nome else "Resumo do seu plano recorrente:"]

    if m["preco_cobrado"] is not None:
        linhas += ["", "Por cliente, por mês:", f"- Preço: {_preco_periodo(m['preco_cobrado'], periodo)}"]
        if periodo != "mensal":
            linhas[-1] += f" ({formatar_moeda(m['preco_mensal_por_cliente'])} por mês)"
        if m["custo_por_cliente_mensal"]:
            encontro = [c for c in m["custos_fixos"] if c["por_encontro"]]
            conta = ""
            if encontro and m.get("encontros_por_mes") and len(encontro) == 1:
                conta = f" ({_fmt_num(m['encontros_por_mes'])} encontros × {formatar_moeda(encontro[0]['valor'])})"
            linhas.append(f"- Materiais: {formatar_moeda(m['custo_por_cliente_mensal'])}{conta}")
        if m["custos_fixos_mensais"]:
            marca = ", estimativa sua" if est_clientes or any(c["estimativa"] for c in fixos) else ""
            linhas.append(
                f"- Custos mensais rateados: {formatar_moeda(m['rateio_por_cliente'])} "
                f"({formatar_moeda(m['custos_fixos_mensais'])} ÷ {clientes} clientes{marca})"
            )
        linhas.append(
            f"- Taxa do cartão ({_fmt_num(m['taxa_maquininha_pct'] * 100)}%): {formatar_moeda(m['taxa_por_cliente'])}"
            if m["taxa_maquininha_pct"] else "- Taxa de pagamento: sem taxa"
        )
        linhas += [
            f"- **Total de custos por cliente:** {formatar_moeda(m['custos_por_cliente'])}",
            f"- **Sobra por cliente:** {formatar_moeda(m['sobra_por_cliente'])}",
        ]
        if m.get("horas_por_cliente") and m.get("valor_hora_real") is not None:
            linhas.append(
                f"- **Valor por hora:** {formatar_moeda(m['valor_hora_real'])} "
                f"({_fmt_num(m['horas_por_cliente'])}h por cliente no mês: {_fmt_num(m['encontros_por_mes'])} "
                f"encontros de {_fmt_num(m['horas_por_encontro'])}h)"
            )
        total = f"No total, com {clientes} clientes{est_clientes}: **{formatar_moeda(m['sobra_real'])} de sobra por mês**"
        if not m.get("horas_por_cliente") and m.get("valor_hora_real") is not None:
            total += (f" ({formatar_moeda(m['valor_hora_real'])} por hora em "
                      f"{_fmt_num(m['horas_mensais'])}h de trabalho)")
        linhas += ["", total + "."]
        if m["sobra_real"] < 0:
            linhas.append("⚠️ Nesse preço os custos passam da receita: você tem prejuízo todo mês.")
    else:
        linhas += ["", f"Custos do mês com {clientes} clientes{est_clientes}: "
                       f"**{formatar_moeda(m['custos_fixos_mensais'] + m['custo_clientes_mensal'])}**"]

    if fixos:
        linhas += ["", "Custos mensais considerados: " + "; ".join(
            f"{c['item']} {formatar_moeda(c['valor_mensal'])}"
            + (f" ({formatar_moeda(c['valor'])} {NOME_PERIODO[c['periodicidade']]})"
               if c["periodicidade"] not in (None, "mensal") else "")
            for c in fixos
        ) + "."]

    linhas.append("")
    if m.get("tem_meta"):
        meta = f"🎯 Sua meta é ganhar {formatar_moeda(m['renda_desejada_mensal'])} por mês"
        if m.get("horas_mensais") and m.get("valor_hora_implicito") is not None:
            meta += (f" (≈ {_fmt_num(m['horas_mensais'])}h de trabalho, "
                     f"{formatar_moeda(m['valor_hora_implicito'])} por hora)")
        linhas.append(meta + ".")
        if m["preco_cobrado"] is not None:
            if m["cobre_custos"]:
                linhas.append(f"- ✅ Você passa da meta em {formatar_moeda(m['sobra_liquida'])} por mês.")
            else:
                linhas.append(f"- ⚠️ **Faltam {formatar_moeda(abs(m['sobra_liquida']))} por mês** para chegar lá.")
        recomendado = (f"💡 **Preço mínimo recomendado:** {_preco_periodo(m['preco_minimo_periodo'], periodo)} "
                       f"por cliente")
        if periodo != "mensal":
            recomendado += f" (equivale a {formatar_moeda(m['preco_minimo'])} por mês)"
        linhas.append(recomendado + f", com {clientes} clientes.")
        if m["clientes_necessarios"] is not None and m["cobre_custos"] is False:
            linhas.append(
                f"Ou, mantendo {_preco_periodo(m['preco_cobrado'], periodo)}, você precisaria de "
                f"**{m['clientes_necessarios']} clientes**."
            )
    else:
        linhas.append(
            f"💡 **Preço mínimo:** {_preco_periodo(m['preco_minimo_periodo'], periodo)} por cliente, com "
            f"{clientes} clientes. Abaixo disso você não cobre os custos. Se quiser, me diga quanto quer ganhar "
            "por mês que eu calculo o preço para essa meta."
        )
    if m["premissas"]:
        linhas += ["", "ℹ️ Para essa conta, considerei: " + "; ".join(m["premissas"])
                   + ". Se algo estiver diferente, é só me corrigir."]
    linhas += _aviso_taxa_padrao(m)
    sugestao = int(math.ceil(max(m["preco_minimo_periodo"], m["preco_cobrado"] or 0) * 1.1 / 10) * 10)
    mais_clientes = int(math.ceil(m["quantidade_clientes"] * 1.5))
    linhas += ["", f"Quer testar? Pergunte: 'E se eu cobrar R$ {sugestao}?' ou 'E se eu tiver {mais_clientes} clientes?'"]
    return "\n".join(linhas)


def _texto_simulacao_recorrente(s: dict) -> str:
    sim, orig = s["metricas_simuladas"], s["metricas_original"]
    periodo = sim["periodicidade_preco"]
    clientes_sim = _fmt_num(sim["quantidade_clientes"])
    linhas = []
    if sim["preco_cobrado"] is not None:
        cabecalho = f"🔎 Simulação: {clientes_sim} clientes × **{_preco_periodo(sim['preco_cobrado'], periodo)}**"
        tem_orig = orig and orig["preco_cobrado"] is not None
        if tem_orig:
            cabecalho += (f" (hoje: {_fmt_num(orig['quantidade_clientes'])} × "
                          f"{_preco_periodo(orig['preco_cobrado'], periodo)})")
        linhas.append(cabecalho)
        if tem_orig:
            linhas.append(f"- Sobra por cliente: {formatar_moeda(orig['sobra_por_cliente'])} → "
                          f"**{formatar_moeda(sim['sobra_por_cliente'])}**")
            dif = s["diferenca_sobra_real"]
            total = (f"- Sobra total no mês: {formatar_moeda(orig['sobra_real'])} → "
                     f"**{formatar_moeda(sim['sobra_real'])}**")
            if dif:
                total += f" ({formatar_moeda(abs(dif))} {'a mais' if dif > 0 else 'a menos'} por mês)"
            linhas.append(total)
        else:
            linhas.append(f"- Sobra total no mês: **{formatar_moeda(sim['sobra_real'])}**")
        if sim.get("tem_meta"):
            if sim["cobre_custos"]:
                linhas.append("- ✅ Com isso você atinge a sua meta.")
            else:
                linhas.append(f"- Ainda faltam {formatar_moeda(abs(sim['sobra_liquida']))} por mês para a sua meta.")
        elif sim["sobra_real"] < 0:
            linhas.append("- ⚠️ Ainda não cobre os custos do mês.")
    else:
        linhas.append(f"🔎 Simulação com **{clientes_sim} clientes**:")
    linhas.append("")
    minimo = f"💡 Preço mínimo com {clientes_sim} clientes: **{_preco_periodo(sim['preco_minimo_periodo'], periodo)}**"
    if orig and orig["quantidade_clientes"] != sim["quantidade_clientes"]:
        minimo += (f" (com {_fmt_num(orig['quantidade_clientes'])}: "
                   f"{_preco_periodo(orig['preco_minimo_periodo'], periodo)})")
    linhas.append(minimo + ".")
    return "\n".join(linhas)


def formatar_explicacao_humana(metricas: dict, intencao: str) -> str:
    """
    Converte os números calculados em texto simples.
      - informar_dados / corrigir_dado / confirmar: recebe o dict de métricas
        (opcionalmente com "nome_servico").
      - simular_preco: recebe o dict de simular_novo_preco.
    """
    recorrente = metricas.get("modelo_cobranca") == "recorrente"
    if intencao == "simular_preco":
        return _texto_simulacao_recorrente(metricas) if recorrente else _texto_simulacao(metricas)
    resumo = _texto_resumo_recorrente(metricas) if recorrente else _texto_resumo(metricas)
    if intencao == "corrigir_dado":
        return "✅ Pronto, dado corrigido! Veja como ficou:\n\n" + resumo
    return resumo


def _pergunta(campo: str) -> str:
    if campo.startswith("rendimento:"):
        item = campo.split(":", 1)[1]
        return f"quanto rende \"{item}\" (quantos atendimentos ou unidades vêm na embalagem)"
    return PERGUNTAS[campo]


def _texto_faltantes(faltantes: list[str], opcionais: list[str]) -> str:
    linhas = ["Para calcular, ainda preciso saber:"] + [f"- {_pergunta(c)}" for c in faltantes]
    if opcionais:
        linhas += ["", "Se puder, me conte também:"] + [f"- {_pergunta(c)}" for c in opcionais]
    return "\n".join(linhas)


def _texto_invalidos(invalidos: list[dict]) -> str:
    linhas = []
    for inv in invalidos:
        campo, valor = inv["campo"], inv["valor"]
        prefixo, _, item = campo.partition(":")
        if item:
            rotulo = {"material": f"o valor de \"{item}\"", "custo": f"o custo \"{item}\"",
                      "rendimento": f"o rendimento de \"{item}\"",
                      "quantidade": f"a quantidade usada de \"{item}\""}.get(prefixo, campo)
        else:
            rotulo = f"o {ROTULOS.get(campo, campo)}" if campo not in ("taxa_maquininha_pct",) else "a taxa de pagamento"
        dinheiro = prefixo in ("material", "custo") or campo in (
            "preco_cobrado", "novo_preco_simulado", "deslocamento", "outros_gastos", "valor_hora_desejado",
            "renda_desejada",
        )
        valor_txt = formatar_moeda(valor) if dinheiro else _fmt_num(valor)
        if inv["motivo"] == "negativo":
            linhas.append(f"⚠️ Recebi {rotulo} negativo ({valor_txt}). Foi erro de digitação? "
                          "Me mande o valor certo (se não tiver esse gasto, pode dizer 'não tenho').")
        elif campo == "tempo_horas":
            linhas.append("⚠️ O tempo não pode ser 0 horas: preciso do tempo real de cada atendimento "
                          "(contando preparo e limpeza) para calcular quanto você ganha por hora.")
        elif campo in ("preco_cobrado", "novo_preco_simulado"):
            linhas.append("⚠️ Você informou preço de R$ 0,00. Foi erro de digitação ou esse serviço é gratuito? "
                          "Para calcular, preciso do valor que você cobra.")
        else:
            linhas.append(f"⚠️ {rotulo[0].upper() + rotulo[1:]} não pode ser 0. Pode conferir?")
    return "\n".join(linhas)


def _texto_nao_sabe(campos: list[str]) -> str:
    if any("materia" in _chave(c) for c in campos):
        return ("Sem problema, vamos estimar juntos! Pense nos produtos que você usa em um atendimento: quanto pagou "
                "em cada um e para quantos atendimentos ele dá (ex: 'base de R$ 120 rende uns 40 atendimentos'). "
                "Se preferir, me diga um valor aproximado (ex: 'uns R$ 40') que eu marco como estimativa sua.")
    nomes = ", ".join(ROTULOS.get(c, c) for c in campos)
    return f"Sem problema: me diga um valor aproximado de {nomes} que eu marco como estimativa sua."


def _texto_anotado(dados: dict, estado: dict) -> str:
    """Repete o que foi entendido nesta mensagem, para a pessoa conferir."""
    itens = []
    marcados = set(dados.get("campos_estimados") or [])

    def est(campo):
        return " (estimativa sua)" if campo in marcados else ""

    if dados.get("preco_cobrado") is not None and dados.get("intencao") != "simular_preco":
        periodo = dados.get("periodicidade_preco") or estado.get("periodicidade_preco")
        itens.append(f"preço {_preco_periodo(dados['preco_cobrado'], periodo) if periodo else formatar_moeda(dados['preco_cobrado'])}")
    if dados.get("tempo_horas") is not None:
        itens.append(f"tempo {_fmt_num(dados['tempo_horas'])}h{est('tempo_horas')}")
    for novo in dados.get("materiais") or []:
        indice = _encontrar_material(estado.get("materiais") or [], novo["item"], True)
        if indice is None:
            continue
        m = estado["materiais"][indice]
        if m.get("valor") is None:
            continue
        if m.get("embalagem") and not m.get("rendimento"):
            itens.append(f"{m['item']} por {formatar_moeda(m['valor'])} (falta saber quanto rende)")
            continue
        d = _materiais_detalhe([m])[0]
        itens.append(_texto_material(d) + " por atendimento" if (d["rendimento"] or d["quantidade_usada"])
                     else _texto_material(d))
    if dados.get("deslocamento") is not None:
        itens.append(f"deslocamento {formatar_moeda(dados['deslocamento'])}{est('deslocamento')}"
                     if dados["deslocamento"] else "sem deslocamento")
    if dados.get("taxa_maquininha_pct") is not None:
        itens.append(f"taxa de {_fmt_num(dados['taxa_maquininha_pct'] * 100)}% no cartão"
                     if dados["taxa_maquininha_pct"] else "pagamento sem taxa")
    if dados.get("quantidade_clientes") is not None:
        itens.append(f"{_fmt_num(dados['quantidade_clientes'])} clientes{est('quantidade_clientes')}")
    if dados.get("encontros_por_mes") is not None:
        txt = f"{_fmt_num(dados['encontros_por_mes'])} encontros por mês"
        if dados.get("horas_por_encontro"):
            txt += f" de {_fmt_num(dados['horas_por_encontro'])}h"
        itens.append(txt)
    for c in dados.get("custos_fixos") or []:
        if c.get("valor") is not None:
            itens.append("nenhum gasto mensal fixo" if _chave(c["item"]) == "nenhum" and not c["valor"]
                         else _descricao_custo(c))
    if dados.get("atendimentos_por_mes") is not None:
        itens.append(f"{_fmt_num(dados['atendimentos_por_mes'])} atendimentos por mês")
    if dados.get("valor_hora_desejado") is not None:
        itens.append(f"meta de R$ {_fmt_num(dados['valor_hora_desejado'])} por hora")
    if dados.get("renda_desejada") is not None:
        itens.append(f"meta de {formatar_moeda(dados['renda_desejada'])} "
                     f"{NOME_PERIODO.get(dados.get('periodicidade_renda') or 'mensal')}")
    return ("Anotei: " + "; ".join(itens) + ".") if itens else ""


def _texto_confirmacao(estado: dict) -> str:
    nome = estado.get("nome_servico")
    estimados = estado.get("estimados") or []

    def est(campo):
        return " (estimativa sua)" if campo in estimados else ""

    linhas = [f"Confira os dados antes de eu calcular (**{nome}**):" if nome else "Confira os dados antes de eu calcular:"]
    taxa = estado.get("taxa_maquininha_pct")
    if taxa is None:
        pagamento = f"não informado: vou considerar {_fmt_num(TAXA_PADRAO * 100)}% de taxa de cartão"
    elif taxa == 0:
        pagamento = "sem taxa (Pix ou dinheiro)"
    else:
        pagamento = f"cartão com taxa de {_fmt_num(taxa * 100)}%"

    if _modo(estado) == "recorrente":
        periodo = estado.get("periodicidade_preco") or "mensal"
        if estado.get("preco_cobrado") is not None:
            linhas.append(f"- Preço: {_preco_periodo(estado['preco_cobrado'], periodo)} por cliente")
        else:
            linhas.append("- Preço: ainda não definido (vou calcular o preço para a sua meta)")
        linhas.append(f"- Clientes: {_fmt_num(estado['quantidade_clientes'])}{est('quantidade_clientes')}")
        if estado.get("encontros_por_mes"):
            txt = f"- Encontros: {_fmt_num(estado['encontros_por_mes'])} por mês"
            if estado.get("horas_por_encontro"):
                txt += f", de {_fmt_num(estado['horas_por_encontro'])}h cada"
            linhas.append(txt)
        custos = [c for c in estado.get("custos_fixos") or [] if c.get("valor") is not None]
        material = [c for c in custos if c.get("por_cliente") or c.get("por_encontro")]
        fixos = [c for c in custos if not c.get("por_cliente") and not c.get("por_encontro") and c["valor"]]
        linhas.append("- Material: " + ("; ".join(_descricao_custo(c) for c in material)
                                        if material else "não informado (considerei R$ 0)"))
        if fixos:
            total = sum((x["valor_mensal"] for x in _custos_fixos_mensais(fixos)), Decimal("0"))
            linhas.append("- Custos mensais: " + "; ".join(_descricao_custo(c) for c in fixos)
                          + f" (total {formatar_moeda(_arred(total))} por mês)")
        else:
            linhas.append("- Custos mensais: nenhum")
        if estado.get("horas_trabalhadas") is not None:
            linhas.append(f"- Tempo de trabalho: {_fmt_num(estado['horas_trabalhadas'])}h "
                          f"{NOME_PERIODO.get(estado.get('periodicidade_horas') or 'mensal')}")
        linhas.append(f"- Pagamento: {pagamento}")
        if estado.get("renda_desejada") is not None:
            linhas.append(f"- Meta: ganhar {formatar_moeda(estado['renda_desejada'])} "
                          f"{NOME_PERIODO.get(estado.get('periodicidade_renda') or 'mensal')}")
    else:
        detalhe = _materiais_detalhe(estado.get("materiais") or [])
        total = sum((_dec(d["custo"]) for d in detalhe), Decimal("0"))
        linhas.append(f"- Preço: {formatar_moeda(estado['preco_cobrado'])}")
        linhas.append(f"- Tempo total: {_fmt_num(estado['tempo_horas'])}h{est('tempo_horas')}")
        reais = [d for d in detalhe if not (_chave(d["item"]) == "nenhum" and not d["custo"])]
        if not reais:
            linhas.append("- Materiais: sem custo")
        elif len(reais) == 1:
            linhas.append("- " + _texto_material(reais[0], "Materiais por atendimento"))
        else:
            linhas.append(f"- Materiais por atendimento: {formatar_moeda(_arred(total))}")
            linhas += [f"- {_texto_material(d)}" for d in reais]
        desloc = estado.get("deslocamento")
        linhas.append(f"- Deslocamento: {formatar_moeda(desloc)}{est('deslocamento')}" if desloc
                      else "- Deslocamento: sem custo")
        if estado.get("outros_gastos"):
            linhas.append(f"- Outros gastos: {formatar_moeda(estado['outros_gastos'])}{est('outros_gastos')}")
        linhas.append(f"- Pagamento: {pagamento}")
        fixos = [c for c in estado.get("custos_fixos") or [] if c.get("valor")]
        if fixos:
            total_fixos = sum((x["valor_mensal"] for x in _custos_fixos_mensais(fixos)), Decimal("0"))
            if estado.get("atendimentos_por_mes"):
                linhas.append(f"- Custos mensais: {formatar_moeda(_arred(total_fixos))} ÷ "
                              f"{_fmt_num(estado['atendimentos_por_mes'])} atendimentos{est('atendimentos_por_mes')}")
            else:
                linhas.append(f"- Custos mensais: {formatar_moeda(_arred(total_fixos))} "
                              "(sem o número de atendimentos por mês, ficam fora da conta)")
        if estado.get("valor_hora_desejado"):
            linhas.append(f"- Sua meta: R$ {_fmt_num(estado['valor_hora_desejado'])} por hora")
    linhas += ["", "Está tudo certo? Responda **sim** para eu calcular, ou me diga o que corrigir."]
    return "\n".join(linhas)


_RESPOSTAS_SIM = {
    "s", "sim", "ok", "okay", "pode", "pode calcular", "pode sim", "isso", "isso mesmo", "certo", "correto",
    "esta certo", "ta certo", "tudo certo", "esta tudo certo", "confirmo", "beleza", "claro", "yes", "manda ver",
}


def _eh_confirmacao_simples(mensagem: str) -> bool:
    """'sim', 'ok, pode calcular'... sem números nem correções: confirma sem gastar chamada da IA."""
    texto = " ".join(re.sub(r"[^\w\s]", " ", _chave(mensagem)).split())
    if not texto or re.search(r"\d", texto) or len(texto.split()) > 4:
        return False
    return texto in _RESPOSTAS_SIM or texto.split()[0] in ("sim", "ok", "isso", "correto", "confirmo", "beleza")


# Função principal (integração com o front-end)

def processar_mensagem_usuario(
    mensagem: str,
    estado_atual: dict = None,
    extrator: Optional[Callable[[str, Optional[dict]], dict]] = None,
) -> dict:
    """
    Processa uma mensagem do chat e devolve SEMPRE um dict com as chaves:

        sucesso (bool)             False só em falha de IA/validação
        intencao (str|None)
        dados_extraidos (dict|None)  JSON normalizado vindo da IA
        estado (dict)              -> guarde e devolva na próxima chamada
        metricas (dict|None)       preenchido quando há cálculo
        simulacao (dict|None)      preenchido em simular_preco
        informacoes_faltantes (list[str])  dados essenciais que ainda faltam
        resposta (str)             texto pronto para exibir no chat (markdown simples)
        erro (str|None)            detalhe técnico para log, não exibir ao usuário

    Fluxo: pergunta o que falta -> mostra os dados e pede confirmação -> calcula.
    Depois do primeiro cálculo, correções recalculam direto.
    `extrator` permite injetar outra fonte de extração (ex.: mock nos testes).
    Em caso de erro o estado anterior é devolvido intacto.
    """
    estado_base = copy.deepcopy(estado_atual) if estado_atual else copy.deepcopy(ESTADO_VAZIO)
    resultado = {
        "sucesso": False,
        "intencao": None,
        "dados_extraidos": None,
        "estado": estado_base,
        "metricas": None,
        "simulacao": None,
        "informacoes_faltantes": [],
        "resposta": "",
        "erro": None,
    }

    if not mensagem or not str(mensagem).strip():
        resultado["resposta"] = (
            "Me conte sobre o seu serviço: quanto você cobra, quanto tempo leva, "
            "quanto gasta com materiais e com deslocamento."
        )
        resultado["erro"] = "mensagem_vazia"
        return resultado

    # Extração (IA). Um "sim" simples na hora de confirmar não precisa da IA.
    try:
        if estado_base.get("aguardando_confirmacao") and _eh_confirmacao_simples(str(mensagem)):
            dados = normalizar_extracao({"intencao": "confirmar"})
        else:
            dados = normalizar_extracao((extrator or extrair_dados_com_ia)(mensagem, estado_base))
    except Exception as exc:
        logger.exception("Falha na extração")
        resultado["erro"] = str(exc)
        resultado["resposta"] = (
            "Não consegui entender sua mensagem agora. Pode tentar de novo, "
            "de um jeito um pouco diferente?"
        )
        return resultado

    intencao = dados["intencao"]
    resultado["intencao"] = intencao
    resultado["dados_extraidos"] = dados

    # Estado + cálculo. O estado vem do navegador (não confiável):
    # valor malformado vira resposta amigável em vez de erro 500.
    try:
        novo_estado = atualizar_estado(estado_base, dados)
        faltantes = campos_faltantes(novo_estado)
        opcionais = campos_opcionais(novo_estado)
        blocos: list[str] = []
        if dados["invalidos"]:
            blocos.append(_texto_invalidos(dados["invalidos"]))
        if dados["nao_sabe"]:
            blocos.append(_texto_nao_sabe(dados["nao_sabe"]))
        bloqueado = bool(dados["invalidos"] or dados["nao_sabe"])

        if intencao == "simular_preco":
            novo_preco = dados.get("novo_preco_simulado")
            novos_clientes = dados.get("novo_quantidade_clientes") if _modo(novo_estado) == "recorrente" else None
            faltantes_sim = [c for c in faltantes if c not in ("preco_cobrado", "preco_recorrente")]
            resultado["informacoes_faltantes"] = faltantes_sim
            if bloqueado:
                pass
            elif not novo_preco and not novos_clientes:
                blocos.append("Qual valor você quer testar? Ex: 'E se eu cobrar R$ 90?'")
            elif faltantes_sim:
                blocos.append(_texto_faltantes(faltantes_sim, opcionais))
            else:
                simulacao = simular_novo_preco(novo_estado, novo_preco, novos_clientes)
                resultado["simulacao"] = simulacao
                blocos.append(formatar_explicacao_humana(simulacao, intencao))
                novo_estado["aguardando_confirmacao"] = False
                novo_estado["confirmado"] = True
        else:
            resultado["informacoes_faltantes"] = faltantes
            anotado = _texto_anotado(dados, novo_estado)
            if bloqueado or faltantes:
                novo_estado["aguardando_confirmacao"] = False
                if anotado:
                    blocos.append(anotado)
                if faltantes:
                    blocos.append(_texto_faltantes(faltantes, opcionais))
            elif PEDIR_CONFIRMACAO and not novo_estado.get("confirmado") and intencao != "confirmar":
                novo_estado["aguardando_confirmacao"] = True
                blocos.append(_texto_confirmacao(novo_estado))  # o resumo já mostra o que foi anotado
            else:
                novo_estado["aguardando_confirmacao"] = False
                novo_estado["confirmado"] = True
                metricas = _metricas_do_estado(novo_estado)
                resultado["metricas"] = metricas
                blocos.append(formatar_explicacao_humana(
                    {**metricas, "nome_servico": novo_estado.get("nome_servico")}, intencao
                ))
        resultado["resposta"] = "\n\n".join(b for b in blocos if b)
    except (ValueError, ArithmeticError, TypeError, KeyError, AttributeError) as exc:
        logger.warning("Dados inválidos na sessão: %s", exc)
        resultado["intencao"] = None
        resultado["erro"] = str(exc)
        resultado["metricas"] = None
        resultado["simulacao"] = None
        resultado["resposta"] = (
            "Algum valor ficou estranho (negativo ou taxa acima de 100%). "
            "Pode conferir e me mandar de novo?"
        )
        return resultado

    resultado["estado"] = novo_estado
    resultado["sucesso"] = True
    return resultado