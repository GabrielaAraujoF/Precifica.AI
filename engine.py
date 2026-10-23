# mensagem do usuário
# extrair_dados_com_ia()        (LLM: SÓ converte texto em JSON)
# normalizar_extracao()         (valida e corrige tipos vindos da LLM)
# atualizar_estado()            (mescla com a sessão)
# calcular_metricas_financeiras() / simular_novo_preco()   (matemática)
# formatar_explicacao_humana()  (texto final para o chat)
#
# Dois modelos de cobrança:
#   por_servico: preço por atendimento/peça (manicure, costura, artesanato...)
#   recorrente:  preço por período (assinatura, mensalidade, hospedagem, aulas...)


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

TAXA_PADRAO = 0.05  # 5% de taxa de maquininha quando o usuário não informa

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
# Gemini 3+: o Google recomenda temperatura padrão (1.0); valores menores podem causar loops e respostas lentas.
NIVEL_RACIOCINIO = os.getenv("GEMINI_THINKING", "low")  # minimal | low | medium | high
TIMEOUT_LLM_SEGUNDOS = float(os.getenv("GEMINI_TIMEOUT", "15"))  # por chamada
# Tempo máximo somando todas as tentativas. Depois disso o usuário recebe a mensagem amigável.
PRAZO_TOTAL_SEGUNDOS = float(os.getenv("GEMINI_PRAZO_TOTAL", "25"))

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
Sua ÚNICA função é converter a mensagem em linguagem natural do usuário em um JSON estruturado.
NUNCA faça cálculos matemáticos ou adicione explicações fora do JSON.

### Intenções Possíveis:
- "informar_dados": O usuário está fornecendo custos, tempos ou valores de serviços.
- "corrigir_dado": O usuário quer alterar um dado digitado antes.
- "simular_preco": O usuário pergunta o resultado de um valor alternativo (ex: "E se eu cobrar X?").

### Formato Obrigatório da Saída (JSON):
{
  "intencao": "informar_dados" | "corrigir_dado" | "simular_preco",
  "categoria_servico": string ou null,
  "nome_servico": string ou null,
  "preco_cobrado": float ou null,
  "novo_preco_simulado": float ou null,
  "tempo_horas": float ou null,
  "valor_hora_desejado": float ou null,
  "materiais": [
    {"item": string, "valor": float}
  ],
  "deslocamento": float ou null,
  "outros_gastos": float ou null,
  "taxa_maquininha_pct": float ou null,
  "campo_corrigido": string ou null,
  "informacoes_faltantes": [strings],
  "modelo_cobranca": "por_servico" | "recorrente" | null,
  "periodicidade_preco": "semanal" | "mensal" | "trimestral" | "semestral" | "anual" | null,
  "quantidade_clientes": float ou null,
  "novo_quantidade_clientes": float ou null,
  "atendimentos_por_mes": float ou null,
  "custos_fixos": [
    {"item": string, "valor": float, "periodicidade": "diario" | "semanal" | "mensal" | "trimestral" | "semestral" | "anual" | "unico" | null, "por_cliente": boolean}
  ],
  "horas_trabalhadas": float ou null,
  "periodicidade_horas": "diario" | "semanal" | "mensal" | null,
  "renda_desejada": float ou null,
  "periodicidade_renda": "semanal" | "mensal" | "anual" | null
}

### Regras de Conversão:
1. Converta textos de números para float (ex: "cinquenta reais" -> 50.0; "uma hora e meia" -> 1.5; "30 minutos" -> 0.5).
2. Se a mensagem for "E se eu cobrar R$ 80?", classifique como "simular_preco" e coloque 80.0 em "novo_preco_simulado".
3. Retorne APENAS o JSON.
4. Gastos com transporte, gasolina, Uber ou passagem vão em "deslocamento", nunca em "materiais".
5. "taxa_maquininha_pct" é uma fração decimal: 5% -> 0.05.
6. Se a mensagem não mencionar materiais, retorne "materiais": [].
7. Em "corrigir_dado", preencha SOMENTE o que foi corrigido. Se for um material, coloque o item em "materiais" com o novo valor, usando o mesmo nome do item que aparece no ESTADO ATUAL, e use "campo_corrigido": "materiais". Para outros campos, use o nome do campo (ex: "tempo_horas").
8. Em "simular_preco", se o usuário citar o preço atual (ex: "em vez de R$ 60"), coloque esse valor em "preco_cobrado".
9. O ESTADO ATUAL enviado junto é só contexto: não repita dados dele que o usuário não mencionou nesta mensagem.

### Modelo de cobrança:
10. "recorrente": o cliente paga por período (assinatura, mensalidade, plano, hospedagem, aulas mensais, contrato). "por_servico": cobra por atendimento, peça ou trabalho feito. Se não der para saber, null.
11. Em "recorrente": o preço que cada cliente paga vai em "preco_cobrado" com "periodicidade_preco"; o número de assinantes/alunos/clientes vai em "quantidade_clientes"; o tempo de trabalho vai em "horas_trabalhadas" + "periodicidade_horas" (ex: "3 horas por dia" -> 3.0 e "diario"); a meta de ganho vai em "renda_desejada" + "periodicidade_renda" (ex: "quero ganhar R$ 5000 por mês" -> 5000.0 e "mensal"). Nesse modelo NÃO use "tempo_horas"; use "valor_hora_desejado" só se o usuário falar em valor por hora.
12. Custos de estrutura (aluguel, domínio, servidor, softwares, ferramentas, internet, desenvolvimento, marketing) vão em "custos_fixos", com a periodicidade que o usuário disse. Se ele não disser o período, use null (não invente). Custo que existe para cada cliente (ex: "R$ 10 de servidor por aluno") tem "por_cliente": true.
13. Em "por_servico", o que é gasto em cada serviço continua em "materiais"; custos mensais da estrutura (ex: aluguel do salão) vão em "custos_fixos"; "atendo 40 clientes por mês" -> "atendimentos_por_mes": 40.0.
14. "E se eu tiver 20 assinantes?" -> "simular_preco" com "novo_quantidade_clientes": 20.0.
15. Se não houver custos fixos, retorne "custos_fixos": [].
"""

INTENCOES_VALIDAS = ("informar_dados", "corrigir_dado", "simular_preco")

CAMPOS_NUMERICOS = (
    "preco_cobrado",
    "novo_preco_simulado",
    "tempo_horas",
    "valor_hora_desejado",
    "deslocamento",
    "outros_gastos",
    "taxa_maquininha_pct",
    "quantidade_clientes",
    "novo_quantidade_clientes",
    "atendimentos_por_mes",
    "horas_trabalhadas",
    "renda_desejada",
)
CAMPOS_TEXTO = ("categoria_servico", "nome_servico", "campo_corrigido")
CAMPOS_PERIODO = ("periodicidade_preco", "periodicidade_horas", "periodicidade_renda")
MODELOS_COBRANCA = ("por_servico", "recorrente")

# Campos que ficam guardados na sessão (o "estado").
ESTADO_VAZIO: dict = {
    "categoria_servico": None,
    "nome_servico": None,
    "preco_cobrado": None,
    "tempo_horas": None,
    "valor_hora_desejado": None,
    "materiais": [],
    "deslocamento": None,
    "outros_gastos": None,
    "taxa_maquininha_pct": None,
    "modelo_cobranca": None,
    "periodicidade_preco": None,
    "quantidade_clientes": None,
    "atendimentos_por_mes": None,
    "custos_fixos": [],
    "horas_trabalhadas": None,
    "periodicidade_horas": None,
    "renda_desejada": None,
    "periodicidade_renda": None,
}

# Sem estes três não dá para calcular. Materiais/deslocamento/outros assumem 0
# e a taxa assume TAXA_PADRAO.
CAMPOS_OBRIGATORIOS = {
    "preco_cobrado": "quanto você cobra pelo serviço (ex: 'cobro R$ 80')",
    "tempo_horas": "quanto tempo leva para fazer (ex: 'demoro 2 horas')",
    "valor_hora_desejado": "quanto quer ganhar por hora de trabalho (ex: 'quero R$ 20 por hora')",
}
# Obrigatórios no modelo recorrente: clientes + meta de ganho (renda ou horas × valor/hora)
DESCRICOES_FALTANTES = {
    **CAMPOS_OBRIGATORIOS,
    "quantidade_clientes": "quantos clientes ou assinantes você tem (ex: 'tenho 12 assinantes')",
    "renda_desejada": "quanto quer ganhar com isso por mês (ex: 'quero ganhar R$ 5.000 por mês')",
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


def para_float(valor: Any) -> Optional[float]:
    """
    Converte números vindos da LLM em float, aceitando formatos brasileiros.
    Exemplos: 35 -> 35.0 | "35,50" -> 35.5 | "R$ 1.234,56" -> 1234.56
    Retorna None para valores ausentes, inválidos ou negativos.
    """
    if valor is None or isinstance(valor, bool):
        return None
    if isinstance(valor, (int, float)):
        numero = float(valor)
    elif isinstance(valor, str):
        texto = re.sub(r"[^\d,.\-]", "", valor.strip())
        if not texto or texto in {"-", ",", "."}:
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
    else:
        return None

    if math.isnan(numero) or math.isinf(numero) or numero < 0:
        return None
    return numero


def _normalizar_taxa(valor: Any) -> Optional[float]:
    """Aceita 0.05 ou 5 (quando a LLM devolve em pontos percentuais)."""
    taxa = para_float(valor)
    if taxa is None:
        return None
    if taxa >= 1:
        taxa = taxa / 100
    return taxa if 0 <= taxa < 1 else None


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
    Garante que a saída da LLM tenha exatamente o schema esperado, com tipos
    corretos. Nunca confia cegamente na LLM: campos inválidos viram None.
    É idempotente (pode ser chamada mais de uma vez sem efeito colateral).
    """
    if not isinstance(bruto, dict):
        raise ErroExtracao("A resposta da IA não é um objeto JSON.")

    dados: dict = {}
    for campo in CAMPOS_NUMERICOS:
        if campo == "taxa_maquininha_pct":
            dados[campo] = _normalizar_taxa(bruto.get(campo))
        else:
            dados[campo] = para_float(bruto.get(campo))

    for campo in CAMPOS_TEXTO:
        valor = bruto.get(campo)
        dados[campo] = valor.strip() if isinstance(valor, str) and valor.strip() else None

    materiais = []
    for m in bruto.get("materiais") or []:
        if not isinstance(m, dict):
            continue
        item = m.get("item")
        valor = para_float(m.get("valor"))
        if isinstance(item, str) and item.strip() and valor is not None:
            materiais.append({"item": item.strip(), "valor": valor})
    dados["materiais"] = materiais

    for campo in CAMPOS_PERIODO:
        dados[campo] = _normalizar_periodo(bruto.get(campo))
    if dados["periodicidade_preco"] == "unico":
        dados["periodicidade_preco"] = None

    custos_fixos = []
    for c in bruto.get("custos_fixos") or []:
        if not isinstance(c, dict):
            continue
        item = c.get("item")
        valor = para_float(c.get("valor"))
        if isinstance(item, str) and item.strip() and valor is not None:
            custos_fixos.append({
                "item": item.strip(),
                "valor": valor,
                "periodicidade": _normalizar_periodo(c.get("periodicidade")),
                "por_cliente": c.get("por_cliente") is True,
            })
    dados["custos_fixos"] = custos_fixos

    for campo in ("quantidade_clientes", "novo_quantidade_clientes", "atendimentos_por_mes"):
        if dados[campo] is not None and dados[campo] <= 0:
            dados[campo] = None

    modelo = bruto.get("modelo_cobranca")
    if modelo not in MODELOS_COBRANCA:
        # Sinais fortes de cobrança recorrente quando a IA não classificou
        sinais = (dados["quantidade_clientes"], dados["novo_quantidade_clientes"], dados["periodicidade_preco"])
        modelo = "recorrente" if any(v is not None for v in sinais) else None
    dados["modelo_cobranca"] = modelo

    faltantes = bruto.get("informacoes_faltantes") or []
    dados["informacoes_faltantes"] = [str(f) for f in faltantes if f] if isinstance(faltantes, list) else []

    intencao = bruto.get("intencao")
    simulando = dados["novo_preco_simulado"] is not None or dados["novo_quantidade_clientes"] is not None
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
            thinking_config=types.ThinkingConfig(thinking_level=NIVEL_RACIOCINIO),
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
    contexto = {k: estado_atual.get(k) for k in ESTADO_VAZIO}
    if not any(v not in (None, []) for v in contexto.values()):
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
                raise ErroExtracao(f"Falha na chamada ao Gemini ({modelo}): {exc}") from exc

    raise ErroExtracao("Nenhum modelo respondeu. Tentativas: " + " | ".join(falhas))


# Motor financeiro determinístico

def calcular_metricas_financeiras(
    materiais: list[dict],
    deslocamento: float,
    outros_gastos: float,
    tempo_horas: float,
    valor_hora_desejado: float,
    preco_cobrado: float,
    taxa_maquininha_pct: float = TAXA_PADRAO,
    custos_fixos_rateados: float = 0,
) -> dict:
    """
    Calcula todos os custos, sobra real, margem e preço mínimo recomendável (por serviço).

        C_d   = soma(materiais) + deslocamento + outros_gastos
        C_mo  = tempo_horas * valor_hora_desejado
        C_f   = custos fixos mensais / atendimentos por mês (0 se não informado)
        C_t   = C_d + C_mo + C_f
        V_l   = preco_cobrado * (1 - taxa)
        S     = V_l - C_t
        Marg% = (S / preco_cobrado) * 100        (None se preço = 0)
        P_min = C_t / (1 - taxa)

    Valores None são tratados como 0 (e taxa None como TAXA_PADRAO).
    Lança ValueError para valores negativos ou taxa fora de [0, 1).
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
    custo_mao_obra = entradas["tempo_horas"] * entradas["valor_hora_desejado"]
    custo_total = custo_direto + custo_mao_obra + entradas["custos_fixos_rateados"]

    preco = entradas["preco_cobrado"]
    valor_taxa = preco * taxa
    valor_liquido = preco - valor_taxa  # = preco * (1 - taxa)
    sobra = valor_liquido - custo_total
    margem = (sobra / preco * 100) if preco > 0 else None
    preco_minimo = custo_total / (1 - taxa)

    return {
        "modelo_cobranca": "por_servico",
        "total_materiais": _arred(total_materiais),
        "deslocamento": _arred(entradas["deslocamento"]),
        "outros_gastos": _arred(entradas["outros_gastos"]),
        "custo_direto": _arred(custo_direto),
        "tempo_horas": float(entradas["tempo_horas"]),
        "valor_hora_desejado": _arred(entradas["valor_hora_desejado"]),
        "custo_mao_obra": _arred(custo_mao_obra),
        "custos_fixos_rateados": _arred(entradas["custos_fixos_rateados"]),
        "custo_total": _arred(custo_total),
        "preco_cobrado": _arred(preco),
        "taxa_maquininha_pct": float(taxa),
        "valor_taxa": _arred(valor_taxa),
        "valor_liquido": _arred(valor_liquido),
        "sobra_liquida": _arred(sobra),
        "margem_pct": _arred(margem) if margem is not None else None,
        "preco_minimo": _arred(preco_minimo),
        "cobre_custos": sobra >= 0,
    }


def _modo(estado: dict) -> str:
    return "recorrente" if estado.get("modelo_cobranca") == "recorrente" else "por_servico"


def _custos_fixos_mensais(custos_fixos: list[dict]) -> list[dict]:
    """Converte cada custo fixo para base mensal. Sem período informado = mensal (premissa)."""
    itens = []
    for c in custos_fixos or []:
        periodo = c.get("periodicidade")
        fator = FATOR_MENSAL.get(periodo or "mensal")
        valor = _dec(c.get("valor"))
        if valor < 0:
            raise ValueError("Valores não podem ser negativos.")
        itens.append({
            "item": c["item"],
            "valor": valor,
            "periodicidade": periodo,
            "por_cliente": bool(c.get("por_cliente")),
            "valor_mensal": valor * fator,
        })
    return itens


def _metricas_do_estado(estado: dict, preco: Optional[float] = None, clientes: Optional[float] = None) -> dict:
    """Atalho: calcula as métricas a partir de um estado de sessão, conforme o modelo de cobrança."""
    preco_usado = estado.get("preco_cobrado") if preco is None else preco
    if _modo(estado) == "recorrente":
        return calcular_metricas_recorrentes(
            custos_fixos=estado.get("custos_fixos") or [],
            quantidade_clientes=estado.get("quantidade_clientes") if clientes is None else clientes,
            renda_desejada=estado.get("renda_desejada"),
            periodicidade_renda=estado.get("periodicidade_renda"),
            horas_trabalhadas=estado.get("horas_trabalhadas"),
            periodicidade_horas=estado.get("periodicidade_horas"),
            valor_hora_desejado=estado.get("valor_hora_desejado"),
            preco_cobrado=preco_usado,
            periodicidade_preco=estado.get("periodicidade_preco"),
            taxa_maquininha_pct=estado.get("taxa_maquininha_pct"),
        )

    fixos = _custos_fixos_mensais(estado.get("custos_fixos") or [])
    atendimentos = estado.get("atendimentos_por_mes")
    rateio = Decimal("0")
    if fixos and atendimentos:
        rateio = sum((c["valor_mensal"] for c in fixos), Decimal("0")) / _dec(atendimentos)
    metricas = calcular_metricas_financeiras(
        materiais=estado.get("materiais") or [],
        deslocamento=estado.get("deslocamento"),
        outros_gastos=estado.get("outros_gastos"),
        tempo_horas=estado.get("tempo_horas"),
        valor_hora_desejado=estado.get("valor_hora_desejado"),
        preco_cobrado=preco_usado,
        taxa_maquininha_pct=estado.get("taxa_maquininha_pct"),
        custos_fixos_rateados=rateio,
    )
    metricas["custos_fixos_mensais"] = _arred(sum((c["valor_mensal"] for c in fixos), Decimal("0")))
    metricas["atendimentos_por_mes"] = atendimentos
    return metricas


def calcular_metricas_recorrentes(
    custos_fixos: list[dict],
    quantidade_clientes: float,
    renda_desejada: Optional[float],
    periodicidade_renda: Optional[str],
    horas_trabalhadas: Optional[float],
    periodicidade_horas: Optional[str],
    valor_hora_desejado: Optional[float],
    preco_cobrado: Optional[float],
    periodicidade_preco: Optional[str],
    taxa_maquininha_pct: Optional[float] = TAXA_PADRAO,
) -> dict:
    """
    Cobrança recorrente (assinatura, mensalidade, aulas...). Tudo em base MENSAL.

        F     = soma(custos fixos convertidos para mês)
        Ccli  = soma(custos por cliente/mês) * clientes
        R     = renda desejada/mês  (ou horas/mês * valor_hora_desejado)
        C_t   = F + Ccli + R
        Rec   = preço/mês * clientes ; V_l = Rec * (1 - taxa) ; S = V_l - C_t
        P_min = C_t / (clientes * (1 - taxa))          (por cliente, por mês)
        N_min = (F + R) / (preço/mês * (1 - taxa) - custo por cliente)   (arredondado p/ cima)

    preco_cobrado é opcional: sem ele, calcula só o preço recomendado.
    """
    taxa = _dec(TAXA_PADRAO if taxa_maquininha_pct is None else taxa_maquininha_pct)
    if not (Decimal("0") <= taxa < Decimal("1")):
        raise ValueError("taxa_maquininha_pct deve estar entre 0 e 1 (ex: 0.05 = 5%).")
    clientes = _dec(quantidade_clientes)
    if clientes <= 0:
        raise ValueError("quantidade_clientes deve ser maior que zero.")

    premissas: list[str] = []
    itens = _custos_fixos_mensais(custos_fixos)
    for c in itens:
        if c["periodicidade"] is None:
            premissas.append(f"\"{c['item']}\" como gasto mensal")
        elif c["periodicidade"] == "unico":
            premissas.append(f"o gasto único com \"{c['item']}\" dividido em {MESES_AMORTIZACAO} meses")
    fixos = sum((c["valor_mensal"] for c in itens if not c["por_cliente"]), Decimal("0"))
    por_cliente = sum((c["valor_mensal"] for c in itens if c["por_cliente"]), Decimal("0"))

    horas_mes = None
    if horas_trabalhadas is not None:
        periodo_h = periodicidade_horas or "mensal"
        if periodicidade_horas is None:
            premissas.append("as horas informadas como horas por mês")
        elif periodo_h == "diario":
            premissas.append(f"{DIAS_TRABALHO_MES} dias de trabalho por mês")
        horas_mes = _dec(horas_trabalhadas) * FATOR_TRABALHO[periodo_h]

    if renda_desejada is not None:
        if periodicidade_renda is None:
            premissas.append("a renda desejada como mensal")
        renda = _dec(renda_desejada) * FATOR_TRABALHO[periodicidade_renda or "mensal"]
        origem_renda = "renda_desejada"
    elif horas_mes is not None and valor_hora_desejado is not None:
        renda = horas_mes * _dec(valor_hora_desejado)
        origem_renda = "horas_x_valor_hora"
    else:
        raise ValueError("Informe a renda desejada ou horas trabalhadas e valor da hora.")
    if renda < 0:
        raise ValueError("Valores não podem ser negativos.")

    custo_clientes = por_cliente * clientes
    custo_total = fixos + custo_clientes + renda
    preco_minimo_mensal = custo_total / (clientes * (1 - taxa))

    fator_preco = FATOR_MENSAL[periodicidade_preco or "mensal"]
    if preco_cobrado is not None and periodicidade_preco is None:
        premissas.append("o preço informado como mensal")

    resultado = {
        "modelo_cobranca": "recorrente",
        "custos_fixos": [
            {
                "item": c["item"],
                "valor": _arred(c["valor"]),
                "periodicidade": c["periodicidade"],
                "por_cliente": c["por_cliente"],
                "valor_mensal": _arred(c["valor_mensal"]),
            }
            for c in itens
        ],
        "custos_fixos_mensais": _arred(fixos),
        "custo_por_cliente_mensal": _arred(por_cliente),
        "custo_clientes_mensal": _arred(custo_clientes),
        "renda_desejada_mensal": _arred(renda),
        "origem_renda": origem_renda,
        "horas_mensais": _arred(horas_mes) if horas_mes is not None else None,
        "valor_hora_implicito": _arred(renda / horas_mes) if horas_mes else None,
        "custo_total": _arred(custo_total),
        "quantidade_clientes": float(clientes),
        "taxa_maquininha_pct": float(taxa),
        "periodicidade_preco": periodicidade_preco or "mensal",
        "preco_minimo": _arred(preco_minimo_mensal),
        "preco_minimo_periodo": _arred(preco_minimo_mensal / fator_preco),
        "preco_cobrado": None,
        "preco_mensal_por_cliente": None,
        "receita_bruta": None,
        "valor_taxa": None,
        "valor_liquido": None,
        "sobra_liquida": None,
        "margem_pct": None,
        "clientes_necessarios": None,
        "cobre_custos": None,
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
        sobra = liquido - custo_total
        contribuicao = preco_mensal * (1 - taxa) - por_cliente
        necessarios = (
            int(math.ceil((fixos + renda) / contribuicao)) if contribuicao > 0 else None
        )
        resultado.update({
            "preco_cobrado": _arred(preco),
            "preco_mensal_por_cliente": _arred(preco_mensal),
            "receita_bruta": _arred(receita),
            "valor_taxa": _arred(valor_taxa),
            "valor_liquido": _arred(liquido),
            "sobra_liquida": _arred(sobra),
            "margem_pct": _arred(sobra / receita * 100) if receita > 0 else None,
            "clientes_necessarios": necessarios,
            "cobre_custos": sobra >= 0,
        })
    return resultado


def simular_novo_preco(
    estado_atual: dict, novo_preco: Optional[float], novos_clientes: Optional[float] = None
) -> dict:
    """
    Recalcula as métricas financeiras mantendo os custos do estado atual,
    porém alterando o preço cobrado para o 'novo_preco'.
    Retorna a diferença de sobra líquida entre o preço original e o simulado.
    No modelo recorrente também aceita 'novos_clientes' (ex.: "e se eu tiver 20 assinantes?").
    Sem preço original (ou sem preço nenhum, no recorrente), as comparações vêm como None.
    """
    recorrente = _modo(estado_atual) == "recorrente"
    if novo_preco is not None and novo_preco <= 0:
        raise ValueError("novo_preco deve ser maior que zero.")
    if novo_preco is None and not (recorrente and novos_clientes):
        raise ValueError("novo_preco deve ser maior que zero.")

    simuladas = _metricas_do_estado(estado_atual, preco=novo_preco, clientes=novos_clientes)
    original = None
    diferenca = None
    tem_original = estado_atual.get("preco_cobrado") is not None or recorrente
    if tem_original:
        original = _metricas_do_estado(estado_atual)
        if original["sobra_liquida"] is not None and simuladas["sobra_liquida"] is not None:
            diferenca = _arred(_dec(simuladas["sobra_liquida"]) - _dec(original["sobra_liquida"]))

    return {
        "modelo_cobranca": "recorrente" if recorrente else "por_servico",
        "preco_original": original["preco_cobrado"] if original else None,
        "preco_simulado": simuladas["preco_cobrado"],
        "clientes_original": original.get("quantidade_clientes") if original else None,
        "clientes_simulado": simuladas.get("quantidade_clientes"),
        "sobra_original": original["sobra_liquida"] if original else None,
        "sobra_simulada": simuladas["sobra_liquida"],
        "diferenca_sobra": diferenca,
        "metricas_original": original,
        "metricas_simuladas": simuladas,
    }


def _encontrar_material(materiais: list[dict], nome: str, aproximado: bool) -> Optional[int]:
    """Índice do material com o mesmo nome (ou nome contido, se aproximado)."""
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


def atualizar_estado(estado_anterior: dict, dados_novos: dict) -> dict:
    """
    Mescla o estado anterior da sessão com os novos dados extraídos da IA.
    Suporta atualização de itens específicos quando intencao == 'corrigir_dado'.

    Regras:
      - Nunca altera o dicionário recebido (trabalha sobre uma cópia).
      - Campos None nos dados novos NÃO apagam o que já existe.
      - Materiais com o mesmo nome são substituídos; novos são adicionados.
        Em 'corrigir_dado' a busca por nome é aproximada ("tecido" casa com
        "Tecido de algodão").
      - Em 'simular_preco' o preço original da sessão é preservado.
    """
    estado = copy.deepcopy(ESTADO_VAZIO)
    estado.update(copy.deepcopy(estado_anterior or {}))
    estado["materiais"] = list(estado.get("materiais") or [])
    estado["custos_fixos"] = list(estado.get("custos_fixos") or [])
    intencao = dados_novos.get("intencao")

    # Modelo de cobrança: só passa a "recorrente"; "por_servico" só define se ainda não houver
    modelo = dados_novos.get("modelo_cobranca")
    if modelo == "recorrente" or (modelo == "por_servico" and estado.get("modelo_cobranca") is None):
        estado["modelo_cobranca"] = modelo

    campos_simples = [c for c in ESTADO_VAZIO if c not in ("materiais", "custos_fixos", "modelo_cobranca")]
    for campo in campos_simples:
        valor = dados_novos.get(campo)
        if valor is None:
            continue
        if intencao == "simular_preco" and campo == "preco_cobrado" and estado.get("preco_cobrado") is not None:
            continue  # o preço "de referência" da simulação não sobrescreve a sessão
        estado[campo] = valor

    aproximado = intencao == "corrigir_dado"
    for novo in dados_novos.get("materiais") or []:
        indice = _encontrar_material(estado["materiais"], novo["item"], aproximado)
        if indice is None:
            estado["materiais"].append(dict(novo))
        else:  # mantém o nome original, atualiza o valor
            estado["materiais"][indice] = {"item": estado["materiais"][indice]["item"], "valor": novo["valor"]}

    for novo in dados_novos.get("custos_fixos") or []:
        indice = _encontrar_material(estado["custos_fixos"], novo["item"], aproximado)
        if indice is None:
            estado["custos_fixos"].append(dict(novo))
        else:
            atual = estado["custos_fixos"][indice]
            estado["custos_fixos"][indice] = {
                "item": atual["item"],
                "valor": novo["valor"],
                "periodicidade": novo.get("periodicidade") or atual.get("periodicidade"),
                "por_cliente": novo.get("por_cliente", atual.get("por_cliente", False)),
            }

    return estado


def campos_faltantes(estado: dict) -> list[str]:
    """Campos obrigatórios ainda não informados na sessão (dependem do modelo de cobrança)."""
    if _modo(estado) == "recorrente":
        faltantes = []
        if not estado.get("quantidade_clientes"):
            faltantes.append("quantidade_clientes")
        tem_meta = estado.get("renda_desejada") is not None or (
            estado.get("horas_trabalhadas") is not None and estado.get("valor_hora_desejado") is not None
        )
        if not tem_meta:
            faltantes.append("renda_desejada")
        return faltantes
    return [c for c in CAMPOS_OBRIGATORIOS if estado.get(c) is None]


# Respostas em linguagem simples

def formatar_moeda(valor: float) -> str:
    """1234.5 -> 'R$ 1.234,50' | -5 -> '-R$ 5,00'"""
    texto = f"{abs(valor):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{'-' if valor < 0 else ''}R$ {texto}"


def _fmt_num(valor: float) -> str:
    """1.5 -> '1,5' | 2.0 -> '2' | 1.25 -> '1,25'"""
    return f"{valor:.2f}".rstrip("0").rstrip(".").replace(".", ",")


def _fmt_pct(valor: Optional[float]) -> str:
    return "—" if valor is None else f"{valor:.1f}%".replace(".", ",")


def _preco_sugerido_para_teste(m: dict) -> int:
    """Sugestão de valor para o usuário simular (15% acima, múltiplo de 5)."""
    base = max(m["preco_cobrado"], m["preco_minimo"]) * 1.15
    return int(math.ceil(base / 5) * 5)


def _texto_resumo(m: dict) -> str:
    nome = m.get("nome_servico")
    taxa_txt = (
        f"considerando {_fmt_num(m['taxa_maquininha_pct'] * 100)}% de taxa do cartão"
        if m["taxa_maquininha_pct"] > 0
        else "sem taxa de cartão"
    )
    linhas = [
        f"Resumo do seu serviço (**{nome}**):" if nome else "Resumo do seu serviço:",
        f"- **Custos diretos (materiais + deslocamento + outros):** {formatar_moeda(m['custo_direto'])}",
        f"- **Sua mão de obra ({_fmt_num(m['tempo_horas'])}h a R$ {_fmt_num(m['valor_hora_desejado'])}/h):** "
        f"{formatar_moeda(m['custo_mao_obra'])}",
    ]
    if m.get("custos_fixos_rateados"):
        linhas.append(
            f"- **Custos fixos rateados ({formatar_moeda(m['custos_fixos_mensais'])}/mês ÷ "
            f"{_fmt_num(m['atendimentos_por_mes'])} atendimentos):** {formatar_moeda(m['custos_fixos_rateados'])}"
        )
    linhas += [
        f"- **Custo total do serviço:** {formatar_moeda(m['custo_total'])}",
        "",
        f"Para o preço cobrado de **{formatar_moeda(m['preco_cobrado'])}** ({taxa_txt}):",
        f"- Você recebe líquido: **{formatar_moeda(m['valor_liquido'])}**",
    ]
    if m["cobre_custos"]:
        linhas.append(
            f"- **Sobra real para o seu negócio:** {formatar_moeda(m['sobra_liquida'])} "
            f"(margem de {_fmt_pct(m['margem_pct'])})"
        )
        linhas.append("")
        linhas.append(
            f"💡 **Preço mínimo recomendado:** {formatar_moeda(m['preco_minimo'])} "
            "(para cobrir custos e mão de obra sem ter prejuízo)."
        )
    else:
        linhas.append(
            f"- ⚠️ **Faltam {formatar_moeda(abs(m['sobra_liquida']))}** para cobrir seus custos e "
            "a sua hora desejada. Nesse preço você trabalha ganhando menos do que planejou."
        )
        linhas.append("")
        linhas.append(
            f"💡 **Cobre pelo menos {formatar_moeda(m['preco_minimo'])}** para não ter prejuízo."
        )
    if m.get("custos_fixos_mensais") and not m.get("custos_fixos_rateados"):
        linhas.append("")
        linhas.append(
            f"ℹ️ Anotei seus custos fixos de {formatar_moeda(m['custos_fixos_mensais'])} por mês, mas eles ainda "
            "não entraram na conta. Me diga quantos atendimentos você faz por mês para incluí-los."
        )
    linhas.append("")
    linhas.append(
        f"Quer testar outro valor? Pergunte: 'E se eu cobrar R$ {_preco_sugerido_para_teste(m)}?'"
    )
    return "\n".join(linhas)


def _texto_simulacao(s: dict) -> str:
    sim = s["metricas_simuladas"]
    orig = s["metricas_original"]
    if orig is None:
        linhas = [
            f"🔎 Simulação cobrando **{formatar_moeda(s['preco_simulado'])}**:",
            f"- Você recebe líquido: **{formatar_moeda(sim['valor_liquido'])}**",
            f"- Sobra real: **{formatar_moeda(sim['sobra_liquida'])}** (margem de {_fmt_pct(sim['margem_pct'])})",
        ]
    else:
        dif = s["diferenca_sobra"]
        if dif > 0:
            comparacao = f"{formatar_moeda(dif)} a mais por serviço"
        elif dif < 0:
            comparacao = f"{formatar_moeda(abs(dif))} a menos por serviço"
        else:
            comparacao = "mesma sobra de antes"
        linhas = [
            f"🔎 Simulação: cobrando **{formatar_moeda(s['preco_simulado'])}** "
            f"em vez de **{formatar_moeda(s['preco_original'])}**",
            f"- Você recebe líquido: **{formatar_moeda(sim['valor_liquido'])}** "
            f"(antes {formatar_moeda(orig['valor_liquido'])})",
            f"- Sobra real: **{formatar_moeda(sim['sobra_liquida'])}** "
            f"(antes {formatar_moeda(orig['sobra_liquida'])}) → {comparacao}",
            f"- Margem: {_fmt_pct(sim['margem_pct'])} (antes {_fmt_pct(orig['margem_pct'])})",
        ]
    linhas.append("")
    if sim["cobre_custos"]:
        linhas.append(
            f"💡 Seus custos continuam os mesmos: o preço mínimo segue em {formatar_moeda(sim['preco_minimo'])}."
        )
    else:
        linhas.append(
            f"⚠️ Esse valor não cobre seus custos e sua hora desejada. "
            f"O mínimo é {formatar_moeda(sim['preco_minimo'])}."
        )
    return "\n".join(linhas)


def _preco_periodo(valor: float, periodo: str) -> str:
    return f"{formatar_moeda(valor)} {NOME_PERIODO.get(periodo, 'por mês')}"


def _texto_resumo_recorrente(m: dict) -> str:
    nome = m.get("nome_servico")
    periodo = m["periodicidade_preco"]
    clientes = _fmt_num(m["quantidade_clientes"])
    linhas = [
        f"Resumo do seu negócio recorrente (**{nome}**):" if nome else "Resumo do seu negócio recorrente:",
        "",
        "Custos por mês:",
    ]
    for c in m["custos_fixos"]:
        origem = ""
        if c["periodicidade"] not in (None, "mensal"):
            origem = f" ({formatar_moeda(c['valor'])} {NOME_PERIODO[c['periodicidade']]})"
        sufixo = f" por cliente × {clientes}" if c["por_cliente"] else ""
        linhas.append(f"- {c['item']}: {formatar_moeda(c['valor_mensal'])}{sufixo}{origem}")
    if len(m["custos_fixos"]) > 1:
        total_custos = m["custos_fixos_mensais"] + m["custo_clientes_mensal"]
        linhas.append(f"- **Total de custos:** {formatar_moeda(total_custos)}")
    renda_txt = f"- **Sua renda desejada:** {formatar_moeda(m['renda_desejada_mensal'])}"
    if m["horas_mensais"]:
        renda_txt += (
            f" (≈ {_fmt_num(m['horas_mensais'])}h de trabalho, "
            f"{formatar_moeda(m['valor_hora_implicito'])} por hora)"
        )
    linhas += [
        renda_txt,
        f"- **Custo total mensal:** {formatar_moeda(m['custo_total'])}",
        "",
    ]
    taxa_txt = (
        f"{_fmt_num(m['taxa_maquininha_pct'] * 100)}% de taxa de pagamento"
        if m["taxa_maquininha_pct"] > 0 else "sem taxa de pagamento"
    )
    if m["preco_cobrado"] is not None:
        linhas.append(
            f"Hoje: {clientes} clientes × {_preco_periodo(m['preco_cobrado'], periodo)} = "
            f"**{formatar_moeda(m['receita_bruta'])} por mês** "
            f"({formatar_moeda(m['valor_liquido'])} líquido, com {taxa_txt})"
        )
        if m["cobre_custos"]:
            linhas.append(
                f"- **Sobra mensal:** {formatar_moeda(m['sobra_liquida'])} além da sua renda "
                f"(margem de {_fmt_pct(m['margem_pct'])})"
            )
        else:
            linhas.append(
                f"- ⚠️ **Faltam {formatar_moeda(abs(m['sobra_liquida']))} por mês** "
                "para cobrir os custos e a renda que você quer."
            )
        linhas.append("")
    recomendado = f"💡 **Preço mínimo recomendado:** {_preco_periodo(m['preco_minimo_periodo'], periodo)} por cliente"
    if periodo != "mensal":
        recomendado += f" (equivale a {formatar_moeda(m['preco_minimo'])} por mês)"
    recomendado += f", com {clientes} clientes e {taxa_txt}."
    linhas.append(recomendado)
    if m["clientes_necessarios"] is not None and not m["cobre_custos"]:
        linhas.append(
            f"Ou, mantendo {_preco_periodo(m['preco_cobrado'], periodo)}, você precisaria de "
            f"**{m['clientes_necessarios']} clientes**."
        )
    elif m["preco_cobrado"] is not None and m["clientes_necessarios"] is None and not m["cobre_custos"]:
        linhas.append("Nesse preço, cada novo cliente não cobre nem o próprio custo: o preço precisa subir.")
    if m["premissas"]:
        linhas += ["", "ℹ️ Para essa conta, considerei: " + "; ".join(m["premissas"]) + ". Se algo estiver diferente, é só me corrigir."]
    sugestao = int(math.ceil(m["preco_minimo_periodo"] / 10) * 10)
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
        if orig and orig["preco_cobrado"] is not None:
            cabecalho += (
                f" (hoje: {_fmt_num(orig['quantidade_clientes'])} × "
                f"{_preco_periodo(orig['preco_cobrado'], periodo)})"
            )
        linhas += [
            cabecalho,
            f"- Receita líquida: **{formatar_moeda(sim['valor_liquido'])} por mês**"
            + (f" (antes {formatar_moeda(orig['valor_liquido'])})" if orig and orig["valor_liquido"] is not None else ""),
        ]
        sobra_txt = f"- Sobra mensal: **{formatar_moeda(sim['sobra_liquida'])}**"
        dif = s["diferenca_sobra"]
        if dif is not None:
            sobra_txt += f" (antes {formatar_moeda(orig['sobra_liquida'])})"
            if dif:
                sobra_txt += f" → {formatar_moeda(abs(dif))} {'a mais' if dif > 0 else 'a menos'} por mês"
        linhas.append(sobra_txt)
        if not sim["cobre_custos"]:
            linhas.append("- ⚠️ Ainda não cobre os custos e a renda que você quer.")
    else:
        linhas.append(f"🔎 Simulação com **{clientes_sim} clientes**:")
    linhas.append("")
    minimo = f"💡 Preço mínimo com {clientes_sim} clientes: **{_preco_periodo(sim['preco_minimo_periodo'], periodo)}**"
    if orig and orig["quantidade_clientes"] != sim["quantidade_clientes"]:
        minimo += (
            f" (com {_fmt_num(orig['quantidade_clientes'])}: "
            f"{_preco_periodo(orig['preco_minimo_periodo'], periodo)})"
        )
    linhas.append(minimo + ".")
    return "\n".join(linhas)


def formatar_explicacao_humana(metricas: dict, intencao: str) -> str:
    """
    Converte os números calculados em texto simples.
      - informar_dados / corrigir_dado: recebe o dict de calcular_metricas_financeiras
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


def _texto_faltantes(faltantes: list[str], intencao: str) -> str:
    inicio = "✅ Anotado!" if intencao == "corrigir_dado" else "Anotei!"
    itens = "\n".join(f"- {DESCRICOES_FALTANTES[c]}" for c in faltantes)
    return f"{inicio} Para calcular seu preço justo, ainda preciso saber:\n{itens}"


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
        metricas (dict|None)       preenchido em informar/corrigir com dados completos
        simulacao (dict|None)      preenchido em simular_preco
        informacoes_faltantes (list[str])
        resposta (str)             texto pronto para exibir no chat (markdown simples)
        erro (str|None)            detalhe técnico para log, não exibir ao usuário

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
            "quanto gasta com materiais e quanto quer ganhar por hora."
        )
        resultado["erro"] = "mensagem_vazia"
        return resultado

    # Extração (IA)
    try:
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
        if intencao == "simular_preco":
            novo_preco = dados.get("novo_preco_simulado")
            novos_clientes = dados.get("novo_quantidade_clientes") if _modo(novo_estado) == "recorrente" else None
            faltantes_sim = [c for c in faltantes if c != "preco_cobrado"]
            resultado["informacoes_faltantes"] = faltantes_sim
            if not novo_preco and not novos_clientes:
                resultado["resposta"] = "Qual valor você quer testar? Ex: 'E se eu cobrar R$ 90?'"
            elif faltantes_sim:
                resultado["resposta"] = _texto_faltantes(faltantes_sim, intencao)
            else:
                simulacao = simular_novo_preco(novo_estado, novo_preco, novos_clientes)
                resultado["simulacao"] = simulacao
                resultado["resposta"] = formatar_explicacao_humana(simulacao, intencao)
        else:
            resultado["informacoes_faltantes"] = faltantes
            if faltantes:
                resultado["resposta"] = _texto_faltantes(faltantes, intencao)
            else:
                metricas = _metricas_do_estado(novo_estado)
                resultado["metricas"] = metricas
                resultado["resposta"] = formatar_explicacao_humana(
                    {**metricas, "nome_servico": novo_estado.get("nome_servico")}, intencao
                )
    except (ValueError, ArithmeticError, TypeError, KeyError, AttributeError) as exc:
        logger.warning("Dados inválidos na sessão: %s", exc)
        resultado["intencao"] = None
        resultado["erro"] = str(exc)
        resultado["resposta"] = (
            "Algum valor ficou estranho (negativo ou taxa acima de 100%). "
            "Pode conferir e me mandar de novo?"
        )
        return resultado

    resultado["estado"] = novo_estado
    resultado["sucesso"] = True
    return resultado