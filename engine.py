# mensagem do usuário
# extrair_dados_com_ia()        (LLM: SÓ converte texto em JSON)
# normalizar_extracao()         (valida e corrige tipos vindos da LLM)
# atualizar_estado()            (mescla com a sessão)
# calcular_metricas_financeiras() / simular_novo_preco()   (matemática)
# formatar_explicacao_humana()  (texto final para o chat)


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
  "informacoes_faltantes": [strings]
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
)
CAMPOS_TEXTO = ("categoria_servico", "nome_servico", "campo_corrigido")

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
}

# Sem estes três não dá para calcular. Materiais/deslocamento/outros assumem 0
# e a taxa assume TAXA_PADRAO.
CAMPOS_OBRIGATORIOS = {
    "preco_cobrado": "quanto você cobra pelo serviço (ex: 'cobro R$ 80')",
    "tempo_horas": "quanto tempo leva para fazer (ex: 'demoro 2 horas')",
    "valor_hora_desejado": "quanto quer ganhar por hora de trabalho (ex: 'quero R$ 20 por hora')",
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

    faltantes = bruto.get("informacoes_faltantes") or []
    dados["informacoes_faltantes"] = [str(f) for f in faltantes if f] if isinstance(faltantes, list) else []

    intencao = bruto.get("intencao")
    if intencao not in INTENCOES_VALIDAS:
        intencao = "simular_preco" if dados["novo_preco_simulado"] is not None else "informar_dados"
    # Proteção: LLM marcou simulação mas colocou o valor no campo errado.
    if intencao == "simular_preco" and dados["novo_preco_simulado"] is None and dados["preco_cobrado"] is not None:
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
) -> dict:
    """
    Calcula todos os custos, sobra real, margem e preço mínimo recomendável.

        C_d   = soma(materiais) + deslocamento + outros_gastos
        C_mo  = tempo_horas * valor_hora_desejado
        C_t   = C_d + C_mo
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
    }
    valores_materiais = [_dec(m.get("valor")) for m in (materiais or [])]
    if any(v < 0 for v in list(entradas.values()) + valores_materiais):
        raise ValueError("Valores não podem ser negativos.")

    total_materiais = sum(valores_materiais, Decimal("0"))
    custo_direto = total_materiais + entradas["deslocamento"] + entradas["outros_gastos"]
    custo_mao_obra = entradas["tempo_horas"] * entradas["valor_hora_desejado"]
    custo_total = custo_direto + custo_mao_obra

    preco = entradas["preco_cobrado"]
    valor_taxa = preco * taxa
    valor_liquido = preco - valor_taxa  # = preco * (1 - taxa)
    sobra = valor_liquido - custo_total
    margem = (sobra / preco * 100) if preco > 0 else None
    preco_minimo = custo_total / (1 - taxa)

    return {
        "total_materiais": _arred(total_materiais),
        "deslocamento": _arred(entradas["deslocamento"]),
        "outros_gastos": _arred(entradas["outros_gastos"]),
        "custo_direto": _arred(custo_direto),
        "tempo_horas": float(entradas["tempo_horas"]),
        "valor_hora_desejado": _arred(entradas["valor_hora_desejado"]),
        "custo_mao_obra": _arred(custo_mao_obra),
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


def _metricas_do_estado(estado: dict, preco: Optional[float] = None) -> dict:
    """Atalho: calcula as métricas a partir de um estado de sessão."""
    return calcular_metricas_financeiras(
        materiais=estado.get("materiais") or [],
        deslocamento=estado.get("deslocamento"),
        outros_gastos=estado.get("outros_gastos"),
        tempo_horas=estado.get("tempo_horas"),
        valor_hora_desejado=estado.get("valor_hora_desejado"),
        preco_cobrado=estado.get("preco_cobrado") if preco is None else preco,
        taxa_maquininha_pct=estado.get("taxa_maquininha_pct"),
    )


def simular_novo_preco(estado_atual: dict, novo_preco: float) -> dict:
    """
    Recalcula as métricas financeiras mantendo os custos do estado atual,
    porém alterando o preço cobrado para o 'novo_preco'.
    Retorna a diferença de sobra líquida entre o preço original e o simulado.
    Se o estado não tiver preço original, "metricas_original" e
    "diferenca_sobra" vêm como None.
    """
    if novo_preco is None or novo_preco <= 0:
        raise ValueError("novo_preco deve ser maior que zero.")

    simuladas = _metricas_do_estado(estado_atual, preco=novo_preco)
    original = None
    diferenca = None
    if estado_atual.get("preco_cobrado") is not None:
        original = _metricas_do_estado(estado_atual)
        diferenca = _arred(_dec(simuladas["sobra_liquida"]) - _dec(original["sobra_liquida"]))

    return {
        "preco_original": original["preco_cobrado"] if original else None,
        "preco_simulado": simuladas["preco_cobrado"],
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
    intencao = dados_novos.get("intencao")

    campos_simples = [c for c in ESTADO_VAZIO if c != "materiais"]
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

    return estado


def campos_faltantes(estado: dict) -> list[str]:
    """Campos obrigatórios ainda não informados na sessão."""
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


def formatar_explicacao_humana(metricas: dict, intencao: str) -> str:
    """
    Converte os números calculados em texto simples.
      - informar_dados / corrigir_dado: recebe o dict de calcular_metricas_financeiras
        (opcionalmente com "nome_servico").
      - simular_preco: recebe o dict de simular_novo_preco.
    """
    if intencao == "simular_preco":
        return _texto_simulacao(metricas)
    resumo = _texto_resumo(metricas)
    if intencao == "corrigir_dado":
        return "✅ Pronto, dado corrigido! Veja como ficou:\n\n" + resumo
    return resumo


def _texto_faltantes(faltantes: list[str], intencao: str) -> str:
    inicio = "✅ Anotado!" if intencao == "corrigir_dado" else "Anotei!"
    itens = "\n".join(f"- {CAMPOS_OBRIGATORIOS[c]}" for c in faltantes)
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
            faltantes_sim = [c for c in faltantes if c != "preco_cobrado"]
            resultado["informacoes_faltantes"] = faltantes_sim
            if not novo_preco:
                resultado["resposta"] = "Qual valor você quer testar? Ex: 'E se eu cobrar R$ 90?'"
            elif faltantes_sim:
                resultado["resposta"] = _texto_faltantes(faltantes_sim, intencao)
            else:
                simulacao = simular_novo_preco(novo_estado, novo_preco)
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