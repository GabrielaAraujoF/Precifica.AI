# Preço Justo — Módulo Back-end & IA

Motor de precificação para trabalhadores autônomos (beleza, costura, artesanato, manutenção etc.). O usuário escreve em linguagem natural ("Cobro 60 reais na unha em gel, demoro 2 horas...") e o módulo devolve custos, sobra real, margem e preço mínimo, além de um texto pronto para exibir no chat.

**Princípio central:** a IA (Google Gemini) **só extrai dados** do texto. **Toda a matemática é feita em Python**, de forma determinística, com `Decimal` e arredondamento em centavos. A IA nunca faz contas.

---

## Sumário

1. [Instalação e configuração](#1-instalação-e-configuração)
2. [Como o código funciona](#2-como-o-código-funciona)
3. [Fórmulas financeiras](#3-fórmulas-financeiras)
4. [Saída da função principal](#4-saída-da-função-principal)
5. [Como tratar no front-end](#5-como-tratar-no-front-end)
6. [Resiliência da chamada à IA](#6-resiliência-da-chamada-à-ia)
7. [Testes](#7-testes)
8. [Limitações conhecidas](#8-limitações-conhecidas)

---

## 1. Instalação e configuração

```bash
pip install -r requirements.txt
```

Crie o arquivo `.env` a partir do `.env.example` e coloque sua chave (gratuita em https://aistudio.google.com/apikey):

```
GEMINI_API_KEY=sua_chave_aqui
```

> **Nunca** faça commit do `.env`. Ele já está no `.gitignore`.

Todas as variáveis opcionais (modelo, timeouts, tentativas) estão documentadas no `.env.example`. Os padrões do código já são os recomendados:

| Variável | Padrão | Para que serve |
|---|---|---|
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Modelo principal (15 req/min e 500 req/dia no plano gratuito) |
| `GEMINI_MODELOS_RESERVA` | `gemini-3.8-flash` | Usados em ordem se o principal falhar |
| `GEMINI_TIMEOUT` | `15` | Timeout de cada chamada (s) |
| `GEMINI_PRAZO_TOTAL` | `25` | Tempo máximo somando todas as tentativas (s) |
| `GEMINI_TENTATIVAS` | `2` | Tentativas por modelo em erros 500/502/503 |
| `GEMINI_THINKING` | `low` | Nível de raciocínio nos modelos Gemini 3+ |

### Estrutura

```
├── engine.py          # integração com a IA + motor financeiro + respostas
├── test_engine.py     # 46 testes locais + 3 de integração (opcionais)
├── requirements.txt   # google-genai, python-dotenv
├── .env.example       # modelo do .env
└── .gitignore
```

---

## 2. Como o código funciona

```
mensagem do usuário
   │
   ▼
extrair_dados_com_ia()          IA: texto → JSON estruturado (nunca calcula)
   │
   ▼
normalizar_extracao()           Python: valida tipos, corrige erros comuns da IA
   │
   ▼
atualizar_estado()              Python: mescla com os dados já informados na sessão
   │
   ├─ informar_dados / corrigir_dado ─► calcular_metricas_financeiras()
   └─ simular_preco ──────────────────► simular_novo_preco()
   │
   ▼
formatar_explicacao_humana()    Python: texto simples para o chat
```

### As três intenções

| Intenção | Exemplo de mensagem | O que acontece |
|---|---|---|
| `informar_dados` | "Cobro 60 na unha em gel, demoro 2h, gasto 15 de gel" | Adiciona os dados à sessão e calcula (se já houver o suficiente) |
| `corrigir_dado` | "Na verdade o tecido custou 35" | Atualiza **só** o item corrigido, mantendo o resto |
| `simular_preco` | "E se eu cobrar R$ 90?" | Compara o preço atual com o simulado, **sem** alterar o preço salvo |

### Funções públicas

| Função | Descrição |
|---|---|
| `processar_mensagem_usuario(mensagem, estado_atual=None)` | **Ponto de entrada do front-end.** Faz o fluxo inteiro. |
| `calcular_metricas_financeiras(...)` | Função pura com todas as fórmulas. |
| `simular_novo_preco(estado, novo_preco)` | Recalcula com outro preço e devolve a diferença de sobra. |
| `atualizar_estado(estado_anterior, dados_novos)` | Mescla os dados novos na sessão (nunca altera o dict recebido). |
| `formatar_explicacao_humana(metricas, intencao)` | Converte números em texto para o usuário. |

### Regras de negócio importantes

- **Dados obrigatórios para calcular:** `preco_cobrado`, `tempo_horas` e `valor_hora_desejado`. Se faltar algum, o módulo **pergunta** em vez de calcular com zero.
- **Dados opcionais:** materiais, deslocamento e outros gastos assumem `0`. A taxa da maquininha assume **5%**.
- **Campos `null` vindos da IA nunca apagam** dados já existentes na sessão.
- **Materiais com o mesmo nome são substituídos**, e os novos são adicionados. Na correção, a busca é aproximada: "tecido" encontra "Tecido de algodão".
- **"Sobra" já desconta a mão de obra.** Sobra negativa não significa necessariamente prejuízo em dinheiro. Significa que o usuário está ganhando **menos do que a hora desejada**. O texto da resposta explica isso.

---

## 3. Fórmulas financeiras

| Métrica | Fórmula |
|---|---|
| Custo direto (`custo_direto`) | Σ materiais + deslocamento + outros gastos |
| Mão de obra (`custo_mao_obra`) | tempo_horas × valor_hora_desejado |
| Custo total (`custo_total`) | custo direto + mão de obra |
| Valor líquido (`valor_liquido`) | preço × (1 − taxa) |
| Sobra (`sobra_liquida`) | valor líquido − custo total |
| Margem (`margem_pct`) | sobra ÷ preço × 100 |
| Preço mínimo (`preco_minimo`) | custo total ÷ (1 − taxa) |

`taxa_maquininha_pct` é sempre uma **fração** (`0.05` = 5%). Se a IA devolver `5`, o código converte para `0.05`.

---

## 4. Saída da função principal

`processar_mensagem_usuario` **sempre** devolve um dicionário com as mesmas 9 chaves, inclusive em caso de erro. Todos os valores são serializáveis em JSON.

| Chave | Tipo | Descrição | Mostrar ao usuário? |
|---|---|---|---|
| `sucesso` | `bool` | `False` apenas se a IA falhou ou um valor era inválido | Não (use para a UI) |
| `intencao` | `str \| null` | `informar_dados`, `corrigir_dado`, `simular_preco` (ou `null` se falhou) | Não |
| `dados_extraidos` | `dict \| null` | JSON normalizado vindo da IA | **Não** (debug) |
| `estado` | `dict` | Dados acumulados da sessão | **Não**, mas **guarde** |
| `metricas` | `dict \| null` | Preenchido em `informar_dados`/`corrigir_dado` quando há dados suficientes | Opcional (cards, gráficos) |
| `simulacao` | `dict \| null` | Preenchido em `simular_preco` | Opcional (comparativo) |
| `informacoes_faltantes` | `list[str]` | Campos obrigatórios que ainda faltam | Opcional (dicas) |
| `resposta` | `str` | Texto pronto em **markdown simples** | **Sim, sempre** |
| `erro` | `str \| null` | Detalhe técnico | **Não** (só log) |

### Exemplo 1: cálculo completo (`informar_dados`)

Mensagem: *"Cobro 60 reais para fazer unha em gel. Demoro 2 horas. Gasto 15 de gel e lixas, e 5 de transporte. Quero ganhar 15 reais por hora."*

```json
{
  "sucesso": true,
  "intencao": "informar_dados",
  "estado": {
    "categoria_servico": "Beleza",
    "nome_servico": "Unha em gel",
    "preco_cobrado": 60.0,
    "tempo_horas": 2.0,
    "valor_hora_desejado": 15.0,
    "materiais": [{ "item": "gel e lixas", "valor": 15.0 }],
    "deslocamento": 5.0,
    "outros_gastos": null,
    "taxa_maquininha_pct": null
  },
  "metricas": {
    "total_materiais": 15.0,
    "deslocamento": 5.0,
    "outros_gastos": 0.0,
    "custo_direto": 20.0,
    "tempo_horas": 2.0,
    "valor_hora_desejado": 15.0,
    "custo_mao_obra": 30.0,
    "custo_total": 50.0,
    "preco_cobrado": 60.0,
    "taxa_maquininha_pct": 0.05,
    "valor_taxa": 3.0,
    "valor_liquido": 57.0,
    "sobra_liquida": 7.0,
    "margem_pct": 11.67,
    "preco_minimo": 52.63,
    "cobre_custos": true
  },
  "simulacao": null,
  "informacoes_faltantes": [],
  "resposta": "Resumo do seu serviço (**Unha em gel**):\n- **Custos diretos ...",
  "erro": null
}
```

O campo `resposta`, renderizado:

> Resumo do seu serviço (**Unha em gel**):
> - **Custos diretos (materiais + deslocamento + outros):** R$ 20,00
> - **Sua mão de obra (2h a R$ 15/h):** R$ 30,00
> - **Custo total do serviço:** R$ 50,00
>
> Para o preço cobrado de **R$ 60,00** (considerando 5% de taxa do cartão):
> - Você recebe líquido: **R$ 57,00**
> - **Sobra real para o seu negócio:** R$ 7,00 (margem de 11,7%)
>
> 💡 **Preço mínimo recomendado:** R$ 52,63 (para cobrir custos e mão de obra sem ter prejuízo).
>
> Quer testar outro valor? Pergunte: 'E se eu cobrar R$ 70?'

### Exemplo 2: simulação (`simular_preco`)

Mensagem: *"E se eu cobrar R$ 90 em vez de R$ 60?"* (sessão com custo total de R$ 40)

```json
{
  "sucesso": true,
  "intencao": "simular_preco",
  "metricas": null,
  "simulacao": {
    "preco_original": 60.0,
    "preco_simulado": 90.0,
    "sobra_original": 17.0,
    "sobra_simulada": 45.5,
    "diferenca_sobra": 28.5,
    "metricas_original": { "...": "mesmas chaves de metricas" },
    "metricas_simuladas": { "...": "mesmas chaves de metricas" }
  },
  "resposta": "🔎 Simulação: cobrando **R$ 90,00** em vez de **R$ 60,00**\n..."
}
```

> 🔎 Simulação: cobrando **R$ 90,00** em vez de **R$ 60,00**
> - Você recebe líquido: **R$ 85,50** (antes R$ 57,00)
> - Sobra real: **R$ 45,50** (antes R$ 17,00) → R$ 28,50 a mais por serviço
> - Margem: 50,6% (antes 28,3%)
>
> 💡 Seus custos continuam os mesmos: o preço mínimo segue em R$ 42,11.

Se a sessão ainda não tiver preço original, `preco_original`, `sobra_original`, `diferenca_sobra` e `metricas_original` vêm como `null`.

### Exemplo 3: dados incompletos

Mensagem: *"Cobro 80 na escova"*

```json
{
  "sucesso": true,
  "metricas": null,
  "informacoes_faltantes": ["tempo_horas", "valor_hora_desejado"],
  "resposta": "Anotei! Para calcular seu preço justo, ainda preciso saber:\n- quanto tempo leva para fazer (ex: 'demoro 2 horas')\n- quanto quer ganhar por hora de trabalho (ex: 'quero R$ 20 por hora')"
}
```

### Exemplo 4: falha da IA

```json
{
  "sucesso": false,
  "intencao": null,
  "estado": { "...": "estado ANTERIOR, intacto" },
  "resposta": "Não consegui entender sua mensagem agora. Pode tentar de novo, de um jeito um pouco diferente?",
  "erro": "Nenhum modelo respondeu a tempo. Tentativas: ..."
}
```

Mensagem vazia retorna `sucesso: false` e `erro: "mensagem_vazia"`, **sem chamar a IA**.

---

## 5. Como tratar no front-end

### Regra de ouro

```
1. Envie (mensagem, estado)
2. Mostre `resposta` no chat
3. Guarde `estado` e devolva na próxima mensagem
```

O módulo **não guarda nada entre chamadas**. Quem mantém a conversa é quem chama. O `estado` é um JSON simples e pode ficar:

- **no servidor**, associado à sessão do usuário (recomendado), ou
- **no cliente**, enviado e recebido a cada requisição.

Para começar um novo serviço do zero, envie `estado = null`.

### Exemplo de API (FastAPI)

> Não está no `requirements.txt`: `pip install fastapi uvicorn`

```python
# api.py
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from engine import processar_mensagem_usuario

app = FastAPI()

class Entrada(BaseModel):
    mensagem: str
    estado: dict | None = None

@app.post("/chat")
async def chat(entrada: Entrada):
    # A função é síncrona e pode levar alguns segundos (chamada à IA):
    # rodar em thread evita travar o servidor para os outros usuários.
    r = await asyncio.to_thread(processar_mensagem_usuario, entrada.mensagem, entrada.estado)
    return {
        "resposta": r["resposta"],
        "estado": r["estado"],
        "sucesso": r["sucesso"],
        "intencao": r["intencao"],
        "metricas": r["metricas"],
        "simulacao": r["simulacao"],
        "informacoes_faltantes": r["informacoes_faltantes"],
        # "erro" e "dados_extraidos" ficam só no log do servidor
    }
```

Rodar: `uvicorn api:app --reload`

### Exemplo de cliente (JavaScript)

```javascript
let estado = null; // estado da conversa atual

async function enviar(mensagem) {
  mostrarCarregando(true);           // a IA pode levar de 2 a 25 segundos
  bloquearBotaoEnviar(true);         // evita mensagens em paralelo
  try {
    const resp = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mensagem, estado }),
    });
    const r = await resp.json();
    estado = r.estado;               // sempre atualize, mesmo com sucesso=false
    adicionarBalao(renderizarMarkdown(r.resposta));
    if (r.metricas) atualizarCards(r.metricas);
    if (r.simulacao) mostrarComparativo(r.simulacao);
  } catch (e) {
    adicionarBalao("Sem conexão. Tente de novo em instantes.");
  } finally {
    mostrarCarregando(false);
    bloquearBotaoEnviar(false);
  }
}
```

### Checklist de UI

**Obrigatório**
- [ ] **Renderizar `resposta` como markdown.** O texto usa `**negrito**`, listas com `- ` e quebras `\n`. Se não usar uma lib de markdown, troque pelo menos `\n` por `<br>` e `**x**` por `<strong>x</strong>`.
- [ ] **Indicador de carregamento.** A chamada leva normalmente de 2 a 5 s. No pior caso, com o Google instável, até ~25 s.
- [ ] **Bloquear envio durante o processamento.** Duas mensagens simultâneas na mesma sessão podem sobrescrever o estado.
- [ ] **Sempre substituir o `estado` local pelo retornado**, inclusive com `sucesso: false` (nesse caso vem o anterior, intacto).
- [ ] **Nunca exibir `erro` nem `dados_extraidos`** ao usuário. São só para log e depuração.

**Recomendado**
- [ ] **Botão "Novo serviço"**, que faz `estado = null`.
- [ ] **Botões rápidos de simulação** ("E se eu cobrar R$ 70?") após um cálculo. Basta enviar o texto como mensagem.
- [ ] **Destaque visual quando `metricas.cobre_custos === false`** (ex.: card vermelho). O texto já alerta, mas o visual ajuda.
- [ ] **Dicas a partir de `informacoes_faltantes`**, por exemplo chips clicáveis "⏱️ Informar tempo" e "💰 Informar valor da hora".

### Usando os números diretamente

Se quiser montar cards ou gráficos em vez de usar só o texto:

- Valores monetários já vêm **arredondados em 2 casas** (float). Formate no front:
  ```javascript
  const brl = new Intl.NumberFormat("pt-BR", { style: "currency", currency: "BRL" });
  brl.format(r.metricas.sobra_liquida); // "R$ 7,00"
  ```
- `taxa_maquininha_pct` é **fração**: exiba `taxa * 100` + `%`.
- `margem_pct` já está em **porcentagem** (`11.67` = 11,67%) e **pode ser `null`** quando o preço é zero.
- `sobra_liquida` e `diferenca_sobra` **podem ser negativos**.
- **Não refaça as contas no front.** Use sempre os valores do back-end, que são a fonte única da verdade.

### Limite de uso da IA (plano gratuito)

Cada mensagem do usuário consome **1 requisição** ao Gemini. Com os padrões atuais, o modelo principal permite cerca de **15 mensagens por minuto** e **500 por dia** para **toda a aplicação** (o limite é da chave, não por usuário). Se o limite for atingido, o módulo tenta o modelo reserva automaticamente e, se nada responder, devolve a mensagem amigável de erro.

---

## 6. Resiliência da chamada à IA

| Erro do Google | Significado | Comportamento |
|---|---|---|
| 500 / 502 / 503 | Sobrecarga momentânea | Espera (backoff) e tenta de novo no mesmo modelo |
| 404 | Modelo indisponível para a conta | Pula direto para o próximo modelo |
| 429 | Cota do modelo esgotada | Pula direto para o próximo modelo |
| 504 | Modelo lento demais | Pula direto para o próximo modelo |
| 400 / 401 / 403 | Chave ou requisição inválida | Falha na hora |
| JSON inválido | IA respondeu fora do formato | Tenta de novo |

Tudo isso respeita o `GEMINI_PRAZO_TOTAL` (25 s). Passado esse tempo, o usuário recebe a resposta amigável em vez de ficar esperando.

**Sobre a temperatura:** nos modelos Gemini 1.x/2.x usamos `temperature=0.0`. Nos modelos Gemini 3+ mantemos a temperatura padrão e raciocínio `low`, porque o Google recomenda isso e valores baixos causaram lentidão (erro 504) nos testes. O determinismo dos **números** não depende disso: toda conta é feita em Python e a saída da IA é validada antes de qualquer cálculo.

---

## 7. Testes

```bash
# Testes locais (não usam a API nem consomem cota)
python -m unittest -v test_engine
```

Os testes locais substituem a IA por um extrator falso e cobrem:

- os 3 cenários obrigatórios (manicure, costura, artesanato);
- o motor financeiro;
- a normalização da saída da IA;
- a mescla de estado;
- a robustez do contrato;
- a resiliência a erros da API.

```powershell
# Testes de integração com o Gemini real (gastam 3 requisições da cota)
$env:RUN_INTEGRATION = "1"
python -m unittest -v test_engine.TestIntegracaoGemini
```

Diagnóstico rápido de latência e de qual modelo respondeu:

```powershell
python -c "import logging,time,engine; logging.basicConfig(level=logging.WARNING); t=time.time(); r=engine.processar_mensagem_usuario('E se eu cobrar 90?', {'preco_cobrado':60,'tempo_horas':2,'valor_hora_desejado':10,'materiais':[]}); print(round(time.time()-t,1),'s |', r['sucesso'], r['intencao'])"
```

---

## 8. Limitações conhecidas

- **Um serviço por sessão.** O estado guarda um serviço de cada vez. Para precificar outro, inicie um novo estado.
- **Remover itens não é suportado.** "Tira o transporte" não zera o deslocamento. O usuário pode corrigir para zero ("o transporte foi 0").
- **A extração depende da IA.** Frases muito ambíguas podem ser mal interpretadas. Os números calculados são sempre corretos para os dados extraídos, mas vale exibir o resumo para o usuário conferir.
- **Sem persistência.** O módulo não salva nada. Guardar histórico ou sessões é responsabilidade da camada que o chama.