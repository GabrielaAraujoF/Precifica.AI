
const formulario = document.getElementById("form-chat");
const campoMensagem = document.getElementById("campo-mensagem");
const mensagens = document.getElementById("mensagens");
const botaoEnviar = document.getElementById("btn-enviar");
const botaoNovoServico = document.getElementById("novo-servico");

// Guarda as informações da conversa atual
let estado = null;
let versaoConversa = 0;
let historico = [];

const CHAVE_CONVERSA = "precificaAI_conversa";
const MENSAGEM_ERRO_COMUNICACAO =
    "Não foi possível processar sua mensagem. Tente novamente.";
const mensagemInicial = mensagens.innerHTML;

// Salva a conversa no navegador
function salvarConversa() {
    sessionStorage.setItem(
        CHAVE_CONVERSA,
        JSON.stringify({ estado, historico })
    );
}

// Adiciona negrito Markdown sem interpretar HTML vindo do backend
function adicionarTextoFormatado(elemento, texto) {
    const partes = texto.split(/(\*\*[^\n*]+\*\*)/g);

    partes.forEach(parte => {
        const negrito = parte.startsWith("**") && parte.endsWith("**");

        if (negrito) {
            const strong = document.createElement("strong");
            strong.textContent = parte.slice(2, -2);
            elemento.appendChild(strong);
        } else if (parte) {
            elemento.appendChild(document.createTextNode(parte));
        }
    });
}

// Renderiza somente o subconjunto de Markdown usado nas respostas do chatbot
function renderizarMarkdownSeguro(elemento, texto) {
    const linhas = texto.replace(/\r\n?/g, "\n").split("\n");
    let paragrafo = null;
    let lista = null;

    linhas.forEach(linha => {
        const itemLista = linha.match(/^\s*-\s+(.+)$/);

        if (itemLista) {
            paragrafo = null;

            if (!lista) {
                lista = document.createElement("ul");
                elemento.appendChild(lista);
            }

            const item = document.createElement("li");
            adicionarTextoFormatado(item, itemLista[1]);
            lista.appendChild(item);
            return;
        }

        lista = null;

        if (!linha.trim()) {
            paragrafo = null;
            return;
        }

        if (!paragrafo) {
            paragrafo = document.createElement("p");
            elemento.appendChild(paragrafo);
        } else {
            paragrafo.appendChild(document.createElement("br"));
        }

        adicionarTextoFormatado(paragrafo, linha);
    });
}

// Adiciona mensagens na interface
function adicionarMensagem(texto, tipo, registrar = true) {
    const balao = document.createElement("div");

    balao.classList.add("mensagem", tipo);

    if (tipo === "bot") {
        renderizarMarkdownSeguro(balao, texto);
    } else {
        balao.textContent = texto;
    }

    mensagens.appendChild(balao);
    mensagens.scrollTop = mensagens.scrollHeight;

    if (registrar) {
        historico.push({ texto, tipo });
        salvarConversa();
    }

    return balao;
}

// Recupera a conversa ao atualizar a página
function restaurarConversa() {
    const salvo = sessionStorage.getItem(CHAVE_CONVERSA);

    if (!salvo) return;

    try {
        const dados = JSON.parse(salvo);

        if (!dados || !Array.isArray(dados.historico)) return;

        estado = dados.estado ?? null;

        historico = dados.historico.filter(
            item =>
                item &&
                typeof item.texto === "string" &&
                ["usuario", "bot"].includes(item.tipo)
        );

        mensagens.innerHTML = mensagemInicial;

        historico.forEach(item => {
            adicionarMensagem(item.texto, item.tipo, false);
        });

    } catch (erro) {
        console.error("Erro ao restaurar conversa:", erro);
        sessionStorage.removeItem(CHAVE_CONVERSA);
    }
}

restaurarConversa();

// Envia uma mensagem ao FastAPI
async function enviarMensagem(mensagem) {
    const versaoAtual = versaoConversa;

    botaoEnviar.disabled = true;
    campoMensagem.disabled = true;

    adicionarMensagem(mensagem, "usuario");

    const carregando = adicionarMensagem(
        "Analisando sua mensagem...",
        "bot",
        false
    );

    try {
        const resposta = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                mensagem: mensagem,
                estado: estado
            })
        });

        // Ignora respostas de conversas reiniciadas
        if (versaoAtual !== versaoConversa) return;

        if (!resposta.ok) {
            throw new Error("Erro na requisição");
        }

        const resultado = await resposta.json();

        // Confere novamente se a conversa ainda é a mesma
        if (versaoAtual !== versaoConversa) return;

        const resultadoValido =
            resultado &&
            typeof resultado === "object" &&
            typeof resultado.sucesso === "boolean" &&
            typeof resultado.resposta === "string" &&
            Object.prototype.hasOwnProperty.call(resultado, "estado");

        if (!resultadoValido) {
            throw new Error("Resposta inválida");
        }

        const mensagemResposta = resultado.resposta;

        if (!mensagemResposta.trim() && resultado.sucesso) {
            throw new Error("Resposta vazia");
        }

        // Em sucesso:false, o backend devolve uma mensagem amigável e o
        // estado anterior intacto. Ambos devem ser preservados no frontend.
        estado = resultado.estado;
        salvarConversa();

        carregando.remove();

        adicionarMensagem(
            mensagemResposta.trim()
                ? mensagemResposta
                : MENSAGEM_ERRO_COMUNICACAO,
            "bot"
        );

    } catch (erro) {
        // Não mostra erros de atendimentos antigos
        if (versaoAtual !== versaoConversa) return;

        carregando.remove();

        adicionarMensagem(MENSAGEM_ERRO_COMUNICACAO, "bot");

        console.error("Erro no chat:", erro);

    } finally {
        // Evita que requisições antigas alterem a nova conversa
        if (versaoAtual === versaoConversa) {
            botaoEnviar.disabled = false;
            campoMensagem.disabled = false;
            campoMensagem.focus();
        }
    }
}

// Captura o envio do formulário
formulario.addEventListener("submit", function(evento) {
    evento.preventDefault();

    const mensagem = campoMensagem.value.trim();

    if (!mensagem || botaoEnviar.disabled) {
        return;
    }

    campoMensagem.value = "";

    enviarMensagem(mensagem);
});

// Reinicia o atendimento
botaoNovoServico.addEventListener("click", function() {
    // Invalida as respostas da conversa anterior
    versaoConversa++;

    // Limpa os dados anteriores
    estado = null;
    historico = [];

    sessionStorage.removeItem(CHAVE_CONVERSA);

    // Restaura a mensagem inicial
    mensagens.innerHTML = mensagemInicial;

    // Libera o campo de digitação
    botaoEnviar.disabled = false;
    campoMensagem.disabled = false;
    campoMensagem.value = "";
    campoMensagem.focus();
});
