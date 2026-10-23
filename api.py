
import asyncio

from fastapi import FastAPI
from pydantic import BaseModel
from engine import processar_mensagem_usuario
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.mount(
    "/static",
    StaticFiles(
        directory=Path(__file__).resolve().parent / "frontend"
    ),
    name="static"
)



class Entrada(BaseModel):
    mensagem: str
    estado: dict | None = None


@app.post("/chat")
async def chat(entrada: Entrada):
    resultado = await asyncio.to_thread(
        processar_mensagem_usuario,
        entrada.mensagem,
        entrada.estado
    )

    if not resultado["sucesso"]:
        print("Erro do motor:", resultado.get("erro"))
    

    return {
        "resposta": resultado["resposta"],
        "estado": resultado["estado"],
        "sucesso": resultado["sucesso"],
        "intencao": resultado["intencao"],
        "metricas": resultado["metricas"],
        "simulacao": resultado["simulacao"],
        "informacoes_faltantes": resultado["informacoes_faltantes"]
    }

@app.get("/")
def abrir_chat():
    caminho = Path(__file__).resolve().parent / "frontend" / "index.html"
    return FileResponse(caminho)


