import os
import time
import json
import asyncio
from functools import partial
from typing import List, Union
from starlette.websockets import WebSocketDisconnect

from Robot.AwesomeParser import AwesomeParser
from Robot.DailyParser import DailyParser
from Robot.base import ChatBase
from utils.logger import setup_base_logger, get_logger
from utils.banner import banner
from utils import gen_token

from fastapi import FastAPI, Request, WebSocket, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


# 初始化
try:
    del os.environ["NVIDIA_API_KEY"]
except:
    pass
os.environ["NVIDIA_API_KEY"] = "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"
banner()
setup_base_logger()
_logger = get_logger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# token: parser映射
records = {}


def load_daily_paper(token):
    _, daily_parser = records[token]
    _, outlook, paper_info = daily_parser.load_paper("20240707")
    info = {}
    info["details"] = []
    for title, author, summary, link in paper_info:
        temp = {}
        temp["title"] = title
        temp["author"] = author
        temp["summary"] = summary
        temp["link"] = link.split(";")[0]
        info["details"].append(temp)
    info["outlook"] = outlook
    return info


def fake_llm(user_msg):
    l = ["hello", ", I am", " your", " AI", " assistant. ", " Can I", " help", " you ?", "hello", ", I am", " your", " AI", " assistant. ", " Can I", " help", " you ?"]
    for msg in l:
        yield msg


@app.get("/check")
async def read_token(apikey: Union[str, None] = None):
    res = ChatBase.check_apikey(apikey)
    _logger.info(f"Check result: {res}")
    if res:
        token = gen_token(apikey)
        records[token] = (AwesomeParser(apikey), DailyParser(apikey))
        return {"valid": True, "token": token}
    else:
        return {"valid": False}


@app.get("/index", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(
        context={
            "request": request,
            "proj_name": "AwesomeParser",
            "init": True,
        },
        name="index.html.jinja"
    )


@app.get("/index/{token}", response_class=HTMLResponse)
async def read_item(request: Request, token: str = ""):
    if token not in records:
        return '{"status": "error", "msg": "Invalid token"}'
    else:
        return templates.TemplateResponse(
            context={
                "request": request,
                "proj_name": "AwesomeParser",
                "init": False,
                "token": token
            },
            name="index.html.jinja"
        )


@app.get("/chat/{id}/{token}", response_class=HTMLResponse)
async def chat(request: Request, token: str, id: int):
    if token not in records:
        return '{"status": "error", "msg": "Invalid token"}'
    return templates.TemplateResponse(
        name="chat.html.jinja", context={"request": request, "id": id, "token": token}
    )


@app.post("/{token}/upload")
async def create_upload_files(files: List[UploadFile], token: str):
    if token not in records:
        return {"status": "error", "msg": "Invalid token"}
    if len(files) == 0:
        return {"error": "No files uploaded"}
    awe_parser, _ = records[token]
    contents = ""
    for file in files:
        temp = await file.read()
        contents += temp.decode() + "\n\n"
        # print(contents)
    _, _ = awe_parser.load_paper(contents)
    return {"status": "success"}

# ------------ 即时传 ------------
@app.websocket("/{token}/load")
async def websocket_endpoint(websocket: WebSocket, token: str):
    await websocket.accept()
    if token not in records:
        await websocket.send_json({"status": "error", "msg": "Invalid token"})
    info = load_daily_paper(token)
    await websocket.send_json(info)
    await asyncio.sleep(0.1)
    await websocket.close()

@app.websocket("/{token}/daily_paper_prologue")    # 每日论文开场白
async def websocket_endpoint(websocket: WebSocket, token: str):
    await websocket.accept()
    if token not in records:
        await websocket.send_json({"status": "error", "msg": "Invalid token"})
    _, daily_parser = records[token]
    _logger.info(daily_parser.papers_outlook)
    await websocket.send_text(daily_parser.papers_outlook)
    await asyncio.sleep(0.1)
    await websocket.close()

@app.websocket("/{token}/awe_parser_prologue")    # Awesome开场白
async def websocket_endpoint(websocket: WebSocket, token: str):
    if token not in records:
        await websocket.send_json({"status": "error", "msg": "Invalid token"})
    await websocket.accept()
    awe_parser, _ = records[token]
    _logger.info(awe_parser.papers_outlook)
    await websocket.send_text(awe_parser.papers_outlook)
    await asyncio.sleep(0.1)
    await websocket.close()

# ---------- 持久会话 -------------
@app.websocket("/{token}/{item_id}/chat_ws")
async def websocket_endpoint(websocket: WebSocket, item_id:int, token: str=""):
    try:
        await websocket.accept()
        if token not in records:
            await websocket.send_json({"status": "error", "msg": "Invalid token"})
            await asyncio.sleep(0.1)
        awe_parser, daily_parser = records[token]
        while True:
            user_msg = await websocket.receive_text()
            user_msg = json.loads(user_msg)["message"]
            print(f"[USER]> {user_msg}")
            # 每日论文解读
            if item_id == 1:
                #for msg in fake_llm(user_msg):
                for msg in daily_parser.chat_gen(user_msg, return_buffer=False):
                    await websocket.send_text(msg)
                    await asyncio.sleep(0.1)
                pass
            # awesome 解析
            elif item_id == 2:
                for msg in awe_parser.chat_gen(user_msg, return_buffer=False):
                    await websocket.send_text(msg)
                    await asyncio.sleep(0.1)
                pass
            else:
                await websocket.send_text("No such item")
                await asyncio.sleep(0.1)
    except WebSocketDisconnect as e:
        if e.code == 1001:
            _logger.warning(f"[{e.code}] WebSocket closed by client_{token}")
            try:
                await websocket.close()
            except:
                pass
        else:
            _logger.error(f"[{e.code}] WebSocket closed.")
    except ConnectionResetError as e:
        _logger.warnning(f"{e}")

    # print(data)
    # while True:
    #     data = await websocket.receive_text()
    #     print(data)
    #     await websocket.send_text(f"Message text was: {data}")