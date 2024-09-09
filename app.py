import os
import time
import json
import asyncio
from functools import partial
from dotenv import load_dotenv
from typing import List

from Robot.AwesomeParser import AwesomeParser
from utils.logger import setup_base_logger, get_logger
from Robot.DailyParser import DailyParser

from fastapi import FastAPI, Request, WebSocket, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# 初始化
load_dotenv()
setup_base_logger()
_logger = get_logger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def load_daily_paper():
    global daily_parser
    daily_parser = DailyParser()
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


@app.get("/index", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(
        request=request, name="index.html.jinja"
    )


@app.get("/chat/{id}", response_class=HTMLResponse)
async def chat(request: Request, id: int):
    return templates.TemplateResponse(
        request=request, name="chat.html.jinja", context={"id": id}
    )


@app.post("/upload")
async def create_upload_files(files: List[UploadFile]):
    global awe_parser
    dic = {}
    if len(files) == 0:
        return {"error": "No files uploaded"}
    awe_parser = AwesomeParser()
    contents = ""
    for file in files:
        temp = await file.read()
        contents += temp.decode() + "\n\n"
        # print(contents)
    _, _ = awe_parser.load_paper(contents)
    return dic

# ------------ 即时传 ------------
@app.websocket("/load")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    info = load_daily_paper()
    await websocket.send_json(info)
    await asyncio.sleep(0.1)
    await websocket.close()

@app.websocket("/daily_paper_prologue")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text(daily_parser.papers_outlook)
    await asyncio.sleep(0.1)
    await websocket.close()

@app.websocket("/awe_parser_prologue")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text(awe_parser.papers_outlook)
    await asyncio.sleep(0.1)
    await websocket.close()

# ---------- 持久会话 -------------
@app.websocket("/{item_id}/chat_ws")
async def websocket_endpoint(websocket: WebSocket, item_id:int):
    await websocket.accept()
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
        if item_id == 2:
            pass

    # print(data)
    # while True:
    #     data = await websocket.receive_text()
    #     print(data)
    #     await websocket.send_text(f"Message text was: {data}")