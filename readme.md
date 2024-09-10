
# 配置

```bash
pip install -r requirements.txt
```

# 运行

在项目根目录下执行以下命令

```bash
# 本地测试
uvicorn main:app --reload
python3 -m uvicorn main:app --reload
# 服务器部署
uvicorn main:app --host 0.0.0.0 --port <端口>
python3 -m uvicorn main:app --host 0.0.0.0 --port <端口>
```

浏览器访问

```
# 本地
http://127.0.0.1:8000/index
# 服务端
http://xxxx:xxxx/index
```

## Nvidia

绑定到 5000 端口哦

```bash
python3 -m uvicorn main:app --host 0.0.0.0 --port 5000
```

访问 http://36.150.110.74:9536/index