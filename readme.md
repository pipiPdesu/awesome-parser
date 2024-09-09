
# 配置

```bash
pip install -r requirements.txt
```

# 运行

在项目根目录下创建 .env 文件，添加以下内容

```
NVIDIA_API_KEY=xxxxxxxxx
```

执行以下命令

```bash
# 本地测试
uvicorn main:app --reload
# 服务器部署
uvicorn main:app --host 0.0.0.0 --port <端口>
```

浏览器访问

```
# 本地
http://127.0.0.1:8000/index
# 服务端
http://xxxx:xxxx/index
```