import logging
from rich.logging import RichHandler

PROJECT_NAME = "AwesomeParser"

def setup_base_logger():
    """设置根日志记录器，后续的日志都继承于这个根日志记录器"""
    # logging.basicConfig(
    #     level=logging.DEBUG,
    #     stream=RichHandler(level=logging.DEBUG),
    #     #style=CustomFormatter
    # )
    # 创建一个 logger，这里使用模块名称来命名 logger ，这样仅从 logger 名称就可以直观地看出事件的记录位置。
    logger = logging.getLogger(PROJECT_NAME)

    # 设置 logger 的初始 level
    logger.setLevel(logging.INFO)

    logger.addHandler(
        RichHandler(rich_tracebacks=True)
    )

    # FORMAT = "%(message)s"
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format=FORMAT,
    #     datefmt="[%X]",
    #     handlers=[
    #         RichHandler(
    #             rich_tracebacks=True,
    #         )
    #     ]
    # )

def get_logger(name):
    """获取一个 logger，如果不存在则创建一个新的 logger"""
    return logging.getLogger(f"{PROJECT_NAME}.{name}")