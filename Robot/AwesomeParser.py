import hashlib
from utils.logger import get_logger
from pathlib import Path
from typing import Tuple, List
import sys
from langchain.vectorstores import FAISS

from utils.loader import load_paper_from_awesome, docs2vecstore, query_paper_meta
from Robot.base import ChatBase, aggregate_vstores

_logger = get_logger(__name__)


def file_signature(lst: List) -> str:
    """生成 awesome 文件的 MD5 哈希"""
    lst.sort()
    _md5 = hashlib.md5(str(lst).encode()).hexdigest()
    return _md5


class AwesomeParser(ChatBase):
    """Awesome 解析器"""
    def load_paper(
        self,
        text
    ) -> Tuple[str]:
        matches, _ = load_paper_from_awesome(text, False)
        sig = file_signature(matches)         # 文件签名，后续用于判断是否需要重新加载
        _logger.info(f"Awesome file signature: {sig}")
        path = Path(ChatBase.default_vecstore_folder) / sig  # 本地文档库目标路径
        if not path.exists():
            _logger.info(f"Local vecstore not found, fetching from network")
            _, docs = load_paper_from_awesome(text)
            # for doc in docs:
            #     doc.metadata["Title"]
            #     doc.metadata["Authors"]
            #     doc.metadata["Summary"]
            vecstores = docs2vecstore(docs, AwesomeParser.text_splitter, self.embedder)
            aggregate_vstores(self._docstore, vecstores)
            # 保存到本地
            self._docstore.save_local(folder_path=str(path), index_name=sig)
            _logger.info(f"Local vecstore saved to {path}")
        else:
            _logger.info(f"Local vecstore found, loading from local: {path}")
            # 加载本地文档库
            self._docstore = FAISS.load_local(
                folder_path=path,
                embeddings=self.embedder,
                allow_dangerous_deserialization=True
            )
        _logger.info(f"Paper loaded, cache updated")
        self.papers_outlook = f"你好, 我是 awesome-parser, 一个帮助用户解析 awesome-list 的工具。我从您提供的文档中解析到了 {len(matches)} 篇 paper: \n\n"
        for title, names, link, published in query_paper_meta(matches):
            self.papers_outlook += f"* [{title}]({link}) by {names} pubilshed on {published}.\n\n"
        self.papers_outlook += "请问我可以帮助您吗?"
        return str(path), matches
