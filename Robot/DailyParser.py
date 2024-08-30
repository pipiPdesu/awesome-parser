import json
from pathlib import Path
from typing import Tuple, List

from langchain.document_loaders import ArxivLoader
from langchain.vectorstores import FAISS

from utils.loader import load_paper_from_date, docs2vecstore
from Robot.DocHandler import mapreduce
from Robot.base import ChatBase, aggregate_vstores

from utils.logger import get_logger

_logger = get_logger(__name__)

class DailyParser(ChatBase):
    """每日论文的解析器"""

    def load_paper(
        self,
        date: str = '20240705',
        category: str = 'cs.CR',
        verbose: bool = False
    ) -> Tuple[str, List[List]]:
        """根据给定日期和类型加载 Arxiv 论文
        Args:
            date (str): 日期，格式为 'YYYYMMDD'
            category (str): Arxiv 分类，例如 'cs.CR'
            verbose (bool): 是否将当天 arxiv 论文保存为表格
        Returns:
            Tuple[str]: (path, summary)
        """

        # 加载文档
        name = f"{date}_{category.replace('.', '_')}"
        path = Path(DailyParser.default_vecstore_folder) / name  # 本地文档库目标路径
        df = None
        if not path.exists():
            _logger.info(f"Local vecstore not found, fetching from network")
            df = load_paper_from_date(date, category)
            outlook, detail = mapreduce(DailyParser.model, df)
            paper_info = [
                    # 文章标题, 作者, 进一步的总结, 文章链接
                    [title, author, summary, link] for title, author, summary, link in \
                        zip(detail['title'].to_list(),\
                            detail["authors"].to_list(),\
                            detail['generated_summary'].to_list(),\
                            detail["links"].to_list() \
                        )
                ]
            docs = [
                ArxivLoader(query = detail['id'][i]).load() for i in range(len(detail))
            ]
            vecstores = docs2vecstore(docs, DailyParser.text_splitter, DailyParser.embedder)
            aggregate_vstores(self._docstore, vecstores)
            # 保存到本地
            self._docstore.save_local(folder_path=path, index_name=name)
            # 更新 cache
            self._cache[name] = [str(path), outlook, paper_info]
        else:
            _logger.info(f"Local vecstore found, loading from local: {path}")
            # 加载本地文档库
            self._docstore = FAISS.load_local(
                folder_path=path,
                embeddings=DailyParser.embedder,
                # allow_dangerous_deserialization=True
            )
            if name not in self._cache:
                df = load_paper_from_date(date, category)
                outlook, detail = mapreduce(DailyParser.model, df)
                paper_info = [
                    # 文章标题, 作者, 进一步的总结, 文章链接
                    [title, author, summary, link] for title, author, summary, link in \
                        zip(detail['title'].to_list(),\
                            detail["authors"].to_list(),\
                            detail['generated_summary'].to_list(),\
                            detail["links"].to_list() \
                        )
                ]
                self._cache[name] = [str(path), outlook, paper_info]

        if verbose and df is not None:
            df.to_csv(f"{name}.csv")  # 保存当天的 arxiv 表格

        # 更新 cache 到本地
        with open(DailyParser.default_cache_path, "w") as f:
            json.dump(self._cache, f, indent=4)
        _logger.info(f"Paper loaded, cache updated")
        return self._cache[name]