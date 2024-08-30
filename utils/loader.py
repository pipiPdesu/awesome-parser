import re
import json
import arxivloader as al
from typing import List, Tuple
from datetime import datetime, timedelta
from pandas import DataFrame

from langchain.vectorstores import FAISS
from langchain.document_loaders import ArxivLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain.text_splitter import TextSplitter

from utils.logger import get_logger

_logger = get_logger(__name__)


def load_paper_from_awesome(md_text: str, getdoc: bool = True) -> Tuple[List[str], List[Document]]:
    """从 markdown 文本中提取出 arxiv 链接
    Args:
        md_text (str)    : markdown text, 不是路径
        getdoc  (bool)   : 是否获取论文信息
    Returns:
        (List[str])      : arxiv 论文 id 列表\\
        (List[Document]) : arxiv 论文信息列表
    """
    arxiv_pattern = r'\d{4}\.\d{5,}'

    matches = re.findall(arxiv_pattern, md_text)

    docs = None
    if getdoc:
        docs = [
            ArxivLoader(query = matches[i]).load() for i in range(len(matches))
        ]
    return matches, docs


def load_paper_from_date(date='20240705', category='cs.CR') -> DataFrame:
    """指定日期、类别来获取论文数据
    Args:
        date     (str)  : 日期，格式为 'YYYYMMDD' e.g. '20240705'
        category (str)  : 论文类别，例如 'cs.CR'
    Returns:
        DataFrame: 包含论文信息的 DataFrame
    """
    # TODO: make sure the date and category is valid
    # TODO: make sure it return enough paper not too much not too less.
    # default 100(i have changed my arxivloader the default number
    # it seems a little bit error with arxivloader  though it dosent matter

    _logger.info(f"Loading ArXiv papers from {date} in category {category}")

    def _plus_one_day(date):
        date_obj = datetime.strptime(date, "%Y%m%d")
        date_obj += timedelta(days=1)
        new_date_str = date_obj.strftime("%Y%m%d")
        return new_date_str

    prefix = "cat"
    cat = "cs.CR"
    submittedDate = f"[{date}000000+TO+{_plus_one_day(date)}000000]" #要改时间就在这里改
    query = "search_query={pf}:{cat}+AND+submittedDate:{sd}".format(pf=prefix, cat=cat, sd=submittedDate)

    # ["id", "title", "summary", "authors", "primary_category", "categories", "comments", "updated", "published", "doi", "links"]
    df: DataFrame = al.load(query, sortBy="submittedDate", sortOrder="ascending")
    return df


def docs2vecstore(
        docs: List[Document],
        splitter: TextSplitter,
        embedder: Embeddings
    ) -> FAISS:
    """加载 documents 列表，并返回 FAISS 向量存储
    Args:
        docs (list): 包含论文信息的列表
        splitter (object): 文档切分器
        embedder (object): 嵌入模型
    """
    for doc in docs:
        content = json.dumps(doc[0].page_content)
        if "References" in content:
            doc[0].page_content = content[:content.index("References")]

    # 文档切分
    ## Split the documents and also filter out stubs (overly short chunks)
    _logger.info("Chunking Documents")
    docs_chunks = [splitter.split_documents(doc) for doc in docs]
    docs_chunks = [[c for c in dchunks if len(c.page_content) > 200] for dchunks in docs_chunks]
    #注意这里的200会有误伤  比如 https://arxiv.org/abs/2407.04757 这篇裁完基本上没东西了

    ## Make some custom Chunks to give big-picture details
    doc_string = "Available Documents:"
    doc_metadata = []
    for chunks in docs_chunks:
        metadata = getattr(chunks[0], 'metadata', {})
        doc_string += "\n - " + metadata.get('Title')
        doc_metadata += [str(metadata)]
    extra_chunks = [doc_string] + doc_metadata

    ## Printing out some summary information for reference
    _logger.info(doc_string + '\n')
    for i, chunks in enumerate(docs_chunks):
        _logger.debug(
            f"Document {i}\n"
            f" - # Chunks: {len(chunks)}\n"
            f" - Metadata:\n"
            f"{json.dumps(chunks[0].metadata, indent=4)}"
        )

    # 合并到向量库
    vecstores = [FAISS.from_texts(extra_chunks, embedder)]
    vecstores += [FAISS.from_documents(doc_chunks, embedder) for doc_chunks in docs_chunks]
    return vecstores