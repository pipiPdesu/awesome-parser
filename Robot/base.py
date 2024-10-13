import os
import json
from operator import itemgetter
from typing import Iterable, Tuple
from functools import partial
from abc import ABC, abstractmethod

from faiss import IndexFlatL2
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.document_transformers import LongContextReorder
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableAssign
from langchain_core.output_parsers import StrOutputParser

from utils.logger import get_logger

_logger = get_logger(__name__)

def aggregate_vstores(vstore_dst, vstores_src):
    ## Initialize an empty FAISS Index and merge others into it
    ## We'll use default_faiss for simplicity, though it's tied to your embedder by reference
    for vstore in vstores_src:
        vstore_dst.merge_from(vstore)

def docs2str(docs, title="Document"):
    """Useful utility for making chunks into context string. Optional, but useful"""
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, 'metadata', {}).get('Title', title)
        if doc_name:
            out_str += f"[Quote from {doc_name}] "
        out_str += getattr(doc, 'page_content', str(doc)) + "\n"
    return out_str

## Utility Runnables/Methods
def RPrint(preface=""):
    """Simple passthrough "prints, then returns" chain"""
    def print_and_return(x, preface):
        _logger.debug(f"{preface} {x}")
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))

def save_memory_and_get_output(d, vstore):
    """Accepts 'input'/'output' dictionary and saves to convstore"""
    vstore.add_texts([
        f"User previously responded with {d.get('input')}",
        f"Agent previously responded with {d.get('output')}"
    ])
    return d.get('output')

class ChatBase(ABC):
    """用于聊天的基类"""

    chat_prompt = ChatPromptTemplate.from_messages([("system",
        "You are a document chatbot. Help the user as they ask questions about documents."
        " User messaged just asked: {input}\n\n"
        " From this, we have retrieved the following potentially-useful info: "
        " Conversation History Retrieval:\n{history}\n\n"
        " Document Retrieval:\n{context}\n\n"
        " (Answer only from retrieval. Only cite sources that are used. Make your response conversational.Reply must more than 100 words)"
    ), ('user', '{input}')])
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", ";", ",", " "],
    )
    default_vecstore_folder = "./local_index"
    default_cache_path = "./local_index/cache.json"    # cache 中用于缓存 init_msg

    def __init__(self, apikey: str = None, chat_model_name: str = "ai-mixtral-8x7b-instruct", **kwargs) -> None:
        try:
            self._cache = json.load(open(ChatBase.default_cache_path))  # 读取用户有过哪些向量库
        except:
            self._cache = {}
            json.dump(self._cache, open(ChatBase.default_cache_path, "w"))  # 清空缓存

        if apikey is not None:
            self.embedder = NVIDIAEmbeddings(model="ai-embed-qa-4", apikey=apikey)
            self.model = ChatNVIDIA(model=chat_model_name, apikey=apikey).bind(max_tokens=4096)
        else:
            self.embedder = NVIDIAEmbeddings(model="ai-embed-qa-4")
            self.model = ChatNVIDIA(model=chat_model_name).bind(max_tokens=4096)
        self.embed_dims = len(self.embedder.embed_query("test"))

        # 开场白
        self.papers_outlook = "你好，我是一个论文辅读 AI ，请问有什么可以帮到你的吗？"

        # vecstore
        self._docstore = self._default_FAISS()   # 文档库
        self._convstore = self._default_FAISS()  # 对话历史记录库

        ## 将较长的文档重新排序到输出文本的中心， RunnableLambda在链中运行无参自定义函数 ，长上下文重排序（LongContextReorder）
        long_reorder = RunnableLambda(LongContextReorder().transform_documents)

        # 检索链
        self._retrieval_chain = (
            {'input' : (lambda x: x)}
            | RunnableAssign({'history' : itemgetter('input') | self._convstore.as_retriever() | long_reorder | docs2str})
            | RunnableAssign({'context' : itemgetter('input') | self._docstore.as_retriever()  | long_reorder | docs2str})
            | RPrint()
        )
        # 聊天链
        self._stream_chain = ChatBase.chat_prompt | self.model | StrOutputParser()


    def _default_FAISS(self):
        '''Useful utility for making an empty FAISS vectorstore'''
        return FAISS(
            embedding_function=self.embedder,
            index=IndexFlatL2(self.embed_dims),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            normalize_L2=False
        )


    @staticmethod
    def check_apikey(apikey: str):
        """检查 apikey 是否有效"""
        _logger.info(f"Checking API key: {apikey}")
        if apikey is None or apikey == "" or not apikey.startswith("nvapi-"):
            return False
        try:
            embedder = NVIDIAEmbeddings(model="ai-embed-qa-4", apikey=apikey)
            embedder.embed_query("test")
        except Exception as e:
            _logger.info(f"--------------")
            _logger.info(e.args)
            _logger.info("--------------")
            if "Unauthor" in e.__str__():
                return False
            else:
                raise e
        return True


    @abstractmethod
    def load_paper(self, **kwargs) -> Tuple[str]:
        """加载文档"""
        ...

    def clear_vector_store(self):
        """清空向量库内容"""
        self._docstore = self._default_FAISS()
        self._convstore = self._default_FAISS()


    def chat_gen(self, message: str, history=[], return_buffer=True) -> Iterable:
        """聊天内容生成
        Args:
            message (str): 输入消息
            history (list, optional): 历史对话. Defaults to [].
            return_buffer (bool, optional): 是否返回 buffer 中的内容, buffer 中包含当前状态下已经生成的所有消息.
        """

        buffer = ""

        ## 首先根据输入的消息进行检索
        retrieval = self._retrieval_chain.invoke(message)

        ## 然后流式传输_stream_chain的结果
        for token in self._stream_chain.stream(retrieval):
            buffer += token
            if return_buffer:
                yield buffer
            else:
                yield token

        ## 最后将聊天内容保存到对话内存缓冲区中
        save_memory_and_get_output({'input':  message, 'output': buffer}, self._convstore)
