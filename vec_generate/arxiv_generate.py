'''
从list中读取arxiv的文章信息，生成对应的向量库 返回向量库路径
'''
from langchain.document_loaders import ArxivLoader
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore

def default_FAISS(embedder, embed_dims=1024):
    embed_dims = len(embedder.embed_query("test"))
    # TODO: 改嵌入模型就要改这里的维度
    '''Useful utility for making an empty FAISS vectorstore'''
    return FAISS(
        embedding_function=embedder,
        index=IndexFlatL2(embed_dims),
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
        normalize_L2=False
    )

def aggregate_vstores(vectorstores, embedder):
    ## Initialize an empty FAISS Index and merge others into it
    ## We'll use default_faiss for simplicity, though it's tied to your embedder by reference
    agg_vstore = default_FAISS(embedder)
    for vstore in vectorstores:
        agg_vstore.merge_from(vstore)
    return agg_vstore

def arxiv_generate(arxiv_list, embedder, save_path):
    '''
    从list中读取arxiv的文章信息，生成对应的向量库 返回向量库路径
    :param arxiv_list: str, arxiv文章信息的list路径
    :param embedder: 嵌入模型
    :param save_path: str, 保存向量库的路径
    :return: str, 向量库路径
    '''
    print("Loading Documents")
    docs = [
    ArxivLoader(query = arxiv_list[i]).load() for i in range(len(arxiv_list))
    ]

    print("Cleaning Documents")
    for doc in docs:
        content = json.dumps(doc[0].page_content)
        if "References" in content:
            doc[0].page_content = content[:content.index("References")]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250, chunk_overlap=50,
        separators=["\n\n", "\n", ".", ";", ",", " "],
    )

    ## Split the documents and also filter out stubs (overly short chunks)
    print("Chunking Documents")
    docs_chunks = [text_splitter.split_documents(doc) for doc in docs]
    docs_chunks = [[c for c in dchunks if len(c.page_content) > 200] for dchunks in docs_chunks]
    #注意这里的200会有误伤  比如https://arxiv.org/abs/2407.04757 这篇裁完基本上没东西了

    ## Make some custom Chunks to give big-picture details
    doc_string = "Available Documents:"
    doc_metadata = []
    for chunks in docs_chunks:
        metadata = getattr(chunks[0], 'metadata', {})
        doc_string += "\n - " + metadata.get('Title')
        doc_metadata += [str(metadata)]

    extra_chunks = [doc_string] + doc_metadata

    print("Creating Vectorstores")
    vecstores = [FAISS.from_texts(extra_chunks, embedder)]
    vecstores += [FAISS.from_documents(doc_chunks, embedder) for doc_chunks in docs_chunks]

    docstore = aggregate_vstores(vecstores, embedder)


    docstore.save_local(save_path)