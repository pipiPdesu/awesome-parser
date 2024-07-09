from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from functools import partial
from langchain_core.runnables import RunnableLambda
import sys
sys.path.append('../')

def mapreduce(model, df):
    '''
    mapreduce function for chat for daily arxiv summry
    :param key: str, api key
    :param model: str, model to chat
    :param abstract: list, abstract of a list of arxiv paper
    :return: str, chat result
    '''
    df["generated_summary"] = ""
    def save_and_return():
        def save(x):
            for i in range(len(x)):
                df.loc[i, "generated_summary"] = x[i]
            return x
        return RunnableLambda(save)

    input_keys = ["text"]
    output_keys = ["context"]

    # 定义提示模板
    prompt = PromptTemplate(
        input_variables=input_keys,
        template="总结这篇文章的摘要,以中文回答:\n\n{text}"
    )

    # 构建 Map 链：对每个文档都先进行一轮总结
    map_chain = (
        prompt
        | model
        | StrOutputParser()
    )

    # 构建 Reduce 链：合并之前的所有总结内容
    reduce_chain = (
        save_and_return()
        | {"context": lambda strs: "\n\n".join(strs) }
        | PromptTemplate.from_template("以下是今天发表的最新文章的总结，你可以结合这些总结，分析一下当前的研究趋势以及尚可研究的问题:\n\n{context}")
        | model
        | StrOutputParser()
    )

    # 把两个链合并成 MapReduce 链
    map_reduce = map_chain.map() | reduce_chain
    result = map_reduce.invoke(df['summary'].tolist())
    return result, df


