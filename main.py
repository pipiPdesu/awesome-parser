import os
import json
import gradio as gr
from functools import partial
from dotenv import load_dotenv
from Robot.AwesomeParser import AwesomeParser

from utils.logger import setup_base_logger, get_logger
from Robot.DailyParser import DailyParser

_logger = get_logger(__name__)

head_style = """
<style>
@media (min-width: 1536px)
{
    .gradio-container {
        min-width: 100% !important;
        margin-left: 0px !important;
        margin-right: 0px !important;
    }
}
</style>
"""

css = """
#paper-info {
    height: 100%;
    width: 100%;
}
#chatbot {
    flex-grow: 1 !important;
    overflow: auto !important;
}
"""

def init():
    setup_base_logger()
    load_dotenv()

def gen_chatbot(msg: str):
    return gr.Chatbot(
        height= 500,
        value=[
            [None, msg]
        ],
        render=False   # Ref: https://discuss.huggingface.co/t/clear-chat-interface/49866/3
    )


init_msg = "Hello! I am a document chat agent here to help the user! How can I help you?"
daily_parser = DailyParser()
awe_parser = AwesomeParser()

def main():
    date = '20240707'

    path, outlook, paper_info = daily_parser.load_paper(date)

    with gr.Blocks(
        theme=gr.themes.Soft(),
        fill_height = True,
        title="My daily paper assistant",
        css=css,
        head=head_style
    ) as demo:
        # ----------------- daily arxiv -----------------
        with gr.Tab(label="daily arxiv"):
            with gr.Row(variant='panel', elem_id="paper-info"):
                # 第一列
                with gr.Column(scale=1):
                    ## 论文详细信息
                    with gr.Accordion(f"{date}"):
                        for title, author, summary, link in paper_info:
                            with gr.Accordion(title, open=False):
                                gr.Markdown(author)
                                gr.Markdown(summary)
                # 第二列
                with gr.Column(scale=4):
                    ## 聊天界面
                    gr.ChatInterface(
                        daily_parser.chat_gen,
                        chatbot=gr.Chatbot(
                            height= 500,
                            value=[[None, outlook]],
                            render=False   # Ref: https://discuss.huggingface.co/t/clear-chat-interface/49866/3
                        ),
                        textbox=gr.Textbox(placeholder="Ask me a yes or no question", container=False, scale=7, render=False),
                        #description="Ask it any question",
                        theme="soft",
                        # TODO:
                        # examples=["Hello", "Am I cool?", "Are tomatoes vegetables?"],
                        # cache_examples=True,
                        retry_btn=None,
                        undo_btn="Delete Previous",
                        clear_btn="Clear",
                    ).queue()

        # ----------------- awesome -----------------
        with gr.Tab(label="awsome parser"):
            with gr.Row():
                # 第一列
                with gr.Column(scale=2, min_width= 100):   # 左侧的预设区，上传文件后即可对上传的文件利用 awesome_chat 进行赏析，不过 filemd 输出的文件路径不清楚
                    gr.Markdown(f"<h3 style='text-align: center; margin-bottom: 1rem'>请上传查询的md</h3>")
                    ## 文件上传组件
                    awe_file = gr.File(file_count='single',label="put markdown file here", file_types=["md"])
                # 第二列
                with gr.Column(scale=6):
                    awe_chatbot = gr.Chatbot(
                        height= 500,
                        value=[
                            [None, init_msg]
                        ],
                        render=False   # Ref: https://discuss.huggingface.co/t/clear-chat-interface/49866/3
                    )
                    ## 上传事件
                    @gr.render(inputs=[awe_file], triggers=[awe_file.upload])
                    def refresh(file):
                        if file:
                            _logger.info(f"{file} uploaded and parsing.")
                            _, init_msg = awe_parser.load_paper(file)
                            awe_chatbot.value = [[None, init_msg]]    # TODO: 貌似无用，后端无法更新到前端
                    ## 文件删除事件
                    ...
                    ## 聊天界面
                    gr.ChatInterface(
                        fn = awe_parser.chat_gen,
                        chatbot = awe_chatbot,
                        additional_inputs = None,
                        title = "awsome-parser",
                        submit_btn = "Submit",
                        fill_height = True,
                        undo_btn= "↩️ 清空前言",
                        clear_btn= "🗑️ 清空对话",
                        stop_btn= "⏸ 停止生成",
                        retry_btn= "🔄 重新生成",
                    ).queue()

    try:
        demo.launch(debug=True, share=False, show_api=False, server_port=8090, server_name="0.0.0.0")
        demo.close()
    except Exception as e:
        demo.close()
        _logger.error(e)
        raise e
    return

if __name__ == "__main__":
    init()
    main()
    #main()