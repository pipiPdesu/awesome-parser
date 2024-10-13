from dotenv import load_dotenv
load_dotenv()

from Robot.AwesomeParser import AwesomeParser

# how to use parser to get translated abstract
border = (91, 244)
parser = AwesomeParser(chat_model_name="meta/llama-3.1-405b-instruct")
parser.get_translated_summary("./jailbreak.md", border, "./jailbreak_abstract.md")