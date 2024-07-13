'''
This is the loader for the file with arxiv link
'''
import re

def load_from_awesome(md_text):
    '''
    load arxiv link from markdown text
    :param md_text: str, markdown text
    :return: list of str, arxiv link
    '''
    arxiv_pattern = r'\d{4}\.\d{5,}'

    matches = re.findall(arxiv_pattern, md_text)

    return matches

