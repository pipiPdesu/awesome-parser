'''
load daily new paper from arxiv
'''
import arxivloader as al
from datetime import datetime, timedelta

def plus_one_day(date):
    date_obj = datetime.strptime(date, "%Y%m%d")
    date_obj += timedelta(days=1)
    new_date_str = date_obj.strftime("%Y%m%d")
    return new_date_str

def load_daily_paper(date='20240705', category='cs.CR'):
    '''
    load daily new paper from arxiv
    :param date: str, stand for date ex:'20240707'
    :param category: str, category of daily paper  ex:'cs.CR'
    :return: list of dict, daily paper in type of 2407.xxxxx and its summary
    '''
    # TODO: make sure the date and category is valid
    # TODO: make sure it return enough paper not too much not too less. 
    # default 100(i have changed my arxivloader the default number
    # it seems a little bit error with arxivloader  though it dosent matter

    prefix = "cat"
    cat = "cs.CR"
    submittedDate = f"[{date}000000+TO+{plus_one_day(date)}000000]" #要改时间就在这里改
    query = "search_query={pf}:{cat}+AND+submittedDate:{sd}".format(pf=prefix, cat=cat, sd=submittedDate)
    columns = ["id", "title", "authors", "published","summary"]

    df = al.load(query, columns=columns, sortBy="submittedDate", sortOrder="ascending")

    return df