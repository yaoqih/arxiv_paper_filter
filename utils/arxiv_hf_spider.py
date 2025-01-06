import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests
import datetime
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from postprocess_utils import extract_labels
from purify import PaperCategoryManager
from build_index import build_index
import re
basic_path='paper_outlook/'
def parse_publish_date(date_str):
    # 移除 "Published on " 前缀
    date_str = date_str.replace('Published on ', '')
    
    # 检查是否包含年份
    if not re.search(r'\d{4}$', date_str):
        # 如果没有年份，添加当前年份
        current_year = datetime.datetime.now().year
        date_str = f"{date_str}, {current_year}"
    
    # 转换日期
    date_obj = datetime.datetime.strptime(date_str, '%b %d, %Y')
    return date_obj.strftime('%Y-%m-%d')
def get_abstract_date(id):
    html_content = requests.get(f"https://huggingface.co/papers/{id}").text
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.find('h2',class_='text-xl').parent.contents[2].text.strip().replace('\n',' ').replace(',','，'), parse_publish_date(soup.find('h1').parent.contents[4].contents[0].text)

def get_paper_today(date):
    try:
        html_content=requests.get(f"https://huggingface.co/papers?date={date}").text
    except:
        html_content=requests.get(f"https://huggingface.co/papers?date={date}",proxies={'http':f"http://127.0.0.1:7897",'https':f"http://127.0.0.1:7897"}).text
    # html_content=requests.get(f"https://huggingface.co/papers?date=2024-07-24").text
    soup = BeautifulSoup(html_content, 'html.parser')
    result={}
    for article in soup.find_all('article'):
        id=article.find('a').attrs['href'].split('/')[-1]
        title=article.find('h3').text.replace('"',"'").strip()
        vote_num=int(article.find("svg").parent.contents[4].contents[0]) if '-' not in article.find("svg").parent.contents[4].contents[0] else 0
        abstract,published=get_abstract_date(id)
        result[title]={'id':"https://arxiv.org/abs/"+id,'title':title,'score':vote_num,'abstract':abstract,'authors':[],'published':published,'url':"https://arxiv.org/pdf/"+id,'reason':'','keyword':''}
    # 定义正则表达式模式
    return result

def get_workdays(start_date_str, end_date_str):
    # 将字符串转换为datetime对象
    start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')
    
    # 存储所有工作日
    workdays = []
    
    # 遍历日期范围
    current_date = start_date
    while current_date <= end_date:
        # 检查是否为工作日（周一到周五）
        if current_date.weekday() < 5:  # 0-4 表示周一到周五
            workdays.append(current_date.strftime('%Y-%m-%d'))
        current_date += datetime.timedelta(days=1)
    
    return workdays
if __name__ == '__main__':
    start_date='2024-07-22'
    if not os.path.exists(basic_path):
        os.makedirs(basic_path)
    if os.path.exists(os.path.join(basic_path,'download_process_save.txt')):
        download_process_save=open(os.path.join(basic_path,'download_process_save.txt'),'r').read()
    else:
        download_process_save=''
    if os.path.exists(os.path.join(basic_path,'filtered_papers.csv')):
        filtered_papers=pd.read_csv(os.path.join(basic_path,'filtered_papers.csv'),encoding='utf-8')
    else:
        filtered_papers=pd.DataFrame(columns=['id','title','score','abstract','authors','published','url','reason','keyword'])
    for date in tqdm(get_workdays(start_date,datetime.datetime.now().strftime('%Y-%m-%d'))):
        if date in download_process_save:
            continue
        hf_arxiv_data=get_paper_today(date)
        for key in hf_arxiv_data:
            if key not in filtered_papers['title'].values and key not in download_process_save:
                download_process_save+=key+'\n'
                filtered_papers=pd.concat([filtered_papers,pd.DataFrame([hf_arxiv_data[key]])],ignore_index=True)
        filtered_papers.to_csv(os.path.join(basic_path,'filtered_papers.csv'),index=False,encoding='utf-8')
        open(os.path.join(basic_path,'download_process_save.txt'),'a').write(f'{date}\n')
    extract_labels(os.path.join(basic_path,'filtered_papers.csv'),os.path.join(basic_path,'paper_labeled.csv'),os.path.join(basic_path,'labels.json'),write=True)
    manager = PaperCategoryManager(label_file=os.path.join(basic_path,'labels.json'),out_dir=basic_path)
    manager.process_duplicates()
    manager.find_similar_keywords(manager.processed_data)
    manager.save_results(processed_data=True)
    build_index(os.path.join(basic_path,'labels_purify.json'),os.path.join(basic_path,'labeled_purify.csv'),os.path.join(basic_path,'label_index.json'))