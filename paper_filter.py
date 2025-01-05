import arxiv
import os
import time
from typing import List, Dict
from openai import OpenAI
import logging
from tqdm import tqdm
import pandas as pd
import json
import pymongo
from arxiv_search import ArxivScraper
from progress import ProgressManager
from async_preprocess_utils import evaluate_paper
progress_manager=ProgressManager('data\progress_saving.json')

secret = json.load(open('data/secret.json', 'r'))
# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

class ArxivPaperFilter:
    def __init__(self, api_key: str, output_file: str = "filtered_papers.csv"):
        """
        初始化类
        :param api_key: OpenAI API密钥
        :param output_file: 输出文件路径
        """
        self.api_key = api_key
        self.output_file = output_file
        self.processed_papers = self._load_processed_papers()

    def _load_processed_papers(self) -> pd.DataFrame:
        """
        加载已处理的论文记录
        """
        if os.path.exists(self.output_file):
            try:
                return pd.read_csv(self.output_file, encoding='utf-8')
            except pd.errors.EmptyDataError:
                return pd.DataFrame(columns=['id', 'title', 'abstract', 'authors', 'published', 'url', 'score', 'reason', 'keyword'])
        return pd.DataFrame(columns=['id', 'title', 'abstract', 'authors', 'published', 'url', 'score', 'reason', 'keyword'])

    def _save_processed_papers(self):
        """
        保存处理结果到CSV文件
        """
        self.processed_papers.to_csv(self.output_file, index=False, encoding='utf-8')

    def _is_paper_processed(self, paper_id: str) -> bool:
        """
        检查论文是否已经处理过
        """
        return paper_id in self.processed_papers['id'].values

    def search_and_filter_papers(self, keyword: str, criteria: str, 
                               date_limit: str = "", score_threshold: float = 6.0):
        """
        搜索和筛选论文
        """
        # client = arxiv.Client()
        scraper = ArxivScraper()
        keyword = keyword.strip()
        logging.info(f"正在处理关键词: {keyword}")
        try:
            for paper in tqdm(scraper.search(keyword)):
                if self._is_paper_processed(paper['url']):
                    continue

                paper_description = f"""
                标题：{paper['title']}
                摘要：{paper['abstract']}
                """

                evaluation = evaluate_paper(paper_description, criteria)
                
                # if evaluation["score"] >= score_threshold:
                paper_info = {
                    "id": paper['url'],
                    "title": paper['title'],
                    "abstract": paper['abstract'],
                    "authors": ', '.join([author for author in paper['authors']]),
                    "published": paper['published_date'],
                    "url": paper['pdf_url'],
                    "score": evaluation["score"],
                    "reason": evaluation["reason"],
                    "keyword": keyword
                }
                
                # 将新论文添加到DataFrame
                new_paper_df = pd.DataFrame([paper_info])
                self.processed_papers = pd.concat([self.processed_papers, new_paper_df], ignore_index=True)
                
                # 立即保存，以防中断
                self._save_processed_papers()
                
                time.sleep(1)  # 避免API限制

        except Exception as e:
            logging.error(f"搜索过程中出错: {str(e)}")
            self._save_processed_papers()

        # 最后按分数降序排序并保存
        self.processed_papers = self.processed_papers.sort_values('score', ascending=False)
        self._save_processed_papers()
        logging.info(f"处理完成，结果已保存到 {self.output_file}")
    def filter_papers_by_mongo(self,criteria:str, mongo_client_url: str|None=None, collection: str='paper_daily', data_base: str = 'data', 
                               date_limit: str = "2023-01-01", score_threshold: float = 6.0):
        """
        搜索和筛选论文
        """
        myclient = pymongo.MongoClient(mongo_client_url if mongo_client_url else secret['mongo_client_url'])    
        collection = myclient[collection]
        data_base=collection[data_base]
        client = arxiv.Client()
        for data in data_base.find():
            try:
                print("正在处理数据:",data['date_time'])
                arxiv_ids=[key.split('/')[-1].replace(',','.') for key in data['data']]
                search = arxiv.Search(
                    query='',
                    id_list=arxiv_ids,
                    sort_by=arxiv.SortCriterion.SubmittedDate,
                    sort_order=arxiv.SortOrder.Descending
                )
                for paper in tqdm(client.results(search)):
                    if self._is_paper_processed(paper.entry_id):
                        continue

                    paper_description = f"""
                    标题：{paper.title}
                    摘要：{paper.summary}
                    作者：{', '.join([author.name for author in paper.authors])}
                    发布日期：{paper.published.strftime("%Y-%m-%d")}
                    """

                    evaluation = evaluate_paper(paper_description, criteria)
                    
                    if evaluation["score"] >= score_threshold:
                        paper_info = {
                            "id": paper.entry_id,
                            "title": paper.title,
                            "abstract": paper.summary,
                            "authors": ', '.join([author.name for author in paper.authors]),
                            "published": paper.published.strftime("%Y-%m-%d"),
                            "url": paper.pdf_url,
                            "score": evaluation["score"],
                            "reason": evaluation["reason"],
                            "keyword": ''
                        }
                        
                        # 将新论文添加到DataFrame
                        new_paper_df = pd.DataFrame([paper_info])
                        self.processed_papers = pd.concat([self.processed_papers, new_paper_df], ignore_index=True)
                        
                        # 立即保存，以防中断
                        self._save_processed_papers()
                    
                    time.sleep(1)  # 避免API限制

            except Exception as e:
                logging.error(f"搜索过程中出错: {str(e)}")
                self._save_processed_papers()

        # 最后按分数降序排序并保存
        self.processed_papers = self.processed_papers.sort_values('score', ascending=False)
        self._save_processed_papers()
        logging.info(f"处理完成，结果已保存到 {self.output_file}")

def paper_spider_filter(keywords: List[str], criteria: str,output_file:str='data/filtered_papers.csv'):
    # 配置参数
    keywords=[_.strip() for _ in keywords]
    api_key = secret['api_key']    
    paper_filter = ArxivPaperFilter(api_key,output_file=output_file)
    count=0
    # paper_filter.filter_papers_by_mongo(criteria)
    for keyword in keywords:
        paper_filter.search_and_filter_papers(
            keyword=keyword,
            criteria=criteria,
        )
        count+=1
        progress_manager.update_download_status(completed=count,remaining=len(keywords)-count)
    paper_filter.processed_papers = paper_filter.processed_papers[paper_filter.processed_papers['keyword'].isin(keywords)]
    paper_filter._save_processed_papers()

    

if __name__ == "__main__":
    save_data = json.load(open('data\progress_saving.json','r',encoding='utf-8'))
    paper_spider_filter(save_data['search_config']['keywords'].strip().split('\n'),save_data['search_config']['criterion'])
    
    

