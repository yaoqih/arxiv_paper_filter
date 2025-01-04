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
config = json.load(open('config.json', 'r'))
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
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.chatanywhere.tech"
        )
        self.processed_papers = self._load_processed_papers()

    def _load_processed_papers(self) -> pd.DataFrame:
        """
        加载已处理的论文记录
        """
        if os.path.exists(self.output_file):
            try:
                return pd.read_csv(self.output_file, encoding='utf-8')
            except pd.errors.EmptyDataError:
                return pd.DataFrame(columns=['id', 'title', 'abstract', 'authors', 'published', 'url', 'score', 'evaluation', 'keyword'])
        return pd.DataFrame(columns=['id', 'title', 'abstract', 'authors', 'published', 'url', 'score', 'evaluation', 'keyword'])

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

    def evaluate_paper_with_gpt(self, paper_info: str, criteria: str) -> dict:
        """
        使用GPT评估论文并打分
        """
        try:
            prompt = f"""
            请评估以下论文与给定标准的相关性，并给出1-10的分数评价（10分最相关）：

            论文信息：
            {paper_info}

            评估标准：
            {criteria}

            请按照以下格式回复：
            分数：[1-10的分数]
            理由：[简要解释评分理由]

            注意：
            - 分数标准：
              10分：完全符合核心要求，直接相关
              7-9分：高度相关，但可能不是核心主题
              4-6分：部分相关，有一些相关元素
              1-3分：较少相关性
              0分：完全不相关
            """

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "你是一个专业的论文评估助手，请严格按照给定标准评估论文并给出合理的分数。请严格按照下面的例子给出评分标准，这样才能保证解析不会出错"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            result = response.choices[0].message.content.strip()
            
            # 解析返回的评分和理由
            score = 0
            reason = ""
            for line in result.split('\n'):
                if line.startswith('分数：'):
                    try:
                        score = float(line.replace('分数：', '').strip())
                    except:
                        score = 0
                elif line.startswith('理由：'):
                    reason = line.replace('理由：', '').strip()

            return {
                "score": score,
                "reason": reason
            }

        except Exception as e:
            logging.error(f"GPT评估出错: {str(e)}")
            return {"score": 0, "reason": f"评估错误: {str(e)}"}

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
            # query = f'all:({" AND ".join([f"{_}"for _ in keyword.split(" ")])})'  # 例如：all:"deep learning" AND all:"time series"                
            # search = arxiv.Search(
            #     query=query,
            #     max_results=max_results,
            #     # sort_by=arxiv.SortCriterion.SubmittedDate,
            #     # sort_order=arxiv.SortOrder.Descending
            # )

            for paper in tqdm(scraper.search(keyword)):
                if self._is_paper_processed(paper['url']):
                    continue

                paper_description = f"""
                标题：{paper['title']}
                摘要：{paper['abstract']}
                """

                evaluation = self.evaluate_paper_with_gpt(paper_description, criteria)
                
                if evaluation["score"] >= score_threshold:
                    paper_info = {
                        "id": paper['url'],
                        "title": paper['title'],
                        "abstract": paper['abstract'],
                        "authors": ', '.join([author for author in paper['authors']]),
                        "published": paper['published_date'],
                        "url": paper['pdf_url'],
                        "score": evaluation["score"],
                        "reason": evaluation["reason"],
                        "search_word": keyword
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
        myclient = pymongo.MongoClient(mongo_client_url if mongo_client_url else config['mongo_client_url'])    
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

                    evaluation = self.evaluate_paper_with_gpt(paper_description, criteria)
                    
                    if evaluation["score"] >= score_threshold:
                        paper_info = {
                            "id": paper.entry_id,
                            "title": paper.title,
                            "abstract": paper.summary,
                            "authors": ', '.join([author.name for author in paper.authors]),
                            "published": paper.published.strftime("%Y-%m-%d"),
                            "url": paper.pdf_url,
                            "score": evaluation["score"],
                            "evaluation": evaluation["reason"],
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

def paper_spider_filter(keywords: List[str], criteria: str):
    # 配置参数
    api_key = config['api_key']

    # 创建过滤器实例并运行
    
    paper_filter = ArxivPaperFilter(api_key,output_file='output.csv')
    count=0
    # paper_filter.filter_papers_by_mongo(criteria)
    for keyword in keywords:
        paper_filter.search_and_filter_papers(
            keyword=keyword,
            criteria=criteria,
            # date_limit="2023-01-01",
            # score_threshold=6.0
        )
        count+=1
        save_data = json.load(open(config['config_file'],'r',encoding='utf-8'))
        save_data['download_status']['completed'] = count
        save_data['download_status']['remaining'] = len(keywords) - count
        json.dump(save_data, open(config['config_file'],'w',encoding='utf-8'))
        

if __name__ == "__main__":
    save_data = json.load(open(config['config_file'],'r',encoding='utf-8'))
    paper_spider_filter(save_data['search_config']['keywords'].strip().split('\n'),save_data['search_config']['intent'])
    keywords = [
    "multivariate time series forecasting pre-training fine-tuning",
    "multivariate time series prediction transfer learning",
    "pre-trained models multivariate time series",
    "fine-tuning multivariate time series models",
    "deep learning multivariate time series pre-training",
    "transfer learning time series forecasting",
    "pre-training fine-tuning time series prediction",
    "multivariate time series deep learning transfer",
    "pre-trained neural networks time series",
    "fine-tuned models multivariate forecasting"
    ]

    criteria = """
        请判断一下这篇论文是否与多变量时间序列预测中预训练和微调模型有关，特别是着重考虑以下几个方面：
    多变量时间序列数据处理：
    论文是否处理多变量时间序列数据？
    是否明确说明了涉及多个变量的时间序列预测任务？
    预训练模型：
    论文中是否使用了预训练的方法或模型？
    描述预训练过程，包括使用的数据集、模型架构以及预训练的目标。
    微调策略：
    是否在特定任务或数据集上对预训练模型进行了微调？
    说明微调的具体方法、参数调整以及所取得的成果。
    模型架构与创新：
    论文是否提出了新的模型架构或改进现有模型以适应多变量时间序列预测？
    这些创新是否与预训练和微调策略相关联？
    """
    paper_spider_filter()
    

