import arxiv
import os
import time
from typing import List, Dict
from openai import OpenAI
import logging
from tqdm import tqdm
import pandas as pd
import json
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
                    {"role": "system", "content": "你是一个专业的论文评估助手，请严格按照给定标准评估论文并给出合理的分数。"},
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

    def search_and_filter_papers(self, keywords: List[str], criteria: str, max_results: int = 100, 
                               date_limit: str = "2023-01-01", score_threshold: float = 6.0):
        """
        搜索和筛选论文
        """
        client = arxiv.Client()

        for keyword in keywords:
            logging.info(f"正在处理关键词: {keyword}")
            try:
                query = f'all:"{keyword}" AND submittedDate:[{date_limit}0000 TO 99991231235959]'
                
                search = arxiv.Search(
                    query=query,
                    max_results=max_results,
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

def main():
    # 配置参数
    api_key = config['api_key']
    keywords = [
        "narrative generation",
        "story generation",
        "storyboard generation",
        "story planning",
        "Screenplay Generation"
    ]

    criteria = """
    请判断一下这篇论文是否与脚本/故事生成有关，特别是着重考虑以下几个方面：
    需要检查的关键领域：
    1. 故事生成与讲故事
    - 叙事生成
    - 情节发展
    - 故事结构创建
    - 角色发展
    - 讲故事技巧
    - 故事弧生成
    - 交互式叙事
    - 故事推进

    2. 故事可视化
    - 场景描述生成
    - 视觉叙事创作
    - 故事板生成
    - 场景到脚本转换
    - 视觉叙事
    - 场景构图

    3. 相关技术术语：
    - 叙事AI
    - 故事规划
    - 情节图
    - 故事语法
    - 序列事件生成
    - 叙事流
    - 故事连贯性
    - 故事世界构建

    4. 应用领域
    - 编剧
    - 创意写作
    - 游戏叙事
    - 视觉小说
    """

    # 创建过滤器实例并运行
    paper_filter = ArxivPaperFilter(api_key)
    paper_filter.search_and_filter_papers(
        keywords=keywords,
        criteria=criteria,
        date_limit="2023-01-01",
        score_threshold=6.0
    )

if __name__ == "__main__":
    main()

