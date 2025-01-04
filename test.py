# query = f'all:{"+AND+".join([f"{_}"for _ in keyword.split(" ")])}'
import arxiv
from tqdm import tqdm
import json
max_results = None
client = arxiv.Client()
keyword = 'multivariate time series forecasting pre-training fine-tuning' # 移除引号和加号
query = 'all:' + '(' + keyword + ')'  # 添加括号提高精确度

search = arxiv.Search(
    query=query,
    # max_results=100,  # 设置具体数值
    sort_by=arxiv.SortCriterion.Relevance,  # 使用相关性排序
    sort_order=arxiv.SortOrder.Descending
)
result={}
for paper in tqdm(client.results(search)):
    result[paper.entry_id] = {
                            "id": paper.entry_id,
                            "title": paper.title,
                            "abstract": paper.summary,
                            "authors": ', '.join([author.name for author in paper.authors]),
                            "published": paper.published.strftime("%Y-%m-%d"),
                            "url": paper.pdf_url,
                            # "score": evaluation["score"],
                            # "evaluation": evaluation["reason"],
                            "keyword": ''
                        }
json.dump(result, open("output.json", "w"))