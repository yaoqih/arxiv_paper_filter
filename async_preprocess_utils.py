import json
from openai import OpenAI
from arxiv_search import ArxivScraper
config = json.load(open('config.json', 'r'))
client = OpenAI(
            api_key=config['api_key'],
            base_url="https://api.chatanywhere.tech"
        )
scraper = ArxivScraper()
def intention_analysis(paper_type) -> str:
        """判断重复项中哪个类别是正确的"""
        try:
            response = client.chat.completions.create(
                model="gpt-4o",  # 或其他适合的模型
                messages=[
                    {"role": "system", "content": f"""
你是一位专业的学术文献检索专家。我想检索"[{paper_type}]"的论文。请：

1. 分析检索意图：
- 推断我可能感兴趣的具体研究方向
- 识别潜在的研究问题或目标
- 列出可能相关的子领域

2. 相关性判断标准：
- 列出判断论文相关性的关键指标
- 提供筛选文献的具体建议
- 指出需要重点关注的章节

      
    ```

                    """},
                ],
                temperature=0.7,
            )       
            # 提取返回的JSON内容
            return response.choices[0].message.content.replace('```json\n', '').replace('```', '')
        except Exception as e:
            print(f"Error processing abstract: {str(e)}")
            return None
def intention_analysis_write(paper_type):
    text=intention_analysis(paper_type)
    save_data = json.load(open(config['config_file'],'r',encoding='utf-8'))
    save_data['search_config']['intent']=text
    json.dump(save_data, open(config['config_file'], "w",encoding='utf-8'))
def generate_search_keywords(search_keywords)->str:
    """生成搜索关键词"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # 或其他适合的模型
            messages=[
                {"role": "system", "content": f"""
Given my research interest: {search_keywords}
Generate 4-7 different english search keyword combinations for arXiv. Each combination should:
- Use simple AND operators
- Split complex OR conditions into separate lines
- Focus on different aspects of the topic
- Be specific enough to return relevant results
- For the results to be parsed correctly, divide by the list of keywords in the following example format and do not reply to anything else

Output format:
"keyword1" AND "keyword2"
"keyword1" AND "keyword3"
"keyword3" AND "keyword4" AND "keyword5"
"""},
                {"role": "user", "content": search_keywords}
            ],
            temperature=0.7,
        )       
        # 提取返回的JSON内容
        return response.choices[0].message.content.replace('```json\n', '').replace('```', '')
    except Exception as e:
        print(f"Error processing abstract: {str(e)}")
        return None
def generate_search_keywords_write(search_keywords):
    text=generate_search_keywords(search_keywords)
    save_data = json.load(open(config['config_file'],'r',encoding='utf-8'))
    save_data['search_config']['keywords']=text
    json.dump(save_data, open(config['config_file'], "w",encoding='utf-8'))

def write_search_num(search_keywords):
    save_data = json.load(open(config['config_file'],'r',encoding='utf-8'))
    search_num_dict={}
    search_num_all=0
    for keyword in search_keywords:
        search_num_dict[keyword]=scraper.get_result_num(keyword.strip())
        search_num_all+=search_num_dict[keyword]
    save_data['search_preview']['categories']=search_num_dict
    save_data['search_preview']['total_count']=search_num_all
    json.dump(save_data, open(config['config_file'], "w",encoding='utf-8'))