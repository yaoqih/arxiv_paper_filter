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
                        我想要在人工智能领域查询相关的学术论文。想要查询{paper_type}的论文。由于对该领域了解有限，我的搜索关键词可能过于广泛。请帮助我细化和补充这个检索意图，使其更为准确和全面。请考虑以下方面：
                        具体主题或问题：有哪些特定的子领域或问题是我可能感兴趣的？
                        相关技术或方法：有哪些相关的技术或方法可以与这个主题关联？
                        应用场景：在哪些实际应用场景中，这个主题可能会被应用？
                        前沿研究动态：目前在这个领域有哪些前沿研究或新兴趋势？
                        根据以上这些方面，请帮助我形成一个更加具体和全面的检索意图。
                        不用给出具体的检索关键词，只需要一个更加具体和全面的检索意图即可。谢谢！
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
I am a beginner in the field of artificial intelligence and want to find some paper on {search_keywords}, But my knowledge of this field is limited, I can't write an accurate, specific, and comprehensive search term
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
def generate_criterion(search_keywords)->str:
    """生成论文判断标准"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # 或其他适合的模型
            messages=[
                {"role": "system", "content": f"""

请你作为一个研究领域专家，帮助我制定{search_keywords}论文筛选标准。

第一步：请你先仔细分析我的检索需求
1. 请复述我的检索需求要点
2. 请指出这个检索需求中可能隐含的研究方向、技术路线或应用场景
3. 请列出与这个检索需求相关的关键概念和术语

第二步：基于上述分析，请帮我：
1. 从研究目标维度，列出判断标准
2. 从研究方法维度，列出判断标准
3. 从实验/应用场景维度，列出判断标准
4. 从创新点维度，列出判断标准

第三步：请将这些标准整理成一个结构化的评估框架，包括：
1. 必要条件（论文必须满足的标准）
2. 充分条件（论文最好满足的标准）
3. 排除条件（遇到这些情况应该排除的标准）

---

请根据这些要求，生成一个详细的判断标准。

"""},
                # {"role": "user", "content": search_keywords}
            ],
            temperature=0.7,
        )       
        # 提取返回的JSON内容
        return response.choices[0].message.content.replace('```json\n', '').replace('```', '')
    except Exception as e:
        print(f"Error processing abstract: {str(e)}")
        return None
def generate_criterion_write(search_keywords):
    text=generate_criterion(search_keywords)
    save_data = json.load(open(config['config_file'],'r',encoding='utf-8'))
    save_data['search_config']['criterion']=text
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