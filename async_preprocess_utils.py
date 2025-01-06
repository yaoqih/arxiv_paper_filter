import json
from openai import OpenAI
from arxiv_search import ArxivScraper
from progress import ProgressManager
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import time
from typing import Dict, List
progress_manager=ProgressManager('data\progress_saving.json')

secret = json.load(open('data/secret.json', 'r'))
client = OpenAI(
            api_key=secret['api_key'],
            base_url=secret['base_url']
        )
scraper = ArxivScraper()
def generate_intent(query:str,write=False) -> str:
    """生成搜索意图"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {"role": "system", "content": f"""
                    我想要在人工智能领域查询相关的学术论文。想要查询{query}的论文。由于对该领域了解有限，我的搜索关键词可能过于广泛。请帮助我细化和补充这个检索意图，使其更为准确和全面。请考虑以下方面：
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
        text=response.choices[0].message.content.replace('```json\n', '').replace('```', '').strip()
        if write:
            progress_manager.update_search_config(intent=text)
        return text
    except Exception as e:
        print(f"Error processing abstract: {str(e)}")
        return None
def generate_keywords(query:str,write=False)->str:
    """生成搜索关键词"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # 或其他适合的模型
            messages=[
                {"role": "system", "content": f"""
        I am a beginner in the field of artificial intelligence and want to find some paper on {query}, But my knowledge of this field is limited, I can't write an accurate, specific, and comprehensive search term
        Generate 4-7 different english search keyword combinations for arXiv. Each combination should:

        - Split complex OR conditions into separate lines
        - Focus on different aspects of the topic
        - Be specific enough to return relevant results
        - For the results to be parsed correctly, divide by the list of keywords in the following example format and do not reply to anything else

        Output format:
        keyword1  keyword2
        keyword1  keyword3
        keyword3  keyword4  keyword5
        """},
            ],
            temperature=0.7,
        )       
        text=response.choices[0].message.content.replace('```json\n', '').replace('```', '').strip()
        if write:
            progress_manager.update_search_config(keywords=text)
        return text
    except Exception as e:
        print(f"Error processing abstract: {str(e)}")
        return None
def generate_criterion(query:str,write=False)->str:
    """生成论文判断标准"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # 或其他适合的模型
            messages=[
                {"role": "system", "content": f"""

请你作为一个研究领域专家，帮助我制定{query}论文筛选标准。

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
            ],
            temperature=0.7,
        )       
        text=response.choices[0].message.content.replace('```json\n', '').replace('```', '').strip()
        if write:
            progress_manager.update_search_config(criterion=text)
        return text
    except Exception as e:
        print(f"Error processing abstract: {str(e)}")
        return None

def extract_label(paper):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 或其他适合的模型
            messages=[
                {"role": "system", "content": """
You are a knowledgeable paper reviewer with clear insights.
Please read the title and abstract of the following paper, and extract the most representative keywords based on the paper's research content, methodology, contributions, and application scenarios. Ensure these keywords are specific and accurate, capable of uncovering the paper's characteristics, focus, and innovations across different dimensions. The keywords should cover the following dimensions:

1. **Research_Type**: What type of research is this paper? For example, theoretical research, methodological research, applied research, survey/review research, etc.
2. **Technical_Domain**: What is the core technical field of the paper? For example, computer vision, natural language processing, reinforcement learning, etc.
3. **Methodology**: What are the main technical methods or model architectures used in the paper? For example, supervised learning, Generative Adversarial Networks (GAN), Convolutional Neural Networks (CNN), Transformer, etc.
4. **Evaluation_Metrics**: What performance evaluation metrics were used in the paper? For example, accuracy, inference speed, computational efficiency, etc.
5. **Application_Scenarios**: In which specific applications does this paper's research contribute? For example, healthcare, autonomous driving, finance, industry, etc.
6. **Innovation_Aspects**: In what aspects does the paper present innovations or breakthroughs? For example, theoretical innovation, algorithmic innovation, application innovation, etc.
7. **Experimental_Design**: What experimental design methods were used in the paper? For example, which datasets were used, whether ablation studies were conducted, comparisons with other methods, etc.

The final output should be returned in JSON format as follows:

{
    "Research_Type": [],
    "Technical_Domain": [],
    "Methodology": [],
    "Evaluation_Metrics": [],
    "Application_Scenarios": [],
    "Innovation_Aspects": [],
    "Experimental_Design": []
}

               """},
                {"role": "user", "content": f"""
Title: {paper['title']}
Abstract: {paper['abstract']}

Please use your insight to extract the most representative keywords for each level. Don't be too general, but please respond using our agreed JSON format so my parsing won't go wrong.

""" }
            ],
            temperature=0.7,
        )       
        # 提取返回的JSON内容
        result = json.loads(response.choices[0].message.content.replace('```json\n', '').replace('```', ''))
        return result
    except Exception as e:
        print(f"Error processing abstract: {str(e)}")
        return None
def evaluate_paper(paper_info: str, criteria: str) -> dict:
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

        注意：
        - 分数标准：
            10分：完全符合核心要求，直接相关
            7-9分：高度相关，但可能不是核心主题
            4-6分：部分相关，有一些相关元素
            1-3分：较少相关性
            0分：完全不相关
        
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是一个专业的论文评估助手，请严格按照给定标准评估论文并给出合理的分数。请严格按照约定的格式回复，这样才能保证解析不会出错"},
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
        print(f"GPT评估出错: {str(e)}")
        return {"score": 0, "reason": f"评估错误: {str(e)}"}

def judge_simiar_keywords(keyword1,keyword2,inverse_index) -> int:
    """判断相似关键词中哪个类别是正确的"""
    standard_keys = {
            "Research_Type":"**Research_Type**: What type of research is this paper? For example, theoretical research, methodological research, applied research, survey/review research, etc.",
            "Technical_Domain":"**Technical_Domain**: What is the core technical field of the paper? For example, computer vision, natural language processing, reinforcement learning, etc.", 
            "Methodology":"**Methodology**: What are the main technical methods or model architectures used in the paper? For example, supervised learning, Generative Adversarial Networks (GAN), Convolutional Neural Networks (CNN), Transformer, etc.",
            "Evaluation_Metrics":"**Evaluation_Metrics**: What performance evaluation metrics were used in the paper? For example, accuracy, inference speed, computational efficiency, etc.",
            "Application_Scenarios":"**Application_Scenarios**: In which specific applications does this paper's research contribute? For example, healthcare, autonomous driving, finance, industry, etc.",
            "Innovation_Aspects":"**Innovation_Aspects**: In what aspects does the paper present innovations or breakthroughs? For example, theoretical innovation, algorithmic innovation, application innovation, etc.",
            "Experimental_Design":"**Experimental_Design**: What experimental design methods were used in the paper? For example, which datasets were used, whether ablation studies were conducted, comparisons with other methods, etc.     "
        }
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 或其他适合的模型
            messages=[
                {"role": "system", "content": f"""
                我从论文中提取了一些关键词，其中下面两个关键词非常相似。
                请根据以下两个关键词及其所属类别来判断应该保存哪个关键词，或者是否保留两个关键词。以下是你的判断标准：

                1. **语义差异**：这两个关键词在语义上是否存在明显的差异？如果存在显著差异，则两个关键词都可以保留。
                2. **类别相关性**：两个关键词所属的类别是否相同？如果是相同类别且没有明显的语义差异，可能只保留一个关键词。
                3. **冗余度**：这两个关键词是否有冗余，或是一个关键词能有效包含另一个关键词的含义？如果有冗余，保留更为准确、全面的关键词。
                                    """},
                                    {"role": "user", "content": f"""
                请根据上述标准决定以下关键词是否都保留，或者保留其中一个：
                关键词1：{keyword1}
                类别1：{standard_keys[inverse_index[keyword1]]}
                关键词2：{keyword2}
                类别2：{standard_keys[inverse_index[keyword2]]}

                根据这些信息，输出以下判断：
                - "保留关键词1"
                - "保留关键词2"
                - "保留两个关键词"
                请仅回答上述三个选项中的一个。
                """}

            ],
            temperature=0.7,
        )
        # 提取返回的JSON内容
        if '保留关键词1' in response.choices[0].message.content:
            return 1
        elif '保留关键词2' in response.choices[0].message.content:
            return 2
        else:
            return 0
    except Exception as e:
        print(f"Error processing abstract: {str(e)}")
        return None

def judge_duplicate_categores(keyword, categories: List[str]) -> str:
    """判断重复项中哪个类别是正确的"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 或其他适合的模型
            messages=[
                {"role": "system", "content": f"""
                You are a knowledgeable mentor with clear and accurate insights, and you are also enthusiastic about helping others. I am a student who is just starting out in the field of artificial intelligence. I want to categorize research keywords into different research levels, but there is one keyword that I am unsure which level it belongs to. Below are the levels I have preset, and I hope you can help me determine which level this keyword should be placed in.    ```
                1. **Research_Type**: What type of research is this paper? For example, theoretical research, methodological research, applied research, survey/review research, etc.
                2. **Technical_Domain**: What is the core technical field of the paper? For example, computer vision, natural language processing, reinforcement learning, etc.
                3. **Methodology**: What are the main technical methods or model architectures used in the paper? For example, supervised learning, Generative Adversarial Networks (GAN), Convolutional Neural Networks (CNN), Transformer, etc.
                4. **Evaluation_Metrics**: What performance evaluation metrics were used in the paper? For example, accuracy, inference speed, computational efficiency, etc.
                5. **Application_Scenarios**: In which specific applications does this paper's research contribute? For example, healthcare, autonomous driving, finance, industry, etc.
                6. **Innovation_Aspects**: In what aspects does the paper present innovations or breakthroughs? For example, theoretical innovation, algorithmic innovation, application innovation, etc.
                7. **Experimental_Design**: What experimental design methods were used in the paper? For example, which datasets were used, whether ablation studies were conducted, comparisons with other methods, etc.            
                    ```
                Feel free to use your insights boldly, but when replying, please only tell me the final hierarchical level name (choose one from the following:
                {categories}
                ), so that my parser won't make mistakes.

                """},
                {"role": "user", "content": f"我有一个关键词{keyword}，它可能属于以下类别：{', '.join(categories)}，请帮我判断它应该属于哪个类别。"}
            ],
            temperature=0.7,
        )       
        # 提取返回的JSON内容
        return response.choices[0].message.content.replace('```json\n', '').replace('```', '')
    except Exception as e:
        print(f"Error processing abstract: {str(e)}")
        return None

def embeding_keywords(keywords: List[str]) -> Dict[str, List[float]]:
    """嵌入关键词，支持批量处理超过2000个关键词的情况
    
    Args:
        keywords: 关键词列表
    
    Returns:
        包含关键词及其嵌入向量的字典
    """
    BATCH_SIZE = 2000  # OpenAI API 的单次请求限制
    result = {}
    
    # 将关键词列表分批处理
    for i in range(0, len(keywords), BATCH_SIZE):
        batch = keywords[i:i + BATCH_SIZE]
        embeddings = client.embeddings.create(
            model="text-embedding-3-large",
            input=batch,
            dimensions=256
        )
        # 将当前批次的结果添加到总结果中
        batch_result = {
            keyword: embedding.embedding 
            for keyword, embedding in zip(batch, embeddings.data)
        }
        result.update(batch_result)
    
    return result


def get_result_num_with_retry(keyword: str, max_retries: int = 3, delay: int = 2) -> int:
    """带重试机制的获取搜索结果数量函数"""
    for attempt in range(max_retries):
        try:
            return scraper.get_result_num(keyword.strip())
        except Exception as e:
            if attempt == max_retries - 1:  # 最后一次重试
                print(f"获取关键词 '{keyword}' 的搜索结果失败: {str(e)}")
                return 0
            time.sleep(delay)  # 重试前等待
    return 0

def process_keyword(keyword: str, progress_manager) -> tuple:
    """处理单个关键词的函数"""
    result = get_result_num_with_retry(keyword)
    return keyword, result

def write_search_num(keywords: List[str]) -> None:
    """并行处理关键词搜索数量统计"""
    search_num_dict: Dict[str, int] = {}
    search_num_all: int = 0
    
    # 创建偏函数，固定 progress_manager 参数
    process_func = partial(process_keyword, progress_manager=progress_manager)
    
    # 使用进程池并行处理
    with ProcessPoolExecutor() as executor:
        results = executor.map(process_func, keywords)
        
        # 收集结果
        for keyword, count in results:
            search_num_dict[keyword] = count
            search_num_all += count
    
    # 更新进度
    progress_manager.update_search_preview(
        total_count=search_num_all,
        categories=search_num_dict
    )