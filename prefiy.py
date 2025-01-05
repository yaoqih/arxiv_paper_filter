import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from datetime import datetime
import os
from openai import OpenAI
from tqdm import tqdm
import pandas as pd
import numpy as np
from itertools import combinations
import time

config = json.load(open('config.json', 'r'))
client = OpenAI(
            api_key=config['api_key'],
            base_url="https://api.chatanywhere.tech"
        )
class PaperCategoryManager:
    def __init__(self):
        self.standard_keys = {
            "Research_Type":"**Research_Type**: What type of research is this paper? For example, theoretical research, methodological research, applied research, survey/review research, etc.",
            "Technical_Domain":"**Technical_Domain**: What is the core technical field of the paper? For example, computer vision, natural language processing, reinforcement learning, etc.", 
            "Methodology":"**Methodology**: What are the main technical methods or model architectures used in the paper? For example, supervised learning, Generative Adversarial Networks (GAN), Convolutional Neural Networks (CNN), Transformer, etc.",
            "Evaluation_Metrics":"**Evaluation_Metrics**: What performance evaluation metrics were used in the paper? For example, accuracy, inference speed, computational efficiency, etc.",
            "Application_Scenarios":"**Application_Scenarios**: In which specific applications does this paper's research contribute? For example, healthcare, autonomous driving, finance, industry, etc.",
            "Innovation_Aspects":"**Innovation_Aspects**: In what aspects does the paper present innovations or breakthroughs? For example, theoretical innovation, algorithmic innovation, application innovation, etc.",
            "Experimental_Design":"**Experimental_Design**: What experimental design methods were used in the paper? For example, which datasets were used, whether ablation studies were conducted, comparisons with other methods, etc.     "
        }
        self.raw_data = {}
        self.merged_data = {}
        self.duplicates = {}
        self.processing_history = []
        self.processed_data = {}
    def load_data(self, data: Dict) -> None:
        """加载原始数据"""
        self.raw_data = data
        
    def merge_categories(self) -> None:
        """合并同层次的关键词"""
        merged_dict = defaultdict(set)
        for paper_url, categories in self.raw_data.items():
            for key, values in categories.items():
                merged_dict[key].update(values)
        
        self.merged_data = {k: list(v) for k, v in merged_dict.items()}
        
    def find_duplicates(self) -> None:
        """查找不同层次间的重复关键词"""
        all_terms = defaultdict(list)
        for category, terms in self.merged_data.items():
            for term in terms:
                all_terms[term].append(category)
        self.duplicates = {
            term: categories 
            for term, categories in all_terms.items() 
            if len(categories) > 1
        }
        
    def process_duplicate(self, term: str, keep_category: str) -> None:
        """
        处理单个重复项
        term: 重复的关键词
        keep_category: 保留该关键词的类别
        """
        if term not in self.duplicates:
            raise ValueError(f"Term '{term}' is not in duplicates")
            
        affected_categories = self.duplicates[term]
        if keep_category not in affected_categories:
            raise ValueError(f"Category '{keep_category}' is not valid for term '{term}'")
            
        # 记录处理历史
        process_record = {
            'timestamp': datetime.now().isoformat(),
            'term': term,
            'keep_category': keep_category,
            'removed_from': [cat for cat in affected_categories if cat != keep_category]
        }
        self.processing_history.append(process_record)
        # 更新数据
        self.processed_data = self.merged_data.copy()
        for category in affected_categories:
            if category != keep_category:
                self.processed_data[category].remove(term)
                
    def save_results(self, output_dir: str) -> None:
        """保存所有结果到文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存原始数据
        with open(os.path.join(output_dir, 'raw_data.json'), 'w', encoding='utf-8') as f:
            json.dump(self.raw_data, f, indent=2, ensure_ascii=False)
            
        # 保存合并后的数据
        with open(os.path.join(output_dir, 'merged_data.json'), 'w', encoding='utf-8') as f:
            json.dump(self.merged_data, f, indent=2, ensure_ascii=False)# 保存重复项数据
        with open(os.path.join(output_dir, 'duplicates.json'), 'w', encoding='utf-8') as f:
            json.dump(self.duplicates, f, indent=2, ensure_ascii=False)
            
        # 保存处理历史
        with open(os.path.join(output_dir, 'processing_history.json'), 'w', encoding='utf-8') as f:
            json.dump(self.processing_history, f, indent=2, ensure_ascii=False)
            
        # 保存处理后的数据
        if self.processed_data:
            with open(os.path.join(output_dir, 'processed_data.json'), 'w', encoding='utf-8') as f:
                json.dump(self.processed_data, f, indent=2, ensure_ascii=False)
    def load_results(self, output_dir: str) -> None:
        """加载所有结果"""
        with open(os.path.join(output_dir, 'raw_data.json'), 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
            
        with open(os.path.join(output_dir, 'merged_data.json'), 'r', encoding='utf-8') as f:
            self.merged_data = json.load(f)
            
        with open(os.path.join(output_dir, 'duplicates.json'), 'r', encoding='utf-8') as f:
            self.duplicates = json.load(f)
            
        with open(os.path.join(output_dir, 'processing_history.json'), 'r', encoding='utf-8') as f:
            self.processing_history = json.load(f)
            
        with open(os.path.join(output_dir, 'processed_data.json'), 'r', encoding='utf-8') as f:
            self.processed_data = json.load(f)
    def get_status_report(self) -> Dict:
        """获取当前处理状态的报告"""
        return {
            'total_categories': len(self.merged_data),
            'total_duplicates': len(self.duplicates),
            'processed_duplicates': len(self.processing_history),
            'remaining_duplicates': len(self.duplicates) - len(self.processing_history)
        }
    def process_duplicates(self) -> None:
        """处理所有重复项"""
        for term, categories in tqdm(self.duplicates.items(),'process_duplicates'):
            # 这里可以根据具体规则选择保留的类别
            keep_category = self.judge_duplicate_categores(term, categories)
            if keep_category:
                self.process_duplicate(term, keep_category)
    def judge_duplicate_categores(self,keyword, categories: List[str]) -> str:
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
    def find_similar_keywords(self, data: dict, threshold: float = 0.77) -> List[str]:
        """查找与给定关键词相似的关键词"""
        inverse_index = self.build_inverse_index(data)
        keywords_all=set(inverse_index.keys())
        embeddings = self.embeding_keywords(keywords_all)
        similarity_list = self.calculate_similarities(embeddings) 
        while similarity_list and similarity_list[0][0] > threshold:
            sim = similarity_list.pop(0)
            keep_category = self.judge_simiar_keywords(sim[1], sim[2],inverse_index)
            if keep_category==1:
                self.process_similar_keywords(sim[1], sim[2])
            elif keep_category==2:
                self.process_similar_keywords(sim[2], sim[1])
            index=0
            while index<len(similarity_list):
                if sim[1] in similarity_list[index] or sim[2] in similarity_list[index]:
                    similarity_list.pop(index)
                else:
                    index+=1
            
    def process_similar_keywords(self,keep_keyword: str, remove_keyword: str) -> None:
        """处理相似关键词"""
        # 记录处理历史
        process_record = {
            'timestamp': datetime.now().isoformat(),
            'keep_keyword': keep_keyword,
            'remove_keyword': remove_keyword,
        }
        self.processing_history.append(process_record)
        # 更新数据
        self.processed_data = self.merged_data.copy()
        for category, terms in self.processed_data.items():
            if remove_keyword in terms:
                self.processed_data[category].remove(remove_keyword)
        df_save=pd.read_csv('output_keywords.csv')
        df_save['keywords'] = df_save['keywords'].apply(lambda x: '|'.join([keep_keyword if k == remove_keyword else k for k in x.split('|')]))
        df_save.to_csv('output_keywords_perify.csv',index=False)

    def judge_simiar_keywords(self,keyword1,keyword2,inverse_index) -> int:
        """判断相似关键词中哪个类别是正确的"""
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
                    类别1：{inverse_index[keyword1]}
                    关键词2：{keyword2}
                    类别2：{inverse_index[keyword2]}

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

    def embeding_keywords(self,keywords:List[str])->Dict[str, List[float]]:
        """嵌入关键词"""
        embeddings = client.embeddings.create(
            model="text-embedding-3-large",
            input=keywords
        )
        return {keyword: embedding.embedding for keyword, embedding in zip(keywords, embeddings.data)}
    def calculate_similarities(self, embeddings):
        # 提取所有的词向量
        keywords = list(embeddings.keys())
        vectors = np.array([embeddings[keyword] for keyword in keywords])
        
        # 计算余弦相似度矩阵
        dot_product = np.dot(vectors, vectors.T)  # 矩阵的点积
        norms = np.linalg.norm(vectors, axis=1)  # 计算每个向量的L2范数
        similarity_matrix = dot_product / (norms[:, np.newaxis] * norms)  # 计算余弦相似度

        # 生成相似度结果并排序
        similarity = [
            [similarity_matrix[i, j], keywords[i], keywords[j]]
            for i in range(len(keywords))
            for j in range(i + 1, len(keywords))  # 只考虑上三角矩阵（避免重复计算）
        ]
        
        # 按相似度降序排序
        return sorted(similarity, key=lambda x: x[0], reverse=True)
    def cosine_similarity(self,vec1: List[float], vec2: List[float]) -> float:
        """计算两个向量之间的余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)
    def build_inverse_index(self,data:dict) -> Dict[str, Set[str]]:
        """构建关键词的倒排索引"""
        inverse_index = defaultdict(set)
        for category, terms in data.items():
            for term in terms:
                inverse_index[term].add(category)
        return inverse_index
def prefiy_keywords_write():
    # 示例数据
    sample_data =json.loads( open('key_words.json', 'r').read())

    # 创建管理器实例
    manager = PaperCategoryManager()
    
    # 加载数据
    manager.load_data(sample_data)
    
    # 合并类别
    manager.merge_categories()
    
    # 查找重复项
    manager.find_duplicates()
    
    # 打印状态报告
    print("初始状态报告:", manager.get_status_report())# 处理重复项示例（如果有的话）
    manager.process_duplicates()
    manager.save_results('output')# 打印最终状态报告
    print("最终状态报告:", manager.get_status_report())
    data_save=json.load(open(config['config_file'], 'r',encoding='utf-8'))
    data_save['keyword_refinement']=True
    data_save['last_updated'] = datetime.now().isoformat()
    with open(config['config_file'], 'w', encoding='utf-8') as f:
        json.dump(data_save, f, ensure_ascii=False, indent=2)
# 使用示例
if __name__ == "__main__":
    # 示例数据
    sample_data =json.loads( open('key_words.json', 'r').read())

    # 创建管理器实例
    manager = PaperCategoryManager()
    manager.load_results('output')
    
    # 加载数据
    # manager.load_data(sample_data)
    
    # # 合并类别
    # manager.merge_categories()
    
    # # 查找重复项
    # manager.find_duplicates()
    
    # 打印状态报告
    print("初始状态报告:", manager.get_status_report())# 处理重复项示例（如果有的话）
    # for term, categories in manager.duplicates.items():
    #     # 这里可以根据具体规则选择保留的类别
    #     keep_category = categories[0]  # 示例：保留第一个类别
    #     manager.process_duplicate(term, keep_category)
    # manager.process_duplicates()
    manager.find_similar_keywords(manager.processed_data)
    # 保存所有结果
    manager.save_results('output')# 打印最终状态报告
    print("最终状态报告:", manager.get_status_report())
