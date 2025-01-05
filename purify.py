import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from datetime import datetime
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from async_preprocess_utils import judge_simiar_keywords,judge_duplicate_categores,embeding_keywords
from progress import ProgressManager
progress_manager=ProgressManager('data\progress_saving.json')

class PaperCategoryManager:
    def __init__(self,label_file:str='data/labels.json',out_dir='./data'):
        self.raw_data = json.loads(open(label_file, 'r',encoding='utf-8').read())
        self.duplicates = {}
        self.duplicates_save={}
        self.similar_save={}
        self.embeding_save={}
        self.processing_history = []
        self.processed_data = {}
        self.out_dir=out_dir
        self.find_duplicates()
        self.load_results()
        
    def merge_categories(self) -> None:
        """合并同层次的关键词"""
        merged_dict = defaultdict(set)
        for paper_url, categories in self.raw_data.items():
            for key, values in categories.items():
                merged_dict[key].update(values)
        
        return {k: list(v) for k, v in merged_dict.items()}
        
    def find_duplicates(self) -> None:
        """查找不同层次间的重复关键词"""
        all_terms = defaultdict(list)
        self.processed_data = self.merge_categories()
        for category, terms in self.processed_data.items():
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
        for category in affected_categories:
            if category != keep_category:
                self.processed_data[category].remove(term)
                
    def save_results(self,duplicates_save=False,similar_save=False,embeding_save=False,processed_data=False,all_save=False) -> None:
        """保存所有结果到文件"""
        if all_save or duplicates_save:
            with open(os.path.join(self.out_dir, 'duplicates_save.json'), 'w', encoding='utf-8') as f:
                json.dump(self.duplicates_save, f, indent=2, ensure_ascii=False)
                
        if all_save or similar_save:
            with open(os.path.join(self.out_dir, 'similar_save.json'), 'w', encoding='utf-8') as f:
                json.dump(self.similar_save, f, indent=2, ensure_ascii=False)
            
        if all_save or embeding_save:
            with open(os.path.join(self.out_dir, 'embeding_save.json'), 'w', encoding='utf-8') as f:
                json.dump(self.embeding_save, f, indent=2, ensure_ascii=False)
        if all_save or processed_data:
            with open(os.path.join(self.out_dir, 'labels_purify.json'), 'w', encoding='utf-8') as f:
                json.dump(self.processed_data, f, indent=2, ensure_ascii=False)
        
    def load_results(self) -> None:
        """加载所有结果"""
        if os.path.exists(os.path.join(self.out_dir, 'duplicates_save.json')):
            with open(os.path.join(self.out_dir, 'duplicates_save.json'), 'r', encoding='utf-8') as f:
                self.duplicates_save = json.load(f)
        if os.path.exists(os.path.join(self.out_dir, 'similar_save.json')):
            with open(os.path.join(self.out_dir, 'similar_save.json'), 'r', encoding='utf-8') as f:
                self.similar_save = json.load(f)
        if os.path.exists(os.path.join(self.out_dir, 'embeding_save.json')):
            with open(os.path.join(self.out_dir, 'embeding_save.json'), 'r', encoding='utf-8') as f:
                self.embeding_save = json.load(f)

    def process_duplicates(self) -> None:
        """处理所有重复项"""
        for term, categories in tqdm(self.duplicates.items(),'process_duplicates'):
            # 这里可以根据具体规则选择保留的类别
            if term in self.duplicates_save and self.duplicates_save[term] in categories:
                keep_category = self.duplicates_save[term]
            else:
                keep_category = judge_duplicate_categores(term, categories)
            if keep_category:
                self.process_duplicate(term, keep_category)
                self.duplicates_save[term] = keep_category
                self.save_results(duplicates_save=True)

    def find_similar_keywords(self, data: dict, threshold: float = 0.77) -> List[str]:
        """查找与给定关键词相似的关键词"""
        inverse_index = self.build_inverse_index(data)
        keywords_all=set(inverse_index.keys())
        need_embeding_keywords=keywords_all-set(self.embeding_save.keys())
        if need_embeding_keywords:
            embeddings = embeding_keywords(need_embeding_keywords)
            self.embeding_save.update(embeddings)
        self.save_results(embeding_save=True)
        similarity_list = self.calculate_similarities(self.embeding_save) 
        while similarity_list and similarity_list[0][0] > threshold:
            sim = similarity_list.pop(0)
            key1 = f'{sim[1]}|{sim[2]}'
            key2 = f'{sim[2]}|{sim[1]}'
            
            # 从缓存中查找已存在的相似度判断结果
            cached_result = self.similar_save.get(key1) or self.similar_save.get(key2)
            
            if cached_result is not None:
                keep_category = cached_result
            else:
                keep_category = judge_simiar_keywords(sim[1], sim[2], inverse_index)
            if keep_category==1:
                self.process_similar_keywords(sim[1], sim[2])
            elif keep_category==2:
                self.process_similar_keywords(sim[2], sim[1])
            self.similar_save[f'{sim[1]}|{sim[2]}'] = keep_category
            self.save_results(similar_save=True)
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
        for category, terms in self.processed_data.items():
            if remove_keyword in terms:
                self.processed_data[category].remove(remove_keyword)
        df_save=pd.read_csv(os.path.join(self.out_dir, 'paper_labeled.csv'))
        df_save['labels'] = df_save['labels'].apply(lambda x: '|'.join([keep_keyword if k == remove_keyword else k for k in x.split('|')]))
        df_save.to_csv(os.path.join(self.out_dir, 'labeled_purify.csv'),index=False)

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
    def build_inverse_index(self,data:dict) -> Dict[str, Set[str]]:
        """构建关键词的倒排索引"""
        inverse_index = defaultdict(set)
        for category, terms in data.items():
            for term in terms:
                inverse_index[term]=category
        return inverse_index
def prefiy_keywords_write():
    manager = PaperCategoryManager()
    manager.process_duplicates()
    manager.find_similar_keywords(manager.processed_data)
    manager.save_results(processed_data=True)
    progress_manager.set_keyword_status(purify=True)
# 使用示例
if __name__ == "__main__":
    prefiy_keywords_write()
