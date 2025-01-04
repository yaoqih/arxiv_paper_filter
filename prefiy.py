import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from datetime import datetime
import os
from openai import OpenAI
from tqdm import tqdm

config = json.load(open('config.json', 'r'))
client = OpenAI(
            api_key=config['api_key'],
            base_url="https://api.chatanywhere.tech"
        )
class PaperCategoryManager:
    def __init__(self):
        self.standard_keys = [
            "Discipline",
            "Research_Area", 
            "Research_Topic",
            "Research_Method",
            "Specific_Model",
            "Model_Variants",
            "Specific_Technical_Details"
        ]
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
    **Level 1 (Discipline):**

    *   This is currently the broadest level, representing major academic fields. It's recommended to keep it concise, emphasizing the independence and inclusiveness of each discipline. No significant changes are needed.
    *   **Example Keywords:** Computer Science, Mathematics, Statistics, Physics, Medicine, Psychology, etc.

    **Level 2 (Research_Area):**

    *   This level currently focuses on specific research directions or subfields. When refining it, it's recommended to distinguish between "technical areas" and "application areas" to avoid overlap. For example, Machine Learning is a technical area, while "Image Recognition" or "Natural Language Processing" can be considered application areas.
    *   **Example Keywords:** Machine Learning, Deep Learning, Computer Vision, Natural Language Processing, Data Mining, etc.

    **Level 3 (Research_Topic):**

    *   This level is already quite appropriate. It's recommended to expand "main research questions" to include hot topics, challenges, and technical bottlenecks within the field.
    *   **Example Keywords:** Generative Adversarial Networks, Reinforcement Learning, Multimodal Learning, Pre-trained Models, Transfer Learning, Model Compression, Explainability, etc.

    **Level 4 (Research_Method):**

    *   This level is very important, but it's necessary to pay attention to the specificity and generalizability of the methods. It's possible to differentiate between "fundamental Research_Methods" and "applied Research_Methods" to ensure that each method corresponds to a clear problem domain.
    *   **Example Keywords:** Transformer Architecture, Attention Mechanism, Contrastive Learning, Federated Learning, Meta-Learning, Deep Reinforcement Learning, etc.

    **Level 5 (Specific_Model):**

    *   This level is valuable, but when extracting models, it's possible to further divide them into "base models" and "specific application models." For example, differentiating between general-purpose models and specialized models would add more hierarchical structure.
    *   **Example Keywords:** BERT, GPT-3, GPT-4, ResNet, Stable Diffusion, VGG, CLIP, etc.

    **Level 6 (Model_Variants):**

    *   This level is very clear. It's possible to further refine the content related to model improvements, considering whether to classify them according to different application scenarios (such as domain-specific adjustments, optimizations, etc.).
    *   **Example Keywords:** RoBERTa, ALBERT, GPT-4 (variants implied), ResNet-50, ResNet-101, EfficientNet, Swin-Transformer, etc.

    **Level 7 (Specific_Technical_Details):**

    *   It's recommended to make the technical details level more specific. For example, detailed analysis of Specific_Technical_Details can be carried out according to application scenarios (such as text generation, image processing, etc.).
    *   **Example Keywords:** Self-Attention Mechanism, Positional Encoding, Layer Normalization, Activation Functions, Optimizers, Loss Functions, Convolution Operations, Normalization, etc.
                    
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
    
    # 加载数据
    manager.load_data(sample_data)
    
    # 合并类别
    manager.merge_categories()
    
    # 查找重复项
    manager.find_duplicates()
    
    # 打印状态报告
    print("初始状态报告:", manager.get_status_report())# 处理重复项示例（如果有的话）
    # for term, categories in manager.duplicates.items():
    #     # 这里可以根据具体规则选择保留的类别
    #     keep_category = categories[0]  # 示例：保留第一个类别
    #     manager.process_duplicate(term, keep_category)
    manager.process_duplicates()
    # 保存所有结果
    manager.save_results('output')# 打印最终状态报告
    print("最终状态报告:", manager.get_status_report())
