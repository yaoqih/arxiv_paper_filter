import pandas as pd
from openai import OpenAI
import json
import time
from tqdm import tqdm
import os
import json
config = json.load(open('config.json', 'r'))
client = OpenAI(
            api_key=config['api_key'],
            base_url="https://api.chatanywhere.tech"
        )

def create_improved_prompt(abstract):
    return """Below are the title and abstract of a paper. Could you please help me extract the keywords at different levels? Please provide the results in JSON format only, so that my parsing won't go wrong. Thank you!
""" + abstract

def extract_keywords(abstract, client):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # 或其他适合的模型
            messages=[
                {"role": "system", "content": """
"You are a knowledgeable mentor with clear and accurate insights who is eager to help. I am a beginner student in the field of artificial intelligence. The AI field is developing so fast, with numerous papers published daily, making it difficult to track cutting-edge research. Therefore, I would like you to help me identify the most representative keywords at different levels for research papers. This way, I can quickly filter and study AI research based on keywords. Below is a reference hierarchy. You can extract corresponding keywords at different research levels based on the paper's title, abstract, and other information. It's okay if some levels are missing. Sometimes the abstract may not mention Specific_Models or algorithms; in such cases, you can fill in based on your experience and judgment.

```
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
Please feel free to use your insight to extract the most representative keyword for each level, but please respond in our agreed JSON format so my parsing won't have errors.

```json
{
"Discipline": [],         
"Research_Area": [],      
"Research_Topic": [],    
"Research_Method": [],   
"Specific_Model": [],     
"Model_Variants": [],     
"Specific_Technical_Details": [] 
}
```

I think each level should be highly cohesive, with low coupling between levels. Keywords should be as specific as possible and closely aligned with the paper's research focus. If they're too broad, they won't accurately locate the paper's research."
                 """},
                {"role": "user", "content": create_improved_prompt(abstract)}
            ],
            temperature=0.7,
        )       
        # 提取返回的JSON内容
        result = json.loads(response.choices[0].message.content.replace('```json\n', '').replace('```', ''))
        return result
    except Exception as e:
        print(f"Error processing abstract: {str(e)}")
        return None

def process_csv(input_file,out_file, progress_file="key_words.json"):
    # 读取CSV文件
    df = pd.read_csv(input_file)
    if 'keywords' not in df.columns:
        df['keywords'] = None
    
    # 获取上次处理的进度
    last_processed = 0
    if os.path.exists(progress_file):
        result_save = json.load(open(progress_file, "r"))
    else:
        result_save = {}
    
    try:
        # 使用tqdm显示进度条
        for i in tqdm(range(last_processed, len(df)), initial=last_processed, total=len(df)):
            if df.loc[i, 'id'] not in result_save :
                keywords = extract_keywords(df.loc[i, 'abstract'], client)
                if keywords:
                    all_keywords=[]
                    for values in keywords.values():
                        # for detail in values:
                        all_keywords.extend(values)
                    df.loc[i, 'keywords'] = "|".join(all_keywords)
                    result_save[df.loc[i, 'id']] = keywords
                
                # 每处理10条保存一次进度
                if i % 10 == 0:
                    json.dump(result_save, open(progress_file, "w"))
                    df.to_csv(out_file, index=False)
                
                # 添加延时以避免API限制
                # time.sleep(1)
    
    except Exception as e:
        print(f"Error occurred at index {i}: {str(e)}")
    finally:
        # 保存最终结果
        json.dump(result_save, open(progress_file, "w"))
        df.to_csv(out_file, index=False)
def keywords_extraction_write(input_file = "output.csv",output_file = "output_keywords.csv"):
    process_csv(input_file, output_file)
    config_file = config['config_file']
    data_file = json.load(open(config_file, 'r',encoding='utf-8'))
    data_file['keyword_update'] = True
    json.dump(data_file, open(config_file, 'w',encoding='utf-8'))
if __name__ == "__main__":
    input_file = "output.csv"
    output_file = "output_keywords.csv"
    process_csv(input_file, output_file)
