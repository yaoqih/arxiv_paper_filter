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

def create_improved_prompt(paper):
    return f"""
Title: {paper['title']}
Abstract: {paper['abstract']}

Please use your insight to extract the most representative keywords for each level. Don't be too general, but please respond using our agreed JSON format so my parsing won't go wrong.

""" 

def extract_keywords(abstract, client):
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
                keywords = extract_keywords(df.loc[i], client)
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
