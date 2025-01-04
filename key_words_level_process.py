import json
import pandas as pd
# from googletrans import Translator
from datetime import datetime
from tqdm import tqdm
import traceback
config = json.load(open('config.json', 'r'))

# 读取文件
def load_files(json_path, csv_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.read_csv(csv_path)
    return data, df

def process_keyword(keyword, df, translator=None):
    #翻译（如果不是中文）
    if not any('\u4e00' <= char <= '\u9fff' for char in keyword):
        try:
            translation = translator.translate(keyword,src='en', dest='zh-CN').text
        except:
            translation = keyword
            traceback.print_exc()
    else:
        translation = keyword
    
    # 统计关键词出现次数
    # count = df[df['keywords'].str.contains(keyword, case=False, na=False)].shape[0]
    
    # 获取日期范围
    df['keyword_list'] = df['keywords'].str.split('|')
    date_range = df[df['keyword_list'].apply(lambda x:keyword  in x if isinstance(x, list) else False)]['published']
    # if not matching_rows.empty:
    # else:
    #     date_range = {'earliest': None, 'latest': None}
    
    return {
        'zh-cn': translation,
        # 'number': count,
        'dates': list(date_range)
    }

def transform_json(data, df):
    # translator = Translator()
    
    transformed_data = {}
    
    for category, items in data.items():
        transformed_data[category] = {}
        for item in tqdm(items,'prcessing:'+category):
            transformed_data[category][item] = process_keyword(item, df)
    
    return transformed_data
def keyword_preprocessing_write(json_path = 'output\processed_data.json',csv_path = 'output_keywords.csv',output_path = 'transformed_data.json'):
    data, df = load_files(json_path, csv_path)
    transformed_data = transform_json(data, df)
    # 保存结果
    with open('transformed_data.json', 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, ensure_ascii=False, indent=4)
    data_save=json.load(open(config['config_file'], 'r',encoding='utf-8'))
    data_save['keyword_preprocessing']=True
    data_save['last_updated'] = datetime.now().isoformat()
    with open(config['config_file'], 'w', encoding='utf-8') as f:
        json.dump(data_save, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 使用示例
    json_path = 'output\processed_data.json'
    csv_path = 'output.csv'

    data, df = load_files(json_path, csv_path)
    transformed_data = transform_json(data, df)

    # 保存结果
    with open('transformed_data.json', 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, ensure_ascii=False, indent=4)
