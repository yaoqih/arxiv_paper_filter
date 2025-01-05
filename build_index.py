import json
import pandas as pd
# from googletrans import Translator
from datetime import datetime
from tqdm import tqdm
import traceback
from progress import ProgressManager
progress_manager=ProgressManager('data\progress_saving.json')

# 读取文件
def load_files(json_path, csv_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.read_csv(csv_path)
    return data, df

def process_keyword(label, df, translator=None):
    #翻译（如果不是中文）
    if not any('\u4e00' <= char <= '\u9fff' for char in label) and translator:
        try:
            translation = translator.translate(label,src='en', dest='zh-CN').text
        except:
            translation = label
            traceback.print_exc()
    else:
        translation = label
    
    # 统计关键词出现次数
    # count = df[df['keywords'].str.contains(keyword, case=False, na=False)].shape[0]
    
    # 获取日期范围
    df['labels_list'] = df['labels'].str.split('|')
    date_range = df[df['labels_list'].apply(lambda x:label  in x if isinstance(x, list) else False)]['published']
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
def build_index_write(json_path = 'data/labels_purify.json',csv_path = 'data/labeled_purify.csv',output_path = 'data/label_index.json'):
    data, df = load_files(json_path, csv_path)
    transformed_data = transform_json(data, df)
    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(transformed_data, f, ensure_ascii=False, indent=4)
    progress_manager.set_keyword_status(build_index=True)

if __name__ == "__main__":
    build_index_write()
