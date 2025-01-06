import pandas as pd
import json
from tqdm import tqdm
import os
import json
from async_preprocess_utils import extract_label
from progress import ProgressManager
progress_manager=ProgressManager('data\progress_saving.json')

def extract_labels(input_file, out_file, progress_file, write=False):
    # 临时文件路径
    temp_out_file = progress_file + '.temp'
    
    # 读取输入文件
    df = pd.read_csv(input_file)
    if 'labels' not in df.columns:
        df['labels'] = None
    
    # 如果最终输出文件存在，读取它用于增量更新
    final_progress_file = None
    if os.path.exists(temp_out_file):
        final_progress_file = json.load(open(temp_out_file, 'r',encoding='utf-8'))
    elif os.path.exists(progress_file):
        final_progress_file = json.load(open(progress_file, 'r',encoding='utf-8'))
    else:
        final_progress_file = {}
    try:
        # 使用tqdm显示进度条
        for i in tqdm(range(len(df))):
            current_id = df.loc[i, 'id']
            
            # 检查是否已处理过该ID
            if current_id not in final_progress_file:
                try:
                    keywords = extract_label(df.loc[i])
                    if keywords:
                        final_progress_file[current_id] = keywords
                except Exception as e:
                    print(f"Error processing record {current_id}: {str(e)}")
                    continue
                
                # 每处理10条保存一次进度和临时文件
                if i % 10 == 0:
                    # 保存进度
                    json.dump(final_progress_file, open(temp_out_file, 'w',encoding='utf-8'))
                    # 保存临时文件    
    except Exception as e:
        print(f"Critical error occurred at index {i}: {str(e)}")
        raise
    
    finally:
        # 保存最终进度
        json.dump(final_progress_file, open(progress_file, 'w',encoding='utf-8'))
        
        for i in range(len(df)):
            current_id = df.loc[i, 'id']
            if current_id in final_progress_file:
                lables = [item for sublist in final_progress_file[current_id].values() for item in sublist]
                df.loc[i, 'labels'] = '|'.join(lables)
        df.to_csv(out_file, index=False, encoding='utf-8')

        # 删除临时文件
        if os.path.exists(temp_out_file):
            os.remove(temp_out_file)
        
        if write:
            progress_manager.set_keyword_status(label=True)
def filter_papers_by_csv(csv_file, keywords_list, start_date=None, end_date=None,start_score=None,end_score=None):
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    data_path=csv_file.split('/')[0]+'/'
    wechat_url={}
    if os.path.exists(data_path+'wechat_info.json'):
        wechat_info=json.load(open(data_path+'wechat_info.json',encoding='utf-8'))    
        for key in wechat_info:
            for url in wechat_info[key]['arxiv_url']:
                wechat_url[url]=key
    # 创建一个空的结果列表
    filtered_papers = []
    
    # 遍历每一行
    for _, row in df.iterrows():
        # 将keywords字符串分割成列表
        paper_keywords = set(row['labels'].split('|'))
        
        # 检查是否包含任何目标关键词
        if any(keyword in paper_keywords for keyword in keywords_list):
            # 检查日期范围
            if start_date and row['published'] < start_date:
                continue
            if end_date and row['published'] > end_date:
                continue
            if start_score and row['score'] < start_score:
                continue
            if end_score and row['score'] > end_score:
                continue
                
            # 将符合条件的论文添加到结果列表
            paper_dict = {
                'id': row['id'],
                'title': row['title'],
                'abstract': row['abstract'],
                'authors': row['authors'],
                'published': row['published'],
                'url': row['url'],
                'score': row['score'],
                'labels': row['labels'].split('|'),
                "wechat_url":wechat_url.get(row['id'],'')
            }
            filtered_papers.append(paper_dict)
    
    return filtered_papers
if __name__=='__main__':
    extract_labels('data/filtered_papers.csv','data/paper_labeled.csv','data/labels.json',write=True)
