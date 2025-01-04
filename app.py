from flask import Flask, request, jsonify,render_template
import json
from flask_cors import CORS
import pandas as pd
from datetime import datetime
import os
import threading
import json
from async_preprocess_utils import intention_analysis_write,generate_search_keywords_write,write_search_num
from  paper_filter import paper_spider_filter
from extract_label import keywords_extraction_write
from key_words_level_process import keyword_preprocessing_write
from prefiy import prefiy_keywords_write
config = json.load(open('config.json', 'r'))
app = Flask(__name__)
CONFIG_FILE = config['config_file']
CORS(app)
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

def load_config(is_first=False):
    if is_first or not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(create_default_config(), f, ensure_ascii=False, indent=2)
    else :
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return create_default_config()

def create_default_config():
    return {
        "search_config": {
            "paper_type": "",
            "intent": "",
            "keywords": "",
            "confirmed_keywords": ""
        },
        "search_preview": {
            "total_count": 0,
            "categories": {}
        },
        "download_status": {
            "keywords": [],
            "completed": 0,
            "remaining": 0,
            "papers": []
        },
        "keyword_update": False,
        "keyword_refinement": False,
        "keyword_preprocessing": False,
        "last_updated": datetime.now().isoformat()
    }

def save_config(config):
    config['last_updated'] = datetime.now().isoformat()
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
def filter_papers_by_csv(csv_file, keywords_list, start_date=None, end_date=None):
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 将published列转换为datetime格式
    # df['published'] = pd.to_datetime(df['published'])
    
    # 创建一个空的结果列表
    filtered_papers = []
    
    # 遍历每一行
    for _, row in df.iterrows():
        # 将keywords字符串分割成列表
        paper_keywords = set(row['keywords'].split('|'))
        
        # 检查是否包含任何目标关键词
        if any(keyword in paper_keywords for keyword in keywords_list):
            # 检查日期范围
            if start_date and row['published'] < start_date:
                continue
            if end_date and row['published'] > end_date:
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
                'keywords': row['keywords'].split('|')
            }
            filtered_papers.append(paper_dict)
    
    return filtered_papers
def get_keywords_file():
    return json.load(open("transformed_data.json",encoding='utf-8'))
    
@app.route('/filter_papers', methods=['POST'])
def filter_papers():
    data = request.json
    selected_filters=data['filterLists']
    keywords=get_keywords_file()
    keywords_list = []
    if selected_filters:
        keywords_list.extend([_['key'] for _ in selected_filters])
    else:
        for key in keywords:
            keywords_list.extend(keywords[key].keys())
    
    return jsonify(filter_papers_by_csv('output_keywords.csv', keywords_list,start_date=data['startDate'],end_date=data['endDate']))
@app.route('/get_keywords', methods=['GET'])
def get_keywords():
    return jsonify(get_keywords_file())
@app.route('/filter', methods=['GET'])
def filter():
    return render_template('filter.html')
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')
@app.route('/api/update_progress', methods=['POST'])
def update_progress():
    config = load_config()
    new_data = request.json
    # 更新搜索配置
    if 'search_config' in new_data:
        config['search_config'].update(new_data['search_config'])
    
    save_config(config)
    return jsonify({"status": "success"})
@app.route('/api/get_progress', methods=['GET'])
def get_progress():
    return jsonify(load_config())
@app.route('/openai_process', methods=['POST'])
def openai_process():
    if request.json['type']=='paper-type':
       task=intention_analysis_write
       content=[request.json['contents']['paper-type']]
    elif request.json['type']=='search-intent':
        task=generate_search_keywords_write
        content=[request.json['contents']['paper-type']]
    elif request.json['type']=='search-keywords':
        task=write_search_num
        content=[request.json['contents']['search-keywords'].strip().split('\n')]
    elif request.json['type']=='start_dowanlad':
        task=paper_spider_filter
        content=[request.json['contents']['search-keywords'].strip().split('\n'),request.json['contents']['search-intent']]
    elif request.json['type']=='keyword_update':
        task=keywords_extraction_write
        content=[]
    elif request.json['type']=='keyword_refinement':
        task=prefiy_keywords_write
        content=[]
    elif request.json['type']=='keyword_preprocessing':
        task=keyword_preprocessing_write
        content=[]
    elif request.json['type']=='clean':
        task=clean_file
        content=[]
    thread = threading.Thread(target=task, args=tuple(content), name="MyBackgroundTask")
    thread.daemon = True  # 可选：设置为守护线程，当主线程结束时，子线程也会结束
    thread.start()
    return jsonify({'data':'success'})
def clean_file():
    for file in ['output.csv','output_keywords.csv','key_words.json','transformed_data.json','output\duplicates.json','output\processed_data.json','output\merged_data.json','processing_history.json','raw_data.json']:
        if os.path.exists(file):
            os.remove(file)
    load_config(True)

if __name__ == '__main__':
    app.run(debug=True)
