from flask import Flask, request, jsonify,render_template
import json
from flask_cors import CORS
import pandas as pd
from datetime import datetime
import os
import threading
import json
from async_preprocess_utils import generate_intent,generate_keywords,generate_criterion,write_search_num
from  paper_filter import paper_spider_filter
from postprocess_utils import extract_labels
from build_index import build_index_write
from purify import prefiy_keywords_write
from progress import ProgressManager
app = Flask(__name__)
CORS(app)

progress_manager=ProgressManager('data/progress_saving.json')
secret = json.load(open('data/secret.json', 'r'))

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response
    
def filter_papers_by_csv(csv_file, keywords_list, start_date=None, end_date=None,start_score=None,end_score=None):
    # 读取CSV文件
    df = pd.read_csv(csv_file)
    
    # 将published列转换为datetime格式
    # df['published'] = pd.to_datetime(df['published'])
    
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
                'labels': row['labels'].split('|')
            }
            filtered_papers.append(paper_dict)
    
    return filtered_papers
def get_keywords_file():
    return json.load(open("data/label_index.json",encoding='utf-8'))
    
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
    
    return jsonify(filter_papers_by_csv('data/labeled_purify.csv', keywords_list,start_date=data['startDate'],end_date=data['endDate'],start_score=data['startScore'],end_score=data['endScore']))
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
    new_data = request.json
    # 更新搜索配置
    if 'search_config' in new_data:
        progress_manager.update_search_config(**new_data['search_config'])
    return jsonify({"status": "success"})
@app.route('/api/get_progress', methods=['GET'])
def get_progress():
    return jsonify(progress_manager.get_data())
@app.route('/openai_process', methods=['POST'])
def openai_process():
    if request.json['type']=='generate_intent':
       task=generate_intent
       content=[request.json['contents']['query'],True]
    elif request.json['type']=='generate_keywords':
        task=generate_keywords
        content=[request.json['contents']['query'],True]
    elif request.json['type']=='generate_criterion':
        task=generate_criterion
        content=[request.json['contents']['query'],True]
    elif request.json['type']=='keywords':
        task=write_search_num
        content=[request.json['contents']['keywords'].strip().split('\n')]
    elif request.json['type']=='start_dowanlad':
        task=paper_spider_filter
        content=[request.json['contents']['keywords'].strip().split('\n'),request.json['contents']['criterion']]
    elif request.json['type']=='label':
        task=extract_labels
        content=['data/filtered_papers.csv','data/paper_labeled.csv','data/labels.json',True]
    elif request.json['type']=='purify':
        task=prefiy_keywords_write
        content=[]
    elif request.json['type']=='build_index':
        task=build_index_write
        content=[]
    elif request.json['type']=='clean':
        task=clean_file
        content=[]
    thread = threading.Thread(target=task, args=tuple(content), name="MyBackgroundTask")
    thread.daemon = True  # 可选：设置为守护线程，当主线程结束时，子线程也会结束
    thread.start()
    return jsonify({'data':'success'})
def clean_file():
    for file in ['data/filtered_papers.csv','output_keywords.csv','key_words.json','transformed_data.json','output\duplicates.json','output\processed_data.json','output\merged_data.json','processing_history.json','raw_data.json']:
        if os.path.exists(file):
            os.remove(file)
    progress_manager.clear()

if __name__ == '__main__':
    app.run(debug=True)
