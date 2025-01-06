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
from postprocess_utils import extract_labels,filter_papers_by_csv
from build_index import build_index
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
    
@app.route('/filter_papers', methods=['POST'])
def filter_papers():
    data = request.json
    selected_filters=data['filterLists']
    data_path=data['dataPath']
    keywords=json.load(open(f"{data_path}label_index.json",encoding='utf-8'))
    keywords_list = []
    if selected_filters:
        keywords_list.extend([_['key'] for _ in selected_filters])
    else:
        for key in keywords:
            keywords_list.extend(keywords[key].keys())
    return jsonify(filter_papers_by_csv(f'{data_path}labeled_purify.csv', keywords_list,start_date=data['startDate'],end_date=data['endDate'],start_score=data['startScore'],end_score=data['endScore']))

@app.route('/get_keywords', methods=['POST'])
def get_keywords():
    data = request.json
    return jsonify(json.load(open(f"{data['dataPath']}label_index.json",encoding='utf-8')))

@app.route('/filter', methods=['GET'])
def filter():
    return render_template('filter.html')

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/outlook', methods=['GET'])
def outlook():
    return render_template('outlook.html')

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
        task=build_index
        content=['data/labels_purify.json','data/labeled_purify.csv','data/label_index.json',True]
    elif request.json['type']=='clean':
        task=clean_file
        content=[]
    thread = threading.Thread(target=task, args=tuple(content), name="MyBackgroundTask")
    thread.daemon = True  # 可选：设置为守护线程，当主线程结束时，子线程也会结束
    thread.start()
    return jsonify({'data':'success'})

def clean_file():
    for file in ['data/duplicates_save.json','data/embeding_save.json','data/filtered_papers.csv','data/label_index.json','data/labeled_purify.csv','data/labels_purify.json','data/labels.json','data/paper_labeled.csv','data/progress_saving.json','data\similar_save.json']:
        if os.path.exists(file):
            os.remove(file)
    progress_manager.clear()

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
