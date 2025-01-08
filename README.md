# arxiv_paper_filter
## 使用方法
### 部署与开发
#### 1. 克隆代码
```
git clone https://github.com/yaoqih/arxiv_paper_filter
```
#### 2. 安装依赖
```
pip install -r requirements.txt
```
#### 3. 填写API Key文件
```
data/secret.json    
{
    "api_key": "",
    "base_url": ""
}
```
#### 4. 运行代码
```
python app.py
```
### 目录结构
```
\ARXIV_PAPER_FILTER
│  .gitignore
│  app.py                          // flask 后端启动文件
│  arxiv_search.py                 // arxiv 接口类
│  async_preprocess_utils.py       // openai 接口
│  build_index.py                  // 构建索引
│  paper_filter.py                 // arxiv 爬取与评分
│  postprocess_utils.py            // 后处理的模块，包含打标签和论文筛选
│  progress.py                     // 进度数据保存类
│  purify.py                       // 相似标签查找与清洗
│  README.md
│  requirements.txt
│  __init__.py
│  
├─data
│      duplicates_save.json        // 重复标签清洗进度保存
│      embeding_save.json          // 标签的embeding数据增量保存
│      filtered_papers.csv         // 经过arxiv爬取和评分的论文数据
│      labeled_purify.csv          // 经过标签清理的论文数据
│      labels.json                 // 原始标签保存
│      labels_purify.json          // 清理后的标签
│      label_index.json            // 标签索引
│      paper_labeled.csv           // 提取过标签的论文数据
│      progress_saving.json        // 进度数据保存
│      secret.json                 // 秘钥数据
│      similar_save.json           // 相似标签清理进度保存
│
├─paper_outlook                    // Daily Papers 数据
├─static                           // flask 前端静态文件
├─templates
│      filter.html                 // 论文筛选页面
│      index.html                  // 论文下载页面 
│      outlook.html                //Daily Papers 页面
│
详见仓库代码
```