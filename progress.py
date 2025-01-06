from datetime import datetime
import json
from typing import Dict

class ProgressManager:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = self._init_default_data()
        self._load_data()

    def _init_default_data(self) -> Dict:
        """初始化默认数据结构"""
        return {
            "search_config": {
                "query": "",
                "intent": "",
                "criterion": "",
                "keywords": ""
            },
            "search_preview": {
                "total_count": 0,
                "categories": {}
            },
            "download_status": {
                "completed": 0,
                "remaining": 0,
            },
            "label": False,
            "purify": False,
            "build_index": False,
            "last_updated": datetime.now().isoformat()
        }

    def _load_data(self) -> None:
        """从文件加载数据，如果文件不存在则使用默认数据"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            self.save()

    def save(self) -> None:
        """保存当前数据到文件"""
        self.data["last_updated"] = datetime.now().isoformat()
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False)

    # 搜索配置相关方法
    def update_search_config(self, query: str = None, intent: str = None,
                           criterion: str = None, keywords: str = None) -> None:
        self._load_data()
        """更新搜索配置"""
        if query is not None:
            self.data["search_config"]["query"] = query
        if intent is not None:
            self.data["search_config"]["intent"] = intent
        if criterion is not None:
            self.data["search_config"]["criterion"] = criterion
        if keywords is not None:
            self.data["search_config"]["keywords"] = keywords
        self.save()

    # 搜索预览相关方法
    def update_search_preview(self, total_count: int, categories: Dict) -> None:
        """更新搜索预览数据"""
        self._load_data()
        self.data["search_preview"]["total_count"] = total_count
        self.data["search_preview"]["categories"] = categories
        self.save()

    # 下载状态相关方法
    def update_download_status(self, 
                             completed: int = None, remaining: int = None,
                            ) -> None:
        """更新下载状态"""
        self._load_data()
        if completed is not None:
            self.data["download_status"]["completed"] = completed
        if remaining is not None:
            self.data["download_status"]["remaining"] = remaining
        self.save()

    # 关键词处理状态相关方法
    def set_keyword_status(self, label: bool = None, 
                          purify: bool = None,
                          build_index: bool = None) -> None:
        """设置关键词处理的各个阶段状态"""
        self._load_data()
        if label is not None:
            self.data["label"] = label
        if purify is not None:
            self.data["purify"] = purify
        if build_index is not None:
            self.data["build_index"] = build_index
        self.save()

    # 获取数据的方法
    def get_data(self) -> Dict:
        self._load_data()
        return self.data
    def get_search_config(self) -> Dict:
        return self.data["search_config"]

    def get_search_preview(self) -> Dict:
        return self.data["search_preview"]

    def get_download_status(self) -> Dict:
        return self.data["download_status"]

    def get_keyword_status(self) -> Dict:
        return {
            "update": self.data["label"],
            "refinement": self.data["purify"],
            "preprocessing": self.data["build_index"]
        }

    def get_last_updated(self) -> str:
        return self.data["last_updated"]
    def clear(self):
        self.data = self._init_default_data()
        self.save()