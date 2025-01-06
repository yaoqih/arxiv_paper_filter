import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re
from typing import List, Dict, Optional,Generator

class ArxivScraper:
    """ArXiv advanced search scraper"""
    
    BASE_URL = "https://arxiv.org/search/advanced"
    
    def __init__(self):
        self.session = requests.Session()

    def search_batch(self, 
                    query: str,
                    size: int = 50,
                    date_range: tuple = None,
                    sort_by: str = "-announced_date_first",
                    max_results: Optional[int] = None) -> List[Dict]:
        """
        Search arxiv papers and return all results as a list
        
        Args:
            query: Search query string
            size: Results per page (25, 50, 100, 200) 
            date_range: Tuple of (start_date, end_date) in YYYY-MM-DD format
            sort_by: Sort order
            max_results: Maximum number of results to return (None for all)
            
        Returns:
            List of paper dictionaries
        """
        results = []
        for paper in self.search(query, size, date_range, sort_by):
            results.append(paper)
            if max_results and len(results) >= max_results:
                break
        return results[:max_results] if max_results else results
        
    def search(self, 
              query: str,
              size: int = 200,
              date_range: tuple = None, 
              sort_by: str = "-announced_date_first") -> Generator[Dict, None, None]:
        """
        Search arxiv papers
        
        Args:
            query: Search query string
            size: Results per page (25, 50, 100, 200)
            date_range: Tuple of (start_date, end_date) in YYYY-MM-DD format
            sort_by: Sort order ("-announced_date_first", "announced_date_first", 
                              "-submitted_date", "submitted_date", "relevance")
            max_results: Maximum number of results to return (None for all)
            
        Returns:
            List of paper dictionaries containing:
                - title: Paper title
                - authors: List of authors
                - abstract: Paper abstract
                - url: ArXiv URL
                - pdf_url: PDF URL
                - published_date: Publication date
                - updated_date: Last update date
        """
        
        params = {
            "advanced": "",
            "terms-0-operator": "AND",
            "terms-0-term": query,
            "terms-0-field": "all",
            "classification-physics_archives": "all",
            "classification-include_cross_list": "include",
            "date-filter_by": "date_range" if date_range else "all_dates",
            "date-year": "",
            "date-from_date": date_range[0] if date_range else "",
            "date-to_date": date_range[1] if date_range else "",
            "date-date_type": "submitted_date",
            "abstracts": "show",
            "size": str(size),
            "order": sort_by
        }
        
        page = 0
        while True:
            response = requests.get(self.BASE_URL, params=params)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            papers = soup.find_all("li", class_="arxiv-result")
            
            if not papers:
                break
                
            for paper in papers:
                yield self._parse_paper(paper)
            page += 1
            params["start"] = size*page
    def get_result_num(self, query: str) -> int:
        params = {
            "advanced": "",
            "terms-0-operator": "AND",
            "terms-0-term": query,
            "terms-0-field": "all",
            "classification-physics_archives": "all",
            "classification-include_cross_list": "include",
            "date-filter_by": "all_dates",
            "date-year": "",
            "date-from_date": "",
            "date-to_date":  "",
            "date-date_type": "submitted_date",
            "abstracts": "show",
            "size": str(50),
        }
        
        response = requests.get(self.BASE_URL, params=params)
        soup = BeautifulSoup(response.text, 'html.parser')
        # 查找包含结果数量的h1标签
        results_text = soup.find('h1', class_='title').text.strip()
        if 'Showing 'in results_text:
            # 使用字符串处理提取数字
            total_results = results_text.split('of')[1].split('results')[0].strip()
        else:
            total_results = 0
        return int(total_results)
                
    def _parse_paper(self, paper_html) -> Dict:
        """Parse paper HTML into structured data"""
        
        # Get paper ID and URLs
        paper_id = paper_html.find("p", class_="list-title").a.text.strip().replace("arXiv:", "")
        arxiv_url = f"https://arxiv.org/abs/{paper_id}"
        pdf_url = f"https://arxiv.org/pdf/{paper_id}"# Get title
        title = paper_html.find("p", class_="title").text.strip()
        
        # Get authors
        authors_p = paper_html.find("p", class_="authors")
        authors = [a.text for a in authors_p.find_all("a")]# Get abstract
        abstract = paper_html.find("span", class_="abstract-full").text.strip("△ Less\n")
        
        # Get dates
        dates_p = paper_html.find("p", class_="is-size-7")
        dates_text = dates_p.text
        
        submitted_date = re.search(r"Submitted\s+(.+?);", dates_text)
        submitted_date = submitted_date.group(1) if submitted_date else None
        
        updated_date = re.search(r"v\d+\s+submitted\s+(.+?);", dates_text) 
        updated_date = updated_date.group(1) if updated_date else submitted_date
        
        return {
            "title": title,
            "authors": authors, 
            "abstract": abstract.replace("\n", " "),
            "url": arxiv_url,
            "pdf_url": pdf_url,
            "published_date": datetime.strptime(submitted_date, "%d %B, %Y").strftime("%Y-%m-%d"),
            "updated_date": datetime.strptime(updated_date, "%d %B, %Y").strftime("%Y-%m-%d")
        }
if __name__ == "__main__":
    scraper = ArxivScraper()
    results = scraper.search(
        query="multivariate time series forecasting pre-training fine-tuning",
        # size=200,
        # date_range=("2021-01-01", "2024-12-31"),
        # sort_by="-submitted_date",
    )
    
    for result in results:
        print(result)