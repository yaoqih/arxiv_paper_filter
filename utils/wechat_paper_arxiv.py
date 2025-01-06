import requests
import json
import re
from tqdm import tqdm
# 设置通用headers
headers = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "accept-language": "zh-CN,zh;q=0.9,en;q=0.8", 
    "cache-control": "no-cache",
    "pragma": "no-cache",
    "sec-ch-ua": "\"Microsoft Edge\";v=\"131\", \"Chromium\";v=\"131\", \"Not_A Brand\";v=\"24\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Windows\"",
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "same-origin",
    "upgrade-insecure-requests": "1"
}

# 获取文章内容
def fetch_article():
    url = "https://mp.weixin.qq.com/s"
    params = {
        "__biz": "Mzg5MjQzNDk1NA==",
        "mid": "2247483765",
        "idx": "1",
        "sn": "e89e67a3d46830501743d6110bb5a03d",
        "chksm": "c03f6e10f748e7069305800603fdd6fd49e3397d69566055a8d53d44cf0c0f4f234e15ef71b0"
    }
    
    response = requests.get(url, headers=headers, params=params)
    return response.text
def main():
    wechat_info=json.loads(open('paper_outlook/wechat_info.json','r',encoding='utf-8').read())
    for key in tqdm(wechat_info):
        if 'arxiv_url' in wechat_info[key]:
            continue
        article = requests.get(key, headers=headers).text
        # 提取arxiv url
        arxiv_url = ['https://'+_ for _ in re.findall(r"arxiv.org/abs/\d+\.\d+", article)]
        wechat_info[key]['arxiv_url'] = arxiv_url
        json.dump(wechat_info, open('paper_outlook/wechat_info.json', 'w',encoding='utf-8'), ensure_ascii=False, indent=4)
    print("获取文章成功")

if __name__ == "__main__":
    main()
