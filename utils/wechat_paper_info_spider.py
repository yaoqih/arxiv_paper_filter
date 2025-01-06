import requests
import json
import os
url = "https://mp.weixin.qq.com/mp/appmsgalbum"

# 请求参数
params = {
    "action": "getalbum",
    "__biz": "",
    "album_id": "3557071592793522177",
    "count": "20",
    "is_reverse": "",
    "uin": "",
    "key": "",
    "pass_ticket": "",
    "wxtoken": "",
    "devicetype": "",
    "clientversion": "",
    "appmsg_token": "",
    "x5": "0",
    "f": "json"
}

# 请求头
headers = {
    "accept": "*/*",
    "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
    "cache-control": "no-cache",
    "pragma": "no-cache",
    "priority": "u=1, i",
    "sec-ch-ua": '"Microsoft Edge";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "x-requested-with": "XMLHttpRequest"
}

# 设置 referrer
headers["referer"] = "https://mp.weixin.qq.com/mp/appmsgalbum?action=getalbum&album_id=3557071592793522177"

# 发送请求
if os.path.exists('paper_outlook/wechat_info.json'):
    wechat_info=json.load(open('paper_outlook/wechat_info.json','r',encoding='utf-8'))
else:
    wechat_info={}
# 打印响应
while True:
    response = requests.get(
        url,
        params=params,
        headers=headers,
        cookies={},  # 你需要添加必要的 cookies
    )
    if 'article_list' not in response.json()['getalbum_resp']:
        break
    for article in response.json()['getalbum_resp']['article_list']:
        if article['url'] not in wechat_info:
            wechat_info[article['url']]=article
    params['begin_msgid']=response.json()['getalbum_resp']['article_list'][-1]['msgid']
    params['begin_itemidx']=response.json()['getalbum_resp']['article_list'][-1]['itemidx']
    json.dump(wechat_info,open('paper_outlook/wechat_info.json','w',encoding='utf-8'),indent=2,ensure_ascii=False)
    print(f"已获取{len(wechat_info)}篇文章")