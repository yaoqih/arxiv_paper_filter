import pandas as pd
import requests
import os

# 读取CSV文件
csv_file_path = 'output_translated.csv'  # 替换为你的CSV文件路径
df = pd.read_csv(csv_file_path)

# 创建保存PDF的文件夹
output_folder = 'arxiv_pdfs'
os.makedirs(output_folder, exist_ok=True)

# 遍历每一行，下载PDF
for index, row in df.iterrows():
    arxiv_url = row['id']  # 假设'id'是保存arxiv网址的列名
    if f'{arxiv_url.split("/")[-1]}.pdf' in os.listdir(output_folder):
        print(f"Already downloaded: {arxiv_url.split('/')[-1]}.pdf")
        continue
    if 'abs' in arxiv_url:  # 确保是arxiv的摘要链接
        pdf_url = arxiv_url.replace('abs', 'pdf')  # 替换为PDF链接
        pdf_url = pdf_url.replace('v1', '')  # 去掉版本号
        pdf_name = os.path.join(output_folder, f"{arxiv_url.split('/')[-1]}.pdf")  # 生成文件名

        try:
            response = requests.get(pdf_url)
            response.raise_for_status()  # 检查请求是否成功

            # 保存PDF到指定文件夹
            with open(pdf_name, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {pdf_name}")

        except requests.HTTPError as e:
            print(f"Failed to download {pdf_url}: {e}")

