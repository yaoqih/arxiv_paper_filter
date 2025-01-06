import pandas as pd
from googletrans import Translator

def translate_csv_abstract(input_file, output_file):
    """
    读取CSV文件，将abstract列从英文翻译成中文，并保存到新的CSV文件中
    
    参数:
    input_file (str): 输入CSV文件路径
    output_file (str): 输出CSV文件路径
    """
    # 初始化翻译器
    translator = Translator()
    
    try:
        # 读取CSV文件
        df = pd.read_csv(input_file)
        
        # 检查是否存在'abstract'列
        if 'abstract' not in df.columns:
            raise ValueError("CSV文件中未找到'abstract'列")
        
        # 翻译abstract列
        df['abstract'] = df['abstract'].str.replace('\n', '')  # 替换换行符
        df['abstract_cn'] = df['abstract'].apply(lambda x: translator.translate(str(x), src='en', dest='zh-cn').text)
        df['abstract_cn'] = df['abstract_cn'].str.replace('\n', '')  # 替换换行符
        
        # 保存带有中文翻译的CSV文件
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"翻译完成！结果已保存到 {output_file}")
    
    except Exception as e:
        print(f"发生错误：{e}")

# 使用示例
input_csv_path = 'time_serial.csv'
output_csv_path = 'time_serial_translated.csv'
translate_csv_abstract(input_csv_path, output_csv_path)

