import json
import os
import sys

def clean_config(config_path='config.json'):
    """清空配置文件中的值"""
    try:
        # 读取原始配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 保存原始配置的备份
        backup_path = config_path + '.backup'
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            
        # 递归清空字典中的所有值
        def clean_dict(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    clean_dict(value)
                else:
                    d[key] = ""
        
        clean_dict(config)
        
        # 保存清空后的配置
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            
        print(f"配置已清空并保存到 {config_path}")
        print(f"原始配置已备份到 {backup_path}")
        
    except Exception as e:
        print(f"清空配置文件时出错: {str(e)}")
        sys.exit(1)

def restore_config(config_path='config.json'):
    """从备份文件还原配置"""
    try:
        backup_path = config_path + '.backup'
        
        if not os.path.exists(backup_path):
            print(f"未找到备份文件: {backup_path}")
            return
            
        # 从备份还原
        with open(backup_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        # 保存还原的配置
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            
        print(f"配置已从 {backup_path} 还原")
        
    except Exception as e:
        print(f"还原配置文件时出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("请指定操作: clean 或 restore")
        sys.exit(1)
        
    action = sys.argv[1]
    if action == "clean":
        clean_config()
    elif action == "restore":
        restore_config()
    else:
        print("无效的操作，请使用 clean 或 restore")
        sys.exit(1)
