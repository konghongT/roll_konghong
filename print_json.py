import json

def print_json_fields(file_path):
    try:
        # 读取 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # 检查是否是列表形式
        if isinstance(data, list):
            for i, entry in enumerate(data, 1):
                print(f"\n=== 条目 {i} ===")
                print_fields(entry)
        else:
            print("\n=== 单个条目 ===")
            print_fields(data)
            
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到")
    except json.JSONDecodeError:
        print("错误：文件不是有效的 JSON 格式")
    except Exception as e:
        print(f"发生错误：{str(e)}")

def print_fields(entry):
    # 打印每个字段
    print(f"Temperature: {entry.get('temperature', '无')}")
    print(f"Prompt: {entry.get('prompt', '无')[:100]}...")  # 只显示前100个字符避免太长
    print(f"Dataset Generation: {entry.get('dataset_generation', '无')}...")
    print(f"API Generation: {entry.get('api_generation', '无')}...")
    print(f"Similarity Score: {entry.get('similarity_score', '无')}")

# 使用示例
if __name__ == "__main__":
    file_path = "/root/distill_konghong/test/data/test_merged_data_20250515_180451.json"
    print_json_fields(file_path)