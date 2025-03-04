import csv
import json


def csv_to_json(csv_file_path, json_file_path):
    # 打开CSV文件并读取内容
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)  # 使用DictReader，自动将CSV的第一行作为字典的键
        data = []

        for row in csv_reader:
            # 对每一行数据进行处理并将其加入到列表中
            data.append({
                "sentence": row['sentence'],
                "aspect": row['aspect'],
                "label_id": int(row['label_id']),
                "label": row['label']
            })

    # 将处理后的数据写入JSON文件
    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

    print(f"CSV文件已成功转换为JSON格式，并保存在 {json_file_path}")


# 例子：调用函数转换文件
csv_file_path = 'SemEvalrest14/train.csv'  # 这里是你的CSV文件路径
json_file_path = 'SemEvalrest14/train.json'  # 输出的JSON文件路径

csv_to_json(csv_file_path, json_file_path)
