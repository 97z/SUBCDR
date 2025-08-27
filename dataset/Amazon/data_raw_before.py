import json
import csv
import gzip
import pandas as pd
#Amazon
# 输入 JSON 文件路径和输出 CSV 文件路径
def amazon_data():
    input_file = r'E:\datasets\Amazon\raw\2018\Amazon_Arts_Crafts_and_Sewing\Arts_Crafts_and_Sewing.json.gz'  # 替换为你的 JSON 文件路径
    output_file = r'E:\pycharmcode\CrossAug-main\datasets\raw\Amazon_paper\Arts\Arts.csv'  # 替换为你希望生成的 CSV 文件路径
    csv_data = []
    # 读取 JSON 文件
    with gzip.open(input_file, 'r') as f:
        for line in f:
            # 解析每一行的 JSON 对象
            entry = json.loads(line.strip())
            reviewerID = entry["reviewerID"]
            asin = entry["asin"]
            overall = int(entry["overall"])  # 将 float 转换为 int
            unixReviewTime = entry["unixReviewTime"]
            csv_data.append([reviewerID, asin, overall, unixReviewTime])

    # 写入 CSV 文件
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)

    print(f"CSV 文件已成功生成：{output_file}")
def douban_data():
    # 文件路径
    txt_file_path = r'E:\datasets\Douban\raw\musicreviews_cleaned.txt'  # 替换为你的文件路径
    output_file_path = r'E:\pycharmcode\CrossAug-main\datasets\raw\Douban\Music\Music.csv'  # 替换为保存处理结果的路径

    # 读取文件
    try:
        # 使用Tab分隔符读取文件
        df = pd.read_csv(txt_file_path, sep='\t', header=0)

        # 选取需要的列
        processed_df = df[['user_id', 'music_id', 'rating', 'time']]

        # 去掉时间列中的 '-'
        processed_df['time'] = processed_df['time'].str.replace('-', '', regex=True)

        # 输出到新的文件
        processed_df.to_csv(output_file_path, index=False)
        print(f"处理完成，新表已保存到: {output_file_path}")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")
def cal_user():

    # 数据集文件路径
    data1_path = r'E:\pycharmcode\CrossAug-main\datasets\raw\Douban\Book\Book.csv'  # 替换为第一个数据集路径
    data2_path = r'E:\pycharmcode\CrossAug-main\datasets\raw\Douban\Movie\Movie.csv'  # 替换为第二个数据集路径
    data3_path = r'E:\pycharmcode\CrossAug-main\datasets\raw\Douban\Music\Music.csv'  # 替换为第三个数据集路径
    columns = ['user_id', 'item_id', 'rating', 'time']
    # 加载数据
    data1 = pd.read_csv(data1_path, header=None, names=columns)
    data2 = pd.read_csv(data2_path, header=None, names=columns)
    data3 = pd.read_csv(data3_path, header=None, names=columns)

    # 计算每个数据集的用户和物品总数
    def calculate_unique_counts(data, dataset_name):
        num_users = data['user_id'].nunique()
        num_items = data['item_id'].nunique()
        print(f"{dataset_name}: Users = {num_users}, Items = {num_items}")
        return set(data['user_id'])

    # 获取每个数据集的用户集合
    users1 = calculate_unique_counts(data1, "Dataset 1")
    users2 = calculate_unique_counts(data2, "Dataset 2")
    users3 = calculate_unique_counts(data3, "Dataset 3")

    # 计算重叠用户数量
    overlap_12 = len(users1 & users2)
    overlap_13 = len(users1 & users3)
    overlap_23 = len(users2 & users3)

    print(f"Overlap Users between Dataset 1 and 2: {overlap_12}")
    print(f"Overlap Users between Dataset 1 and 3: {overlap_13}")
    print(f"Overlap Users between Dataset 2 and 3: {overlap_23}")

if __name__ == '__main__':
    cal_user()