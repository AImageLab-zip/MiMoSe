import zipfile
import os

# 示例使用
zip_file_path = 'archive (1).zip'  # 替换为你的 .zip 文件路径/media/GX/project/data
extract_to = 'BraTS2020'  # 替换为你希望解压到的目录

# 创建目标目录（如果不存在）
os.makedirs(extract_to, exist_ok=True)

# 打开 zip 文件并解压
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("解压完成")
# import tarfile
# import os
#
# # 示例使用
# tar_file_path = './BraTS_2021_Data_tar/BraTS2021_Training_Data.tar'  # 替换为你的 .tar 或 .tar.gz 文件路径
# extract_to = './BraTS_2021_Data'  # 替换为你希望解压到的目录
#
# # 创建目标目录（如果不存在）
# os.makedirs(extract_to, exist_ok=True)
#
# # 打开 tar 文件并解压
# with tarfile.open(tar_file_path, 'r') as tar_ref:
#     tar_ref.extractall(extract_to)
#
# print("解压完成")
