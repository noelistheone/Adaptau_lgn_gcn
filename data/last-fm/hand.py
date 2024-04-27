# 打开文件
with open('train-1.txt', 'r') as file:
    # 读取文件内容
    content = file.read()

# 将制表符替换为空格
content = content.replace('\t', ' ')

# 打开文件以写入替换后的内容
with open('train.txt', 'w') as file:
    # 写入替换后的内容
    file.write(content)

print("制表符已成功替换为空格，并保存到 'your_output_file.txt' 中。")
