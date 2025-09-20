import json

# 假设你的 JSON 存在一个文件 input.json
with open("contact_pattern/rwp_n10_a0500_r100_p10_s10.json", "r", encoding="utf-8") as f:
    data = json.load(f)

group_size = 10
unique_data = []

# 按每10个分组
for i in range(0, len(data), group_size):
    group = data[i:i+group_size]
    if group:  # 确保组不为空
        unique_data.append(group[0])  # 每组只取第一个

# 输出结果
with open("contact_pattern/rwp_n10_a0500_r100_p1_s10.json", "w", encoding="utf-8") as f:
    json.dump(unique_data, f, ensure_ascii=False, indent=4)

print("原始长度:", len(data))
print("去重后长度:", len(unique_data))
