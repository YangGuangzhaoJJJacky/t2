import requests
from datasets import load_dataset

OPENAI_API_KEY = ""  
DATASET_NAME = "deepmind/aqua_rat"  

dataset = load_dataset(DATASET_NAME, split="train")
dataset = dataset.shuffle()  # 每次随机打乱
dataset = dataset.select(range(10000))
math_questions = [item['question'] + "\nOptions: \n" + "\n".join(item["options"]) for item in dataset] 

questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(math_questions)]) 

headers = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
}

prompt = f"""
我现在有好多数学题 但是我想给这些数学题进行分类 但我不知道应该分哪些类 你根据我给你看的这些题告我应该分哪些类 给我总结出10个类别
"""

data = {
    "model": "gpt-4.1",
    "messages": [{"role": "system", "content": prompt},{"role": "user", "content": questions_text}],
}

response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers=headers,
    json=data
)

if response.status_code == 200:
    result = response.json()
    print("分类结果：")
    print(result['choices'][0]['message']['content'])
else:
    print(f"请求失败: {response.status_code}")
    print(response.text)