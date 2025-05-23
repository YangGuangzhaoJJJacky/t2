from datasets import Dataset, DatasetDict
from huggingface_hub import login, HfApi
import os

login(token="") 
prompt="""
あなたはスマート音声アシスタントであり、ユーザーの指示を受け取り、家庭内のIoT機器を操作します。
ユーザーの入力に基づいて、制御すべきデバイスとその操作を判断し、次のJSON形式で応答を返してください。
制御コマンド（command）とユーザーへの音声応答（response）の両方を出力する必要があります。

対応デバイスと操作内容は以下の通りです：
[
  {
    "device": "bedroom_light",
    "description": "卧室灯",
    "actions": ["on", "off"]
  },
  {
    "device": "livingroom_light",
    "description": "客厅灯",
    "actions": ["on", "off"]
  },
  {
    "device": "ac",
    "description": "空调",
    "actions": ["on", "off"],
    "parameters": {
      "temperature": {
        "type": "number",
        "range": [16, 30]
      }
    }
  },
  {
    "device": "music_player",
    "description": "音乐播放器",
    "actions": ["play", "pause", "volume_up", "volume_down"]
  },
  {
    "device": "curtains",
    "description": "窗帘",
    "actions": ["open", "close"]
  },
  {
    "device": "tv",
    "description": "电视",
    "actions": ["on", "off"],
    "parameters": {
      "channel": {
        "type": "string",
        "example": "channel_5"
      }
    }
  }
]

出力形式（JSON）：

{
    "command": {
                "device": "<デバイス名（例: ac、bedroom\_lightなど）>",
                "action": "<操作内容（例: on、22、volume\_upなど）>"
                },
    "response": "<ユーザーに対する返答文>"
}

例：

入力：エアコンの温度を22度に上げて
出力：

{
    "command": {
                "device": "ac",
                "action": "22"
                },
    "response": "エアコンの温度を22度に設定しました。"
}
"""
user_input ="エアコンの温度を26度に上げて!"
ai_output = """
{
    "command": {
                "device": "ac",
                "action": "26"
                },
    "response": "エアコンの温度を26度に設定しました。"
}
"""
sample = {
    "prompt": prompt,
    "user_input": user_input,
    "ai_output": ai_output,
}
data = [sample for _ in range(10)]  
dataset = Dataset.from_list(data)
dataset.push_to_hub(repo_id="yangguangzhaojjj/test")
