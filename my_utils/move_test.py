from datasets import load_dataset

dataset = load_dataset("yangguangzhaojjj/aqua_rat_cls", split="test")

dataset.push_to_hub("yangguangzhaojjj/aqua_rat_cls_new", split="test")

