import json, glob, torch, subprocess
import os
from typing import List

def train(node_id):
    print(f"🧠 Node {node_id} is training...")
    result = subprocess.run(["bash", "scripts/train_task_expert.sh", str(node_id)])
    if result.returncode != 0:
        print(f"❌ Node {node_id} training failed.")
        print(result.stderr.decode())
    else:
        print(f"✅ Node {node_id} finished training.")

def load_state_dicts(n_nodes):
    return [torch.load(sorted(glob.glob(f"results/{n}/*/policy_params.pt"))[-1]) for n in range(n_nodes)]

def save_state_dicts(models, n_nodes):
    for i in range(n_nodes):
        save_path = sorted(glob.glob(f"results/{i}/*/policy_params.pt"))[-1]
        torch.save(models[i], save_path)

def exchange_multi_node(models, contact, fl_coeff=0.5):
    n_node = len(models)
    updated_models = []

    for n in range(n_node):
        local = models[n]
        neighbors = contact.get(str(n), [])
        if not neighbors:
            updated_models.append(local)
            continue

        nbr_models = [models[k] for k in neighbors]
        new_state = {}
        for key in local:
            agg = torch.zeros_like(local[key])
            for neighbor_model in nbr_models:
                diff = neighbor_model[key] - local[key]
                agg += diff
            new_state[key] = local[key] + (fl_coeff / (len(nbr_models) + 1)) * agg

        updated_models.append(new_state)
    return updated_models

def federated_multi_exchange(contact_path, fl_coeff=0.5):
    with open(contact_path, 'r') as f:
        contact_list = json.load(f)
    
    n_nodes = len(contact_list[0])
    for iter_no, contact in enumerate(contact_list):
        print(f"\n🔁 ===== EXCHANGE ITER {iter_no + 1} =====")

        print(f"🚀 Training all nodes before exchange {iter_no + 1}...")
        for i in range(n_nodes): 
            train(i)

        # 读取模型参数
        models = load_state_dicts(n_nodes)

        # 执行参数交换
        models = exchange_multi_node(models, contact, fl_coeff)

        # 保存更新后的模型参数
        save_state_dicts(models, n_nodes)

        print(f"✅ Exchange {iter_no + 1} completed and saved.\n")

if __name__ == "__main__":
    federated_multi_exchange("contact_pattern/example.json")
