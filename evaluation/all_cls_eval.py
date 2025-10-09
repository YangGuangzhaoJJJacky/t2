#!/usr/bin/env python3
"""
全面评估脚本：对所有节点和类别组合进行评估
"""

import os
import json
import subprocess
import shutil
import glob
from pathlib import Path
from datetime import datetime


def run_bash_command(command):
    """在前台执行bash命令，实时显示输出"""
    print(f"🚀 执行命令: {command}")
    try:
        # 使用subprocess.run在前台执行，实时显示输出
        result = subprocess.run(command, shell=True, check=True)
        return result.returncode == 0, "", ""
    except subprocess.CalledProcessError as e:
        print(f"❌ 命令执行失败，返回码: {e.returncode}")
        return False, "", f"Command failed with return code {e.returncode}"


def save_results_immediately(all_results, output_file):
    """立即保存当前结果到JSON文件"""
    try:
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"💾 结果已保存到: {output_file} (共{len(all_results)}条记录)")
    except Exception as e:
        print(f"⚠️ 保存结果失败: {e}")


def extract_test_acc_from_json(json_path):
    """从reinforce_log.json文件中提取最终的test_acc"""
    if not os.path.exists(json_path):
        print(f"⚠️ JSON文件不存在: {json_path}")
        return None
    
    try:
        with open(json_path, 'r') as f:
            content = f.read().strip()
        
        # 尝试解析整个文件作为单个JSON对象
        try:
            data = json.loads(content)
            if "test_acc" in data:
                print(f"✅ 成功提取test_acc: {data['test_acc']}")
                return data["test_acc"]
        except json.JSONDecodeError:
            # 如果不是单个JSON对象，尝试按行解析
            lines = content.split('\n')
            for line in reversed(lines):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if "test_acc" in data:
                            print(f"✅ 成功提取test_acc: {data['test_acc']}")
                            return data["test_acc"]
                    except json.JSONDecodeError:
                        continue
        
        print(f"⚠️ 未找到test_acc字段")
        return None
    except Exception as e:
        print(f"❌ 读取JSON文件失败 {json_path}: {e}")
        return None


def create_eval_script():
    """创建eval_cls.sh脚本"""
    script_content = '''#!/bin/bash
export HYDRA_FULL_ERROR=1
export HF_HOME=/mnt/data-raid/yangguangzhao/.cache
export PYTHONPATH=$PYTHONPATH:/home/yangguangzhao/t2/evaluation

# 读取传入的参数
NODE=${1:-0}
CLS=${2:-0}

TASK="aqua_rat"
NUM_ITERS=0  # 只评估，不训练

# 找对应的checkpoint
CHECKPOINT_PATH="/mnt/data-raid/yangguangzhao/t2/results_history/results_self_cls_math_9/$CLS/aqua_rat_1_mm1_qwen306b_RL-lr0.002-mGN0.001-klC0.01-rrN0CNone-st/policy_params.pt"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "错误: 找不到checkpoint文件: $CHECKPOINT_PATH"
    exit 1
fi

# 启动评估
CUDA_VISIBLE_DEVICES=0,1 python svd_reinforce_hydra.py \\
    base_model@_global_=qwen306b \\
    task@_global_=$TASK \\
    mode@_global_=training \\
    optimization@_global_=reinforce \\
    task_loader.node=$NODE \\
    +output_path="results/$NODE" \\
    num_iters=$NUM_ITERS \\
    load_ckpt="$CHECKPOINT_PATH"
'''
    
    script_path = "/mnt/data-raid/yangguangzhao/t2/scripts/eval_cls.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # 添加执行权限
    os.chmod(script_path, 0o755)
    print(f"创建评估脚本: {script_path}")


def main():
    """主评估函数"""
    print("开始全面评估...")
    
    # 创建eval_cls.sh脚本
    create_eval_script()
    
    # 创建结果存储
    all_results = []
    
    # 创建评估目录
    eval_dir = Path("/mnt/data-raid/yangguangzhao/t2/evaluation")
    eval_dir.mkdir(exist_ok=True)
    
    # 结果文件路径
    output_file = eval_dir / "all_cls_eval.json"
    
    # 双重循环：节点和类别
    for node in range(10):
        for cls in range(10):
            print(f"\n{'='*50}")
            print(f"评估 Node={node}, CLS={cls}")
            print(f"{'='*50}")
            
            # 设置环境变量
            env = os.environ.copy()
            env['NODE'] = str(node)
            env['CLS'] = str(cls)
            
            # 执行评估脚本
            command = f"bash scripts/eval_cls.sh {node} {cls}"
            
            success, stdout, stderr = run_bash_command(command)
            
            if success:
                print("✅ 评估成功")
                
                # 查找生成的results目录中的JSON文件
                json_pattern = f"results/{node}/*/reinforce_log.json"
                json_files = glob.glob(json_pattern)
                
                test_acc = None
                if json_files:
                    json_path = json_files[0]  # 取第一个匹配的文件
                    test_acc = extract_test_acc_from_json(json_path)
                    print(f"📊 提取到test_acc: {test_acc}")
                else:
                    print("⚠️ 未找到reinforce_log.json文件")
                
                # 记录结果
                result_entry = {
                    "node": node,
                    "cls": cls,
                    "test_acc": test_acc,
                    "status": "success",
                    "timestamp": datetime.now().isoformat()
                }
                all_results.append(result_entry)
                
                # 立即保存结果
                save_results_immediately(all_results, output_file)
                
                # 清理results目录
                results_dir = f"results/{node}"
                if os.path.exists(results_dir):
                    shutil.rmtree(results_dir)
                    print(f"🗑️ 清理目录: {results_dir}")
                    
            else:
                print(f"❌ 评估失败")
                if stderr:
                    print(f"错误信息: {stderr}")
                
                # 记录失败结果
                result_entry = {
                    "node": node,
                    "cls": cls,
                    "test_acc": None,
                    "status": "failed",
                    "error": stderr,
                    "timestamp": datetime.now().isoformat()
                }
                all_results.append(result_entry)
                
                # 立即保存结果（包括失败的）
                save_results_immediately(all_results, output_file)
    
    print(f"\n🎉 评估完成！最终结果文件: {output_file}")
    
    # 生成汇总报告
    generate_summary_report(all_results, eval_dir)


def generate_summary_report(results, eval_dir):
    """生成汇总报告"""
    print(f"\n{'='*50}")
    print("📈 评估汇总报告")
    print(f"{'='*50}")
    
    total_evals = len(results)
    successful_evals = len([r for r in results if r['status'] == 'success' and r['test_acc'] is not None])
    failed_evals = total_evals - successful_evals
    
    print(f"总评估数: {total_evals}")
    print(f"成功评估: {successful_evals}")
    print(f"失败评估: {failed_evals}")
    
    if successful_evals > 0:
        test_accs = [r['test_acc'] for r in results if r['test_acc'] is not None]
        avg_acc = sum(test_accs) / len(test_accs)
        max_acc = max(test_accs)
        min_acc = min(test_accs)
        
        print(f"\n📊 测试准确率统计:")
        print(f"平均准确率: {avg_acc:.4f}")
        print(f"最高准确率: {max_acc:.4f}")
        print(f"最低准确率: {min_acc:.4f}")
        
        # 找到最佳组合
        best_result = max([r for r in results if r['test_acc'] is not None], key=lambda x: x['test_acc'])
        print(f"\n🏆 最佳组合: Node={best_result['node']}, CLS={best_result['cls']}, Acc={best_result['test_acc']:.4f}")
    
    # 创建结果矩阵
    create_result_matrix(results, eval_dir)


def create_result_matrix(results, eval_dir):
    """创建结果矩阵"""
    import numpy as np
    
    # 创建10x10矩阵
    matrix = np.full((10, 10), np.nan)
    
    for result in results:
        if result['test_acc'] is not None:
            matrix[result['node']][result['cls']] = result['test_acc']
    
    # 保存矩阵到文件
    matrix_file = eval_dir / "result_matrix.txt"
    with open(matrix_file, 'w') as f:
        f.write("结果矩阵 (行=Node, 列=CLS):\n")
        f.write("    " + "  ".join([f"CLS{i}" for i in range(10)]) + "\n")
        for i in range(10):
            row = f"N{i}  "
            for j in range(10):
                if np.isnan(matrix[i][j]):
                    row += "  N/A  "
                else:
                    row += f"{matrix[i][j]:.3f} "
            f.write(row + "\n")
    
    print(f"📋 结果矩阵已保存到: {matrix_file}")


if __name__ == "__main__":
    main()