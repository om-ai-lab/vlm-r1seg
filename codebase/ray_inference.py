import ray
import os
import subprocess
import re
from typing import List, Dict, Tuple


def get_ray_resources(num_gpus: int) -> Dict[str, float]:
    """
    为每个 GPU 创建一个资源
    """
    resources = {}
    for gpu_id in range(num_gpus):
        resources[f"gpu_{gpu_id}"] = 1.0
    return resources

@ray.remote
def run_inference_task(gpu_id: int, global_step: int, batch_size: int, model_path: str) -> Tuple[int, float]:
    """
    远程任务：运行 inference_segzero_batch_simple.py 并返回 global_step 和 giou
    """
    try:
        # 设置 CUDA_VISIBLE_DEVICES 环境变量
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"Task started on GPU {gpu_id} with global_step {global_step}")
        
        # 构建命令
        command = [
            "python", "inference_segzero_batch_simple.py",
            "--batch_size", str(batch_size),
            "--reasoning_model_path", model_path.format(global_step)
        ]
        print()
        
        # 执行命令并捕获输出
        # output = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True)
        try:
            output = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True)
        except subprocess.CalledProcessError as e:
            print(f"Command '{e.cmd}' returned non-zero exit status {e.returncode}")
            print(f"Output: {e.output}")
        # 从输出中提取 giou 值
        giou_match = re.search(r"giou: (\d+\.\d+)", output)
        if giou_match:
            giou = float(giou_match.group(1))
            print(f"Task completed on GPU {gpu_id} with global_step {global_step}, giou: {giou}")
            return (global_step, giou)
        else:
            print(f"Failed to extract giou from output on GPU {gpu_id} with global_step {global_step}")
            return (global_step, None)
    except Exception as e:
        print(f"Task failed on GPU {gpu_id} with global_step {global_step}: {str(e)}")
        return (global_step, None)

if __name__ == "__main__":
    # 配置参数
    batch_size = 100
    # base_model_path = "/training/zilun/Seg-Zero_original/workdir/run_qwen2_5_7b_refCOCOg_kl/global_step_{}/actor/huggingface"
    base_model_path = "/training/zilun/Seg-Zero/workdir/run_qwen2_5_7b_refCOCOg/global_step_{}/actor/huggingface"

    # GPU 列表
    num_gpus = 8  # 0 到 7
    
    # global_step 列表
    global_steps = list(range(50, 551, 50)) + [562]
    # global_steps = list(range(50, 401, 50))
    
    # 为每个 GPU 创建资源
    ray_resources = get_ray_resources(num_gpus)
    ray.init(resources=ray_resources)
    
    # 生成任务
    tasks = []
    for i, global_step in enumerate(global_steps):
        # 为每个任务分配一个 GPU
        gpu_id = i % num_gpus
        # 为每个任务请求对应的 GPU 资源
        tasks.append(
            run_inference_task.options(resources={f"gpu_{gpu_id}": 1.0}).remote(
                gpu_id=gpu_id,
                global_step=global_step,
                batch_size=batch_size,
                model_path=base_model_path
            )
        )
    
    # 执行任务并等待完成
    results = ray.get(tasks)
    
    # 汇总结果
    results_dict = {}
    for global_step, giou in results:
        if giou is not None:
            results_dict[global_step] = giou
    
    # 打印结果
    print("\n" + "=" * 50)
    print("Global Step and GIoU Results:")
    print("=" * 50)
    for global_step in sorted(results_dict.keys()):
        print(f"global_step: {global_step}, giou: {results_dict[global_step]:.4f}")
    print("=" * 50)
    
    # 关闭 Ray
    ray.shutdown()