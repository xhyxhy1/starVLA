#!/usr/bin/env python3
"""
下载Hugging Face模型的脚本，支持断点续传和重试
"""
import os
import sys
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError
import time

def download_with_retry(repo_id, local_dir, max_retries=5, retry_delay=60):
    """
    带重试机制的模型下载
    
    Args:
        repo_id: Hugging Face仓库ID
        local_dir: 本地保存目录
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
    """
    for attempt in range(max_retries):
        try:
            print(f"\n{'='*60}")
            print(f"尝试下载 (第 {attempt + 1}/{max_retries} 次)")
            print(f"仓库: {repo_id}")
            print(f"保存到: {local_dir}")
            print(f"{'='*60}\n")
            
            # 设置环境变量
            os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'  # 禁用hf_transfer
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用镜像
            
            # 增加超时时间（通过环境变量）
            os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'  # 10分钟超时
            
            # 下载模型（支持断点续传）
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True,  # 启用断点续传
            )
            
            print(f"\n{'='*60}")
            print("✅ 下载成功！")
            print(f"{'='*60}\n")
            return True
            
        except HfHubHTTPError as e:
            print(f"\n❌ HTTP错误 (尝试 {attempt + 1}/{max_retries}):")
            print(f"错误信息: {str(e)}")
            
            if attempt < max_retries - 1:
                print(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                print("\n❌ 所有重试都失败了")
                return False
                
        except Exception as e:
            print(f"\n❌ 下载失败 (尝试 {attempt + 1}/{max_retries}):")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            
            if attempt < max_retries - 1:
                print(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                print("\n❌ 所有重试都失败了")
                return False
    
    return False

if __name__ == "__main__":
    repo_id = "StarVLA/Qwen3-VL-PI-LIBERO-4in1"
    local_dir = "results"
    
    # 确保目录存在
    os.makedirs(local_dir, exist_ok=True)
    
    success = download_with_retry(
        repo_id=repo_id,
        local_dir=local_dir,
        max_retries=10,  # 增加重试次数
        retry_delay=120  # 2分钟延迟
    )
    
    if not success:
        print("\n💡 建议:")
        print("1. 检查网络连接")
        print("2. 尝试使用代理或VPN")
        print("3. 检查Hugging Face token权限")
        print("4. 稍后重试")
        sys.exit(1)
    else:
        print(f"\n模型已下载到: {os.path.abspath(local_dir)}")
        sys.exit(0)




