import json
import requests
import concurrent.futures
import time
from copy import deepcopy

# 原始长 prompt 模板（用 {question_num} 占位符替代固定的序号）
QUESTION_TEMPLATE = """
这个是第{question_num}个问题, Mooncake KV Cache 是一种用于大模型推理场景的分布式 KV Cache 存储与传输机制。
它主要用于 disaggregated prefill/decode 架构中，实现不同推理节点之间的 KV Cache 共享。

请详细介绍以下内容：

1. 什么是 KV Cache
2. 为什么大模型推理需要 KV Cache
3. Mooncake 的整体架构
4. MooncakeStore 的作用
5. vLLM 与 Mooncake 的集成流程
6. Prefill 与 Decode 分离的原理
7. RDMA 在 KV Transfer 中的作用
8. AscendDirectTransport 的工作机制
9. KV Cache Put/Get 流程
10. connector 的初始化流程
11. 分页 KV Cache 与连续 KV Cache 的区别
12. multi_layer_kv_transfer 的实现逻辑
13. Mooncake 在 A2 与 A3 上的差异
14. HCCL 与 Mooncake 的关系
15. 为什么大规模 MoE 更依赖 KV Cache
16. DeepSeek 场景中的 KV Cache 复用
17. Prefix Cache 与 Mooncake 的区别
18. KV Cache 生命周期管理
19. GPU/NPU 到 CPU 的 D2H 过程
20. 常见 Mooncake 初始化失败原因

下面开始详细说明：
"""

def build_long_prompt(question_num):
    """为指定序号生成长 prompt，重复8次内容"""
    single_text = QUESTION_TEMPLATE.format(question_num=question_num)
    return single_text * 8

def send_request(request_id, url, headers, model_name, max_tokens, temperature, timeout):
    """发送单个请求的任务函数"""
    prompt = build_long_prompt(request_id)
    print(f"[Request {request_id}] Prompt length: {len(prompt)} characters")
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    
    start_time = time.time()
    try:
        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload),
            timeout=timeout
        )
        elapsed_time = time.time() - start_time
        print(f"[Request {request_id}] Status: {response.status_code}, Time: {elapsed_time:.2f}s")
        
        # 尝试解析返回内容
        if response.status_code == 200:
            try:
                result = response.json()
                response_text = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                print(f"[Request {request_id}] Response length: {len(response_text)} characters")
            except:
                pass
        
        return request_id, response.status_code, elapsed_time
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"[Request {request_id}] Error: {e}, Time: {elapsed_time:.2f}s")
        return request_id, None, elapsed_time

def main():
    # ========== 配置参数 ==========
    url = "http://127.0.0.1:8000/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    model_name = "Qwen3-8B"
    max_tokens = 512
    temperature = 0.7
    timeout = 6000
    
    # 用户可配置的参数
    total_requests = 10      # 一共要发送多少条数据
    concurrency = 10          # 每次并发多少
    
    # 可选：设置请求间隔（秒），避免瞬间压力过大，默认为0
    request_interval = 0.1   # 每启动一个请求间隔0.1秒
    # ============================
    
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Total requests: {total_requests}")
    print(f"  Concurrency: {concurrency}")
    print(f"  Request interval: {request_interval}s")
    print(f"  Model: {model_name}")
    print(f"  URL: {url}")
    print("=" * 60)
    
    results = []
    success_count = 0
    fail_count = 0
    
    # 使用线程池执行并发请求
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {}
        
        # 分批提交任务，控制并发数
        for i in range(1, total_requests + 1):
            # 提交任务
            future = executor.submit(
                send_request, i, url, headers, model_name, 
                max_tokens, temperature, timeout
            )
            futures[future] = i
            
            # 控制提交间隔，避免瞬间大量请求
            if request_interval > 0:
                time.sleep(request_interval)
            
            # 可选：打印进度
            if i % 10 == 0:
                print(f"\n[Progress] Submitted {i}/{total_requests} requests\n")
        
        print(f"\nAll {total_requests} requests submitted, waiting for results...\n")
        
        # 收集结果
        for future in concurrent.futures.as_completed(futures):
            request_id, status_code, elapsed_time = future.result()
            results.append({
                'id': request_id,
                'status': status_code,
                'time': elapsed_time
            })
            
            if status_code == 200:
                success_count += 1
            else:
                fail_count += 1
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total requests: {total_requests}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Success rate: {success_count/total_requests*100:.2f}%")
    
    if results:
        times = [r['time'] for r in results if r['status'] == 200]
        if times:
            print(f"\nResponse time statistics (successful requests only):")
            print(f"  Average: {sum(times)/len(times):.2f}s")
            print(f"  Min: {min(times):.2f}s")
            print(f"  Max: {max(times):.2f}s")
    
    print("=" * 60)

if __name__ == "__main__":
    main()