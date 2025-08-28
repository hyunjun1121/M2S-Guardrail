#!/usr/bin/env python3
"""
GPU에서 실행 중인 프로세스 정보를 확인하는 스크립트
서버에서 실행하여 PID 697115가 무엇인지 파악
"""

import subprocess
import os
import psutil

def check_process_details(pid):
    """특정 PID의 상세 정보 확인"""
    try:
        process = psutil.Process(pid)
        
        print(f"=== Process {pid} Details ===")
        print(f"Name: {process.name()}")
        print(f"Command: {' '.join(process.cmdline())}")
        print(f"Status: {process.status()}")
        print(f"User: {process.username()}")
        print(f"CPU %: {process.cpu_percent()}")
        print(f"Memory %: {process.memory_percent():.2f}%")
        print(f"Memory Info: {process.memory_info()}")
        print(f"Create Time: {process.create_time()}")
        print(f"Working Directory: {process.cwd()}")
        
        # 환경 변수 확인 (중요한 것들만)
        env = process.environ()
        important_env = ['CUDA_VISIBLE_DEVICES', 'HF_HOME', 'TRANSFORMERS_CACHE', 'PYTORCH_CUDA_ALLOC_CONF']
        print(f"\nImportant Environment Variables:")
        for key in important_env:
            if key in env:
                print(f"  {key}: {env[key]}")
        
        # 열린 파일들 확인 (처음 10개만)
        print(f"\nOpen Files (first 10):")
        try:
            files = process.open_files()[:10]
            for f in files:
                print(f"  {f.path}")
        except:
            print("  Could not retrieve open files")
        
        # 네트워크 연결 확인
        print(f"\nNetwork Connections:")
        try:
            connections = process.connections()
            for conn in connections[:5]:  # 처음 5개만
                print(f"  {conn.laddr} -> {conn.raddr if conn.raddr else 'N/A'} ({conn.status})")
        except:
            print("  Could not retrieve network connections")
            
        return True
        
    except psutil.NoSuchProcess:
        print(f"Process {pid} not found")
        return False
    except Exception as e:
        print(f"Error checking process {pid}: {e}")
        return False

def check_gpu_usage():
    """nvidia-smi를 통한 GPU 사용량 상세 확인"""
    try:
        print("=== GPU Memory Usage Details ===")
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', 
                               '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                gpu_info = line.split(', ')
                gpu_id, name, mem_used, mem_total, util = gpu_info
                print(f"GPU {gpu_id} ({name}): {mem_used}MB/{mem_total}MB ({util}% util)")
        
        print("\n=== Process-specific GPU Usage ===")
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', 
                               '--format=csv,noheader'], 
                               capture_output=True, text=True)
        
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    print(f"  {line}")
                    
    except Exception as e:
        print(f"Error checking GPU usage: {e}")

def check_python_processes():
    """시스템의 모든 python 프로세스 확인"""
    print("=== All Python Processes ===")
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_percent']):
        try:
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                cmd = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else 'N/A'
                print(f"PID {proc.info['pid']}: {cmd[:100]}{'...' if len(cmd) > 100 else ''}")
                print(f"  Memory: {proc.info['memory_percent']:.2f}%")
                print()
        except:
            continue

if __name__ == "__main__":
    print("GPU Process Analysis")
    print("=" * 50)
    
    # GPU 사용량 확인
    check_gpu_usage()
    
    print()
    
    # 특정 PID 상세 확인
    target_pid = 697115
    if check_process_details(target_pid):
        print(f"\nProcess {target_pid} analysis completed")
    
    print()
    
    # 모든 Python 프로세스 확인
    check_python_processes()
    
    print("Analysis completed")