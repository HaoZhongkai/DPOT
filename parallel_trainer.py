import yaml
import subprocess
import time
import random
import argparse
import os
import sys


class ParallelTrainer:
    def __init__(self, yaml_path):
        self.config = self.load_yaml(yaml_path)
        self.processes = {}  # Store {gpu_id: process} for monitoring

    @staticmethod
    def load_yaml(yaml_path):
        with open(yaml_path, 'r') as file:
            return yaml.safe_load(file)

    def get_available_gpus(self):
        cmd = "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
        lines = output.split("\n")

        # Extract GPUs with memory usage less than a certain threshold (e.g., 500 MB)
        free_gpus = [int(line.split(",")[0].strip()) for line in lines if
                     int(line.split(",")[1].strip().split(" ")[0]) < 800]

        # If 'device' is specified in the config, filter the available GPUs
        if 'device' in self.config:
            specified_gpus_str = self.config['device']
            specified_gpus = [int(gpu) for gpu in specified_gpus_str.split(',')]
            free_gpus = [gpu for gpu in free_gpus if gpu in specified_gpus]

        return free_gpus

    def run_task(self, task_args, gpu):
        python_executable = sys.executable  # This will give the path to the currently running Python interpreter
        # base_args = [python_executable, self.config['file']]
        base_args = [self.config['file']]
        # Load all non-task parameters excluding 'name' and 'file'
        for key, value in self.config.items():
            if 'task' not in key and key not in ['name', 'file', 'device']:
                if isinstance(value, bool):
                    if value:  # If the value is True, we just add the flag, otherwise we skip it
                        base_args.append(f'--{key}')
                elif isinstance(value, list):
                    base_args.append(f'--{key}')
                    for item in value:
                        base_args.append(str(item))
                else:
                    base_args.extend([f'--{key}', str(value)])
        for key, value in task_args.items():
            if isinstance(value, list):
                base_args.append(f'--{key}')
                for item in value:
                    base_args.append(str(item))
            else:
                base_args.extend([f'--{key}', str(value)])
            # base_args.extend([f'--{key}', str(value)])

        log_path = f"{self.config['model']}_{self.config['dataset']}_{time.strftime('%m%d_%H_%M_%S')}"
        base_args.extend(['--log_path', log_path])
        num_gpus = task_args.get('num_gpus', self.config.get('num_gpus', 1))
        # print(f"Starting task with args: {' '.join(base_args)}")
        cmd = f"CUDA_VISIBLE_DEVICES={','.join(map(str, gpu))} accelerate launch --num_processes={num_gpus} --multi_gpu --main_process_port 50{random.randint(10, 99)} {' '.join(base_args)}:"

        process = subprocess.Popen(cmd,shell=True)
        print(f"Task started on GPU {gpu}")
        self.processes[tuple(gpu)] = process

    def monitor_tasks(self, tasks):
        while True:
            for gpu_key, process in list(self.processes.items()):
                ret_code = process.poll()
                if ret_code is not None:  # Process has finished
                    print(f"Task on GPUs {', '.join(map(str, gpu_key))} has finished.")  # 更新打印语句以显示所有GPU IDs
                    del self.processes[gpu_key]
                    return gpu_key

            if tasks:  # Only check for available GPUs if there are pending tasks
                available_gpus = self.get_available_gpus()
                next_task_num_gpus = tasks[0].get('num_gpus',self.config.get('num_gpus', 1))  # 获取num_gpus值，如果没有指定，则默认为1
                # next_task_num_gpus = tasks[0]['num_gpus']  # 假设每个任务字典中都有一个'num_gpus'键
                if len(available_gpus) >= next_task_num_gpus:
                    return available_gpus[:next_task_num_gpus]
                # free_gpus = [gpu for gpu in available_gpus if gpu not in self.processes.keys()]
                # if free_gpus:
                #     return free_gpus[0]

            time.sleep(5)
        return None

    def start(self):
        print("Starting Trainer...")

        # Load and check tasks from the config
        task_params = self.config.get('tasks', {})

        # Check if 'tasks' exists in the config
        if task_params:
            n = None
            for key, values in task_params.items():
                if not isinstance(values, list):
                    task_params[key] = [values]
                if n is None:
                    n = len(task_params[key])
                elif n != len(task_params[key]):
                    raise ValueError("All parameter sequences in 'tasks' should have the same length.")

            # Generate tasks based on the new format
            tasks = []
            num_tasks = n
            for i in range(num_tasks):
                task = {key: values[i] for key, values in task_params.items()}
                tasks.append(task)
        else:
            # If no 'tasks' specified, consider the entire config as one task (excluding certain keys)
            tasks = [{k: v for k, v in self.config.items() if k not in ['name', 'file', 'device', 'tasks']}]

        # Start tasks for initially available GPUs
        while tasks:
            available_gpus = self.get_available_gpus()  # Get the updated list of available GPUs
            task = tasks[0]  # Look at the first task in the queue, but don't pop it yet
            num_gpus_required = task.get('num_gpus', self.config.get('num_gpus',1))  # Get the number of GPUs required for this task

            # Check if there are enough available GPUs for this task
            if len(available_gpus) >= num_gpus_required:
                tasks.pop(0)  # Now pop the task from the queue
                gpu_ids_for_task = available_gpus[:num_gpus_required]  # Get the IDs of the GPUs for this task
                self.run_task(task, gpu_ids_for_task)  # Start the task with the required GPUs
                available_gpus = available_gpus[num_gpus_required:]  # Update the list of available GPUs
                time.sleep(20)  # Optional: wait for some time before checking again
            else:
                break  # Not enough GPUs for this task, wait for some to become available

        # Wait for tasks to finish and start new tasks on free GPUs
        while tasks:
            free_gpu = self.monitor_tasks(tasks)
            if free_gpu is not None:
                task = tasks.pop(0)
                self.run_task(task, list(free_gpu))
                time.sleep(20)

        print("All tasks are either running or queued. Waiting for them to complete...")
        while self.processes:
            self.monitor_tasks([])

        print("All tasks have finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parallel Training or pretraining Neural Operators')
    parser.add_argument('--config_file',type=str,default='pretrain_tiny.yaml')
    args = parser.parse_args()

    trainer = ParallelTrainer(os.path.join('configs',args.config_file))
    trainer.start()
