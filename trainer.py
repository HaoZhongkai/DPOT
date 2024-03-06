import yaml
import subprocess
import time
import argparse
import os
import sys


class Trainer:
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
        base_args = [python_executable, self.config['file']]

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
        base_args.extend(['--log_path', log_path, '--gpu', str(gpu)])
        # print(f"Starting task with args: {' '.join(base_args)}")
        process = subprocess.Popen(base_args)
        print(f"Task started on GPU {gpu}")
        self.processes[gpu] = process

    def monitor_tasks(self, tasks):
        while True:
            for gpu, process in list(self.processes.items()):
                ret_code = process.poll()
                if ret_code is not None:  # Process has finished
                    print(f"Task on GPU {gpu} has finished.")
                    del self.processes[gpu]
                    return gpu

            if tasks:  # Only check for available GPUs if there are pending tasks
                available_gpus = self.get_available_gpus()
                free_gpus = [gpu for gpu in available_gpus if gpu not in self.processes.keys()]
                if free_gpus:
                    return free_gpus[0]

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

        available_gpus = self.get_available_gpus()

        # Start tasks for initially available GPUs
        while tasks and available_gpus:
            task = tasks.pop(0)
            gpu = available_gpus.pop(0)
            self.run_task(task, gpu)
            time.sleep(20)

        # Wait for tasks to finish and start new tasks on free GPUs
        while tasks:
            free_gpu = self.monitor_tasks(tasks)
            if free_gpu is not None:
                task = tasks.pop(0)
                self.run_task(task, free_gpu)
                time.sleep(20)

        print("All tasks are either running or queued. Waiting for them to complete...")
        while self.processes:
            self.monitor_tasks([])

        print("All tasks have finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training or pretraining Neural Operators')
    parser.add_argument('--config_file',type=str,default='ns2d_pretrain.yaml')
    args = parser.parse_args()

    trainer = Trainer(os.path.join('configs',args.config_file))
    trainer.start()
