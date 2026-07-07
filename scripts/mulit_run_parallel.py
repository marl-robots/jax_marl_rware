import subprocess
import time

import psutil

def multi_run(min_free_gb=1):
    venv_python = r"C:\test\.venv\Scripts\python.exe"

    cmds = [
        [venv_python, "-m", "scripts.train_mappo", "--no-rnn", "--updates", "20"],
        [
            venv_python,
            "-m",
            "scripts.train_mappo",
            "--algo",
            "maa2c",
            "--no-rnn",
            "--updates",
            "20",
        ],
        [
            venv_python,
            "-m",
            "scripts.train_mappo",
            "--algo",
            "ippo",
            "--no-rnn",
            "--updates",
            "20",
        ],
        [
            venv_python,
            "-m",
            "scripts.train_mappo",
            "--algo",
            "ia2c",
            "--no-rnn",
            "--updates",
            "20",
        ],
    ]

    processes = []

    for i, cmd in enumerate(cmds):

        # --- WAIT FOR ENOUGH FREE RAM ---

        while True:
            mem = psutil.virtual_memory()
            free_gb = mem.available / (1024**3)

            if free_gb >= min_free_gb:
                break
            print(f"Waiting for RAM... free={free_gb:.2f} GB, need {min_free_gb} GB")
            time.sleep(5)
        # --- START PROCESS ---

        p = subprocess.Popen(cmd)
        proc = psutil.Process(p.pid)

        # CPU affinity

        cpu_count = psutil.cpu_count(logical=True)
        if cpu_count is not None:
            core = i % cpu_count
        else:
            core = i % 1
        proc.cpu_affinity([core])

        print(f"Started {cmd} on CPU core {core}")
        processes.append(p)
    # Wait for all to finish

    for p in processes:
        p.wait()
    print("All processes finished.")


multi_run()
