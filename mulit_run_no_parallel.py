import subprocess


def multi_run():
    cmd = """

    & C:\\test\\.venv\\Scripts\\Activate.ps1

    python -m scripts.train_mappo --algo ia2c --full-metrics

    """
    print(f"Start {cmd}")
    subprocess.run(["powershell", "-Command", cmd])
    print(f"Done {cmd}")
    
    cmd = """

    & C:\\test\\.venv\\Scripts\\Activate.ps1

    python -m scripts.train_mappo --algo ippo --full-metrics

    """
    print(f"Start {cmd}")
    subprocess.run(["powershell", "-Command", cmd])
    print(f"Done {cmd}")
    
    cmd = """

    & C:\\test\\.venv\\Scripts\\Activate.ps1

    python -m scripts.train_mappo --algo maa2c --full-metrics

    """
    print(f"Start {cmd}")
    subprocess.run(["powershell", "-Command", cmd])
    print(f"Done {cmd}")
    
    cmd = """

    & C:\\test\\.venv\\Scripts\\Activate.ps1

    python -m scripts.train_mappo --full-metrics

    """
    print(f"Start {cmd}")
    subprocess.run(["powershell", "-Command", cmd])
    print(f"Done {cmd}")


multi_run()
