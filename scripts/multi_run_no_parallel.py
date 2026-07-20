import subprocess


def multi_run():

    cmd = """

    & C:\\test\\.venv\\Scripts\\Activate.ps1

    python -m scripts.train_seac --full-metrics

    """
    print(f"Start {cmd}")
    subprocess.run(["powershell", "-Command", cmd])
    print(f"Done {cmd}")

if __name__=="__main__":
    multi_run()
