"""Local scheduler stand-in. Run an identical argv once for every owner."""
import os
import subprocess
import sys
import time


def main(count, index_env, *command):
    tasks = []
    try:
        for index in range(int(count)):
            tasks.append(subprocess.Popen(command, env={**os.environ, index_env: str(index)}))
        while True:
            status = [task.poll() for task in tasks]
            if any(code not in (None, 0) for code in status):
                return 1
            if all(code == 0 for code in status):
                return 0
            time.sleep(.05)
    finally:
        for task in tasks:
            if task.poll() is None:
                task.terminate()
        for task in tasks:
            try:
                task.wait(timeout=2)
            except subprocess.TimeoutExpired:
                task.kill()
                task.wait()


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
