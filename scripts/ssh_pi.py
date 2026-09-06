import argparse
import os
import paramiko
import posixpath
import sys
import time

HOST = "raspberrypi.local"
USER = "admin"
PASSWORD = "Hm361485%"
PORT = 22

def get_client():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=15, allow_agent=False, look_for_keys=False)
    return c

def run(client, command, timeout=600):
    print(f"$ {command}", flush=True)
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout, get_pty=True)
    out = ""
    for line in iter(stdout.readline, ""):
        print(line, end="", flush=True)
        out += line
    rc = stdout.channel.recv_exit_status()
    if rc != 0:
        err = stderr.read().decode(errors="replace")
        if err.strip():
            print(err, end="", flush=True)
    print(f"[exit {rc}]", flush=True)
    return rc

def copy_tree(client, local_dir, remote_dir, exclude=(".git", "build", "cmake-build")):
    sftp = client.open_sftp()
    os.makedirs("/tmp/laicopy", exist_ok=True)
    files = []
    for root, dirs, fnames in os.walk(local_dir):
        dirs[:] = [d for d in dirs if d not in exclude]
        for f in fnames:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, local_dir)
            files.append(rel)
    print(f"copying {len(files)} files to {remote_dir} ...", flush=True)
    total = 0.0
    for rel in files:
        local = os.path.join(local_dir, rel)
        remote = posixpath.join(remote_dir, rel)
        sftp.makedirs(posixpath.dirname(remote))
        total += os.path.getsize(local)
        sftp.put(local, remote)
    sftp.close()
    print(f"copied {len(files)} files ({total/1024/1024:.1f} MB)", flush=True)

def copy_one(client, local_path, remote_path, mode=None):
    sftp = client.open_sftp()
    sftp.makedirs(posixpath.dirname(remote_path))
    sftp.put(local_path, remote_path)
    if mode:
        sftp.chmod(remote_path, mode)
    sftp.close()
    print(f"copied {local_path} -> {remote_path}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "deploy", "tests", "setup", "status", "server", "fetch-model"])
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    client = get_client()
    try:
        if a.cmd == "run":
            run(client, " ".join(a.args))
        elif a.cmd == "status":
            run(client, "cat /proc/cpuinfo | grep -i 'Hardware\|Model\|Revision'; echo; cat /etc/os-release | head -3; echo; free -h; echo; ls /etc/udev/rules.d/ 2>/dev/null | grep -i vc4; echo; ls /usr/lib/arm-linux-gnueabihf/libOpenCL* 2>/dev/null; dpkg -l | grep -i -E 'opencl|vc4|mesa' 2>/dev/null")
        elif a.cmd == "setup":
            run(client, "sudo apt-get update -y")
            run(client, "sudo apt-get install -y cmake g++ libclc spirv-tools")
            run(client, "sudo apt-get install -y mesa-opencl-icd ocl-icd-libopencl1 opencl-headers 2>/dev/null || sudo apt-get install -y mesa-opencl-icd ocl-icd-libopencl1 opencl-headers")
            run(client, "dpkg -l | grep -i -E 'opencl|mesa'")
        elif a.cmd == "deploy":
            import re
            src = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            if "--models" in a.args:
                mm = a.args[a.args.index("--models") + 1]
                copy_tree(client, mm, "~/laic-models", exclude=())
            else:
                copy_tree(client, src, "~/laic")
            copy_one(client, os.path.join(src, "scripts", "test_endpoints.py"), "~/laic/scripts/test_endpoints.py", 0o755)
        elif a.cmd == "tests":
            run(client, "cd ~/laic && rm -rf build && cmake -S . -B build -DLAIC_BUILD_TESTS=ON -DLAIC_ENABLE_NATIVE=ON 2>&1")
            run(client, "cd ~/laic && cmake --build build -j2 2>&1")
            run(client, "cd ~/laic && ./build/tests/laic_tests 2>&1 || ./build/tests/test_core 2>&1 || find build -name laic_tests -exec {} \\;")
        elif a.cmd == "server":
            port = a.args[0] if a.args else "8080"
            modeldir = a.args[1] if len(a.args) > 1 else "~/laic-models"
            run(client, f"pkill -f laic_server; cd ~/laic && nohup ./build/laic_server {port} {modeldir} > /tmp/laic_server.log 2>&1 & sleep 2; cat /tmp/laic_server.log")
        elif a.cmd == "fetch-model":
            name = a.args[0] if a.args else "smollm-135m-instruct"
            run(client, f"cd ~/laic-models 2>/dev/null || mkdir -p ~/laic-models && cd ~/laic-models; ls -la .")
    finally:
        client.close()

if __name__ == "__main__":
    main()