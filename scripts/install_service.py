import os
import sys
import time

import paramiko

HOST = "raspberrypi.local"
USER = "admin"
PASSWORD = "Hm361485%"


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASSWORD, timeout=15,
              allow_agent=False, look_for_keys=False)
    sftp = c.open_sftp()
    sftp.put(os.path.abspath("scripts/laic.service"), "/tmp/laic.service")
    sftp.close()

    def sudo(cmd, t=90):
        i, o, e = c.exec_command("sudo -S -p '' " + cmd, timeout=t)
        i.write(PASSWORD + "\n")
        i.flush()
        out = o.read().decode(errors="replace")
        err = e.read().decode(errors="replace")
        print("$", cmd)
        if out.strip():
            print(out.encode("utf-8", "replace").decode("utf-8", "replace"))
        if err.strip():
            print("ERR:", err.encode("utf-8", "replace").decode("utf-8", "replace"))

    sudo("cp /tmp/laic.service /etc/systemd/system/laic.service")
    sudo("systemctl daemon-reload")
    sudo("systemctl enable laic")
    sudo("systemctl is-enabled laic")
    sudo("systemctl restart laic", t=120)
    time.sleep(4)
    sudo("systemctl status laic --no-pager")
    stdin, stdout, stderr = c.exec_command("ss -tln | grep 8080 || echo NO_LISTENER", timeout=30)
    print(stdout.read().decode(errors="replace"))
    c.close()


if __name__ == "__main__":
    sys.exit(main())