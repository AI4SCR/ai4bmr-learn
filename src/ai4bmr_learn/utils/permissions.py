import grp
import os
import subprocess
from pathlib import Path
import textwrap

def set_permissions(path: str):
    group_name = "spatial_100794-pr-g"
    gid = grp.getgrnam(group_name).gr_gid
    uid = os.getuid()

    path = Path(path).resolve()

    cmd = f"""
    # Change group only for files owned by this user
    find "{path}" -user {uid} -exec chgrp {gid} {{}} +

    # Give group read/write and execute on directories
    find "{path}" -user {uid} -exec chmod g+rwX {{}} +

    # Set setgid bit on directories only (so new files inherit group)
    find "{path}" -type d -user {uid} -exec chmod g+s {{}} +
    """
    cmd = textwrap.dedent(cmd)

    subprocess.run(cmd, shell=True, check=True)