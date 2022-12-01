#!/usr/bin/env python3
import os
import platform
import subprocess
from pathlib import Path


def build_arm64_mac_binaries():
    here = Path(__file__).parent.resolve()
    build_path = f"{here}/build/arm64-mac"
    if not os.path.exists(build_path):
        os.makedirs(build_path)

    subprocess.run(["cmake",
                    "-S", here,
                    "-B", build_path,
                    "-D", f"CMAKE_INSTALL_PREFIX={here}/install/arm64-mac"],
                   check=True)
    subprocess.run(["make", "-C", build_path, "-j8"], check=True)
    subprocess.run(["make", "-C", build_path, "install"], check=True)


def build_x64_linux_binaries():
    here = Path(__file__).parent.resolve()
    build_path = f"{here}/build/x64-linux"
    if not os.path.exists(build_path):
        os.makedirs(build_path)

    subprocess.run(["cmake",
                    "-S", here,
                    "-B", build_path,
                    "-D", f"CMAKE_INSTALL_PREFIX={here}/install/x64-linux"])
    subprocess.run(["make", "-C", build_path, "-j8"])
    subprocess.run(["make", "-C", build_path, "install"])


def main():
    here = Path(__file__).parent.resolve()
    subprocess.run(["python3", f"{here}/bootstrap.py"], check=True)

    if platform.system() == "Darwin":
        build_arm64_mac_binaries()
        return
    elif platform.system() == "Linux":
        build_x64_linux_binaries()
        return

    raise Exception(f"librgbd build not supported.")


if __name__ == "__main__":
	main()
