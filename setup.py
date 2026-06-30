from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
requires_list = []
with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requires_list.append(line)

# Git-based dependencies not in requirements.txt can be appended here.
#requires_list.append("muon @ git+https://github.com/KellerJordan/Muon.git")


setup(
    name="class_free_guide",
    version="1.0",
    description="class-free guide by flow matching",
    author="David Zong",
    author_email="805483796@qq.com",
    packages=find_packages(),
    install_requires=requires_list,
)
