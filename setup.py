from setuptools import setup, find_packages
from codecs import open
from os import path


ext_modules = []

here = path.abspath(path.dirname(__file__))
requires_list = []
with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    for line in f:
        requires_list.append(str(line))


setup(
    name="class_free_guide",
    version="1.0",
    description="class-free guide by flow matching",
    author="David Zong",
    author_email="805483796@qq.com",
    packages=find_packages(where=""),
    install_requires=requires_list,
)
