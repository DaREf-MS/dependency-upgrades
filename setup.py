from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="dependabot_dependency_upgrades",
    version="0.2.7",
    packages=find_packages(),
    install_requires=requirements,
    author="Ali Arabat",
    author_email="arabat50@gmail.com",
    description="Official package of Understanding Depenndabot Dependency Upgrades paper",
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
