from setuptools import setup, find_packages

name = "BDF"
version = "0.0.1"
author = "Jonas Breuling"
author_email = "jonas.breuling@inm.uni-stuttgart.de"
url = "https://github.com/JonasBreuling/BDF"
description = "BDF solver for implicit differential algebraic equations."
long_description = ""
license = "LICENSE"

setup(
    name=name,
    version=version,
    author=author,
    author_email=author_email,
    description=description,
    long_description=long_description,
    install_requires=[
        "numpy>=1.21.3",
        "scipy>=1.10.1",
        "black>=22.1.0",
        "matplotlib",
    ],
    packages=find_packages(),
    python_requires=">=3.10",
)
