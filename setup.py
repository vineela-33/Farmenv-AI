from setuptools import setup, find_packages

setup(
    name="farmenv-ai",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "flask",
        "flask-cors",
        "numpy", 
        "requests",
        "openai"
    ],
    python_requires=">=3.9",
)
