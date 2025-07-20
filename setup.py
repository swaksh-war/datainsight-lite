from setuptools import setup, find_packages

setup(
    name="datainsight-lite",
    version="0.1.0",
    description="One-line EDA report generator for pandas DataFrames.",
    author="Swakshwar Ghosh",
    author_email="ghoshswakshwar@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas",
        "matplotlib",
        "seaborn",
        "jinja2"
    ],
    entry_points={
        "console_scripts": [
            "datainsight-lite=datainsight.cli:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
    python_requires=">=3.7"
)