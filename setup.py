from setuptools import setup, find_packages

setup(
    name="datainsight-lite",
    version="0.1.4a1",
    description="One-line EDA report generator for pandas DataFrames.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Swakshwar Ghosh",
    author_email="ghoshswakshwar@gmail.com",
    packages=find_packages(where="datainsight_lite", include=["datainsight*", "utils*"]),
    include_package_data=True,
    package_data={"datainsight_lite.datainsight": ["templates/*.html"]},
    package_dir={"": "datainsight_lite"},
    install_requires=[
        "pandas",
        "matplotlib",
        "seaborn",
        "jinja2"
    ],
    url="https://github.com/swaksh-war/datainsight-lite",
    project_urls={
        "Bug Tracker": "https://github.com/swaksh-war/datainsight-lite/issues",
        "Documentation": "https://github.com/swaksh-war/datainsight-lite#readme",
        "Source Code": "https://github.com/swaksh-war/datainsight-lite",
    },
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