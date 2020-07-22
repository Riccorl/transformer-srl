import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="srl-bert-verbatlas",  # Replace with your own username
    version="0.9.9",
    author="Riccardo Orlando",
    author_email="orlandoricc@gmail.com",
    description="SRL Bert model trained with VerbAtlas inventory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Riccorl/srl-bert-verbatlas",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "allennlp==0.9"
    ],
    python_requires=">=3.6",
)
