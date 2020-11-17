import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="transformer_srl",  # Replace with your own username
    version="2.4.3",
    author="Riccardo Orlando",
    author_email="orlandoricc@gmail.com",
    description="SRL Transformer model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Riccorl/transformer-srl",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "allennlp==1.2.0",
        "allennlp_models==1.2.0",
        "spacy>=2.3,<2.4"
    ],
    python_requires=">=3.6",
)
