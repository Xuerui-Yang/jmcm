import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="jmcm",
    version="0.1.2",
    author="Xuerui Yang",
    author_email="xuerui-yang@outlook.com",
    description="A statistical package for fit the joint mean-covariance models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=" https://github.com/Xuerui-Yang/jmcm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)