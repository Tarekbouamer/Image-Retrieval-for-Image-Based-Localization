from setuptools import setup, find_packages
from os import path, listdir

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='VisLoc',
    author="Tarek BOUAMER",
    author_email="tarekbouamer199147@gmail.com",
    description="Long Term Visual Localization",
    version="0.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url="https://github.com/Tarekbouamer",

    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],

    # Versioning
    #use_scm_version={"root": ".", "relative_to": __file__, "write_to": "magicpoint/_version.py"},

    # Requirements
    setup_requires=["setuptools_scm"],
    python_requires=">=3, <4",

    # Package description
    packages=[
        "cirtorch.algos",
        "cirtorch.backbones",
        "cirtorch.configuration",
        "cirtorch.datasets",
        "cirtorch.datasets.generic",
        "cirtorch.examples",
        "cirtorch.layers",
        "cirtorch.models",
        "cirtorch.modules",
        "cirtorch.modules.heads",
        "cirtorch.utils",
        "cirtorch.utils.parallel",
        "cirtorch",
    ],
    include_package_data=True,
)