from setuptools import setup, find_packages
from os import path, listdir
from pip.req import parse_requirements
from pip.download import PipSession



here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

req_pkgs = parse_requirements('requirements.txt', session=PipSession())
reqs = [str(pkg.req) for pkg in req_pkgs]

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

    # Requirements
    setup_requires=["setuptools_scm"],
    install_requires=req_pkgs,
    python_requires=">=3, <4",

    # Package description
    packages=[
        "cirtorch",

        "cirtorch.algos",
        "cirtorch.backbones",
        "cirtorch.configuration",
        
        "cirtorch.datasets",
        "cirtorch.datasets.localFeatures",
        "cirtorch.datasets.globalFeatures",
        "cirtorch.datasets.generic",
        "cirtorch.datasets.augmentation",
        
        "cirtorch.enhance",
        "cirtorch.enhance.color",
        
        "cirtorch.filters",
        
        "cirtorch.geometry",
        "cirtorch.geometry.camera",
        "cirtorch.geometry.epipolar",
        "cirtorch.geometry.subpix",
        "cirtorch.geometry.transform",
        "cirtorch.geometry.warp",

        "cirtorch.models",
        
        "cirtorch.modules",
        "cirtorch.modules.heads",
        
        "cirtorch.utils",
        "cirtorch.utils.parallel",
    ],
    include_package_data=True,
)
