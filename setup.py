import setuptools

setuptools.setup(
    name="GeoVMS",
    version="0.0.1",
    author="Liang Ding",
    author_email="liangding86@gmail.com",
    description="Data-Driven Volcanogenic Massive Sulfide (VMS) Prospectivity Mapping",
    long_description="Data-Driven Volcanogenic Massive Sulfide (VMS) Prospectivity Mapping",
    long_description_content_type="text/markdown",
    url="https://github.com/Liang-Ding/geovms",
    project_urls={
        "Bug Tracker": "",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords=[
        "Bayesian deep learning",
        "Prospectivity mapping",
        "Uncertainty quantification",
        "Volcanogenic massive sulfide",
        "Bathurst Mining Camp",
        "Geophysics",
        "Machine Learning",
        "Deep Learning",
    ],
    # package_dir={"": "geovms"},
    python_requires='>=3.12.0',
    install_requires=[
        "torch", "numpy", "h5py", "timm", "matplotlib"
    ],
    packages=setuptools.find_packages(),
)