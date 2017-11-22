# -*- coding: utf-8 -*-

from setuptools import setup
	
setup(
    name="ai",
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    author="Decalogue",
    version="0.0.1",
    author_email="1044908508@qq.com",
    description="Ai lib.",
    license="MIT",
    url="https://github.com/Decalogue/ai",
    packages=["ai"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Natural Language :: English",
		"Natural Language :: Chinese (Simplified)",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Other OS",
        "Programming Language :: Python :: 3.5"
    ]
)
