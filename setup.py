# -*- coding: utf-8 -*-
import os
import sys
import ai
from setuptools import setup, find_packages


version = ai.__version__

if sys.argv[-1] == 'tag':
    os.system("git tag -a %s -m 'version %s'" % (version, version))
    os.system("git push origin --tags")
    os.system("git push --all origin")
    sys.exit()
elif sys.argv[-1] == 'publish':
    # os.system("python setup.py sdist upload")
    # os.system("python setup.py bdist_wheel upload")
    os.system("python setup.py sdist")
    os.system("twine upload dist/*")
    sys.exit()
elif sys.argv[-1] == 'test':
    test_requirements = [
        'pytest',
        'flake8',
        'coverage'
    ]
    try:
        modules = map(__import__, test_requirements)
    except ImportError as e:
        err_msg = e.message.replace("No module named ", "")
        msg = "%s is not installed. Install your test requirments." % err_msg
        raise ImportError(msg)
    os.system('py.test')
    sys.exit()

setup(
    name="ai",
    author="Decalogue",
    version=version,
    author_email="1044908508@qq.com",
    description="Ai lib.",
    license="MIT",
    url="https://github.com/Decalogue/ai",
    packages=find_packages(),
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
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9"
    ]
)
