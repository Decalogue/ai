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
    os.system("rm -rf dist/ build/ *.egg-info")  # 先清理旧文件
    os.system("python setup.py sdist bdist_wheel")  # 构建两种格式
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
    version=version,
    author="Decalogue",
    author_email="1044908508@qq.com",
    description="Ai lib.",
    long_description=open('README.rst').read() if os.path.exists('README.rst') else "Ai lib.",
    long_description_content_type="text/x-rst",
    license="MIT",
    url="https://github.com/Decalogue/ai",
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[],
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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12"
    ]
)
