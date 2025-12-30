import os
import bz2
import time
import requests


def ensure_dir(dir_path):
    '''确保单个目录存在
    '''
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def ensure_dirs(dir_list):
    '''确保多个目录存在
    '''
    for dir_path in dir_list:
        ensure_dir(dir_path)


def unpack_bz2(src_path, dst_path=None):
    '''解压 bz2 文件
    '''
    data = bz2.BZ2File(src_path).read()
    if dst_path is None:
        dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)


def get_current_time(format_string="%Y-%m-%d-%H-%M-%S"):
    return time.strftime(format_string, time.localtime())


def url2file(url, savepath=None):
    assert savepath is not None, 'The savepath should not be None'
    with open(savepath, 'wb') as f:
        f.write(requests.get(url).content)
