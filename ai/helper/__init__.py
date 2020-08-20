# -*- coding: utf-8 -*-
""" ai.helper """

import argparse
import os
import re
import time
import datetime
import hashlib
import inspect
import json
import logging
import random
import requests
import socket
import uuid
import numpy as np
from collections import Counter, OrderedDict
from functools import wraps
from logging import Logger
from logging.handlers import TimedRotatingFileHandler
from time import sleep
from sklearn.feature_extraction.text import CountVectorizer
from ai.conf import Config

cur_dir = os.path.split(os.path.realpath(__file__))[0]
conf = Config(os.path.join(cur_dir, "self.conf"))
not_zh = re.compile(r"[^\u4e00-\u9fa5]")


class Error(Exception):
    """ Base class for exceptions in this module.
    """
    def __init__(self, value):
        self.value = value
        
    def __str__(self):
        return repr(self.value)


class StringPatternError(Error):
    """ Exception raised for errors in the pattern of string args.
    """
    pass


class JsonEncoder(json.JSONEncoder):
    """ JsonEncoder
        解决 json.dumps 不能序列化 datetime 类型的问题：使用 Python 自带的 json.dumps 方法
        转换数据为 json 格式的时候，如果格式化的数据中有 datetime 类型的数据时将会报错。
        TypeError: datetime.datetime(2014, 03, 20, 12, 10, 44) is not JSON serializable
    Usage:
        json.dumps(data, cls=JsonEncoder)
    """
    def default(self, obj): 
        if isinstance(obj, datetime.datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')  
        elif isinstance(obj, datetime.date):
            return obj.strftime('%Y-%m-%d')  
        else:
            return json.JSONEncoder.default(self, obj)


def ensure_dir(path):
    """ 确保目录存在 """
    if not os.path.exists(path):
        os.makedirs(path)


def init_logger(log_name, log_dir):
    """ 日志模块
        1. 同时将日志打印到屏幕和文件中
        2. 默认值保留近30天日志文件
    """
    ensure_dir(log_dir)
    if log_name not in Logger.manager.loggerDict:
        logger = logging.getLogger(log_name)
        logger.setLevel(logging.DEBUG)
        handler = TimedRotatingFileHandler(
            filename=os.path.join(log_dir, "%s.log" % log_name),
            when="D",
            backupCount=30,
        )
        datefmt = "%Y-%m-%d %H:%M:%S"
        format_str = "[%(asctime)s]: %(name)s %(filename)s[line:%(lineno)s] %(levelname)s  %(message)s"
        formatter = logging.Formatter(format_str, datefmt)
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)

        handler = TimedRotatingFileHandler(
            filename=os.path.join(log_dir, "ERROR.log"),
            when="D",
            backupCount=30,
        )
        datefmt = "%Y-%m-%d %H:%M:%S"
        format_str = "[%(asctime)s]: %(name)s %(filename)s[line:%(lineno)s] %(levelname)s  %(message)s"
        formatter = logging.Formatter(format_str, datefmt)
        handler.setFormatter(formatter)
        handler.setLevel(logging.ERROR)
        logger.addHandler(handler)
    logger = logging.getLogger(log_name)
    return logger


def translate_baidu(content, fromLang='zh', toLang='en'):
    """ 百度翻译 API """
    salt = str(random.randint(32768, 65536))
    sign = conf.translate.appid + content + salt + conf.translate.secretkey
    sign = hashlib.md5(sign.encode("utf-8")).hexdigest()
    try:
        paramas = {
            'appid': conf.translate.appid,
            'q': content,
            'from': fromLang,
            'to': toLang,
            'salt': salt,
            'sign': sign
        }
        response = requests.get(conf.translate.apiurl, paramas)
        data = response.json()
        res = [d["dst"] for d in data["trans_result"]]
        return res
    except Exception as e:
        print(content, e)
        return content.split('\n')


def back_translate_zh(paras, limit=1):
    """ 中文回译 """
    try:
        sleep(limit)
        tmp = translate_baidu('\n'.join(paras), fromLang='zh', toLang='en')
        sleep(limit)
        res = translate_baidu('\n'.join(tmp), fromLang='en', toLang='zh')
        return res
    except:
        return paras


def back_translate_en(paras, limit=1):
    """ 英文回译 """
    try:
        sleep(limit)
        tmp = translate_baidu('\n'.join(paras), fromLang='en', toLang='zh')
        sleep(limit)
        res = translate_baidu('\n'.join(tmp), fromLang='zh', toLang='en')
        return res
    except:
        return paras


def order_dict(d, mode='key'):
    """ 对字典按照 key / value 排序
    """
    if mode == 'key':
        res = sorted(d.items(), key=lambda t: t[0])
    elif mode == 'value':
        res = sorted(d.items(), key=lambda t: t[1])
    res = OrderedDict(res)
    return res


def get_mac_address():
    """ Get mac address.
    """
    mac = uuid.UUID(int=uuid.getnode()).hex[-12:]
    return ":".join([mac[e:e+2] for e in range(0, 11, 2)])


def get_hostname():
    """ Get hostname.
    """
    return socket.getfqdn(socket.gethostname())


def get_ip_address(hostname):
    """ Get host ip address.
    """
    return socket.gethostbyname(hostname)


def get_host_ip():
    """ Get host ip address.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def get_current_function_name():
    """ Get current function name.
    """
    return inspect.stack()[1][3]


class Walk():
    """ Walk directory to batch processing.
        遍历目录进行批处理。

        Subclasses may override the 'handle_file' method to provide custom file processing mode.
        子类可以重写 'handle_file' 方法来实现自定义的文件处理方式。

    Public attributes:
        - filelist: All filenames with full path in directory.
        - fnamelist: All filenames in directory.
        - dirlist: All dirnames with full path in directory.
        - dnamelist: All dirnames in directory.
    """
    def __init__(self):
        self.filenum = 0
        self.filelist = []
        self.fnamelist = []
        self.dirlist = []
        self.dnamelist = []
        self.dirstr = '+'
        self.filestr = '-'

    def dir_print(self, level, path):
        """Walk and print all dirs and files in a directory.
        遍历目录打印所有子目录及文件名。

        Args:
            level: Level of current directory. 目录的深度。
            path: Full path of current directory. 目录的完整路径。

        Returns:
            filenum. 遍历目录下的所有文件总数。
        """
        files = os.listdir(path)
        # 先添加目录级别
        self.dirlist.append(str(level))
        for file in files:
            if os.path.isdir(path+'/'+file):
                # 排除隐藏文件夹
                if file[0] == '.':
                    pass
                else:
                    self.dirlist.append(file)
            if os.path.isfile(path + '/' + file):
                # 添加文件
                self.filelist.append(file)
        # 文件夹列表第一个级别不打印
        for dirname in self.dirlist[1:]:
            print('-' * (int(self.dirlist[0])), dirname)
            # 递归打印目录下的所有文件夹和文件，目录级别+1
            self.dir_print((int(self.dirlist[0])+1), path+'/'+dirname)
        for filename in self.filelist:
            print('-' * (int(self.dirlist[0])), filename)
            self.filenum = self.filenum + 1
        return self.filenum

    def str_file(self, level):
        """Get str that represents the level of the file.
        文件层级信息的打印字符表示。
        """
        return '  ' * level + self.filestr

    def str_dir(self, level):
        """Get str that represents the level of the directory.
        目录层级信息的打印字符表示。
        """
        return '  ' * level + self.dirstr

    def dir_process(self, level, path, style="fnamelist"):
        """Walk and process all dirs and files in a directory.
        遍历目录批处理所有文件。

        Args:
            level: Level of current directory. 目录的深度。
            path: Full path of current directory. 目录的完整路径。
            style: Specifies the content to return. 指定要返回的内容。
                The style can be 'filelist', 'fnamelist', 'dirlist' or 'dnamelist'.
                Defaults to "fnamelist".

        Returns:
            filenum. 遍历目录下的所有文件总数。
        """
        if os.path.exists(path):
            files = os.listdir(path)
            for file in files:
                # Exclude hidden folders and files
                if file[0] == '.':
                    continue
                else:
                    subpath = os.path.join(path, file)
                if os.path.isfile(subpath):
                    # Get filelist and fnamelist
                    fname = os.path.basename(subpath)
                    self.filelist.append(subpath)
                    self.fnamelist.append(fname)
                    print(self.str_file(level) + fname)
					# Handle file with specified method by pattern
                    self.handle_file(subpath)
                else:
                    leveli = level + 1
                    # Get dirlist and dnamelist
                    dname = os.path.basename(subpath)
                    self.dirlist.append(subpath)
                    self.dnamelist.append(dname)
                    print(self.str_dir(level) + dname)
                    self.dir_process(leveli, subpath, style)
        # Return the specified list by style
        return self.__dict__[style]

    def handle_file(self, filepath, pattern=None):
        """Handle file with specified method by pattern.
        根据pattern指定的模式处理文件。

        Args:
            filepath: Full path of file. 文件的完整路径。
            pattern: Specifies the pattern to handle file. 指定处理该文件的模式。
                Defaults to None.

        Returns:
            You can customize when you override this method. 当你重写该方法时可以自定义。
        """
        pass


def time_me(info="used", format_string="ms"):
    """ Performance analysis - time
        Decorator of time performance analysis.
        性能分析——计时统计

        系统时间(wall clock time, elapsed time)是指一段程序从运行到终止，系统时钟走过的时间。
        一般系统时间都是要大于 CPU 时间的。通常可以由系统提供，在 C++/Windows 中，可以由 <time.h> 提供。
        注意得到的时间精度是和系统有关系的。
        1.time.clock() 以浮点数计算的秒数返回当前的 CPU 时间。用来衡量不同程序的耗时，比 time.time() 更有用。
        time.clock() 在不同的系统上含义不同。在 UNIX 系统上，它返回的是'进程时间'，它是用秒表示的浮点数（时间戳）。
        而在 WINDOWS 中，第一次调用，返回的是进程运行的实际时间。而第二次之后的调用是自第一次调用以后到现在的运行时间。
        （实际上是以 WIN32 上 QueryPerformanceCounter() 为基础，它比毫秒表示更为精确）
        2.time.perf_counter() 能够提供给定平台上精度最高的计时器。计算的仍然是系统时间，
        这会受到许多不同因素的影响，例如机器当前负载。
        3.time.process_time() 提供进程时间。

    Args:
        info: Customize print info. 自定义提示信息。
        format_string: Specifies the timing unit. 指定计时单位，例如's': 秒，'ms': 毫秒。
            Defaults to 's'.
    """
    def _time_me(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            start = time.clock()
            # start = time.perf_counter()
            # start = time.process_time()
            result = func(*args, **kwargs)
            end = time.clock()
            if format_string == "s":
                print("%s %s %s"%(func.__name__, info, end - start), "s")
            elif format_string == "ms":
                print("%s %s %s" % (func.__name__, info, 1000*(end - start)), "ms")
            return result
        return _wrapper
    return _time_me


def get_timestamp(s=None, style='%Y-%m-%d %H:%M:%S', pattern='s'):
    """ Get timestamp. 获取指定日期表示方式的时间戳或者当前时间戳。
    
    Args:
        style: Specifies the format of time. 指定日期表示方式。
            Defaults to '%Y-%m-%d %H:%M:%S'.
        pattern: Specifies the timestamp unit. 指定时间戳单位，'s': 秒，'ms': 毫秒。
            Defaults to 's'.
    """
    w = {'s': 1, 'ms': 1000}
    if isinstance(s, str):
        try:
            return int(time.mktime(time.strptime(s, style)) * w[pattern])
        except:
            raise StringPatternError(" The style must be '%Y-%m-%d %H:%M:%S'\
                or coincide with your custom format.")
    else:
        return int(time.time() * w[pattern])


def get_current_time(format_string="%Y-%m-%d-%H-%M-%S", info=None):
    """ Get current time with specific format_string.
        获取指定日期表示方式的当前时间。

        For Python3
        On Windows, time.strftime() and Unicode characters will raise UnicodeEncodeError.
        http://bugs.python.org/issue8304

    Args:
        format_string: Specifies the format of time. 指定日期表示方式。
            Defaults to '%Y-%m-%d-%H-%M-%S'.
    """
    assert isinstance(format_string, str), "The format_string must be a string."
    try:
        current_time = time.strftime(format_string, time.localtime())
    except UnicodeEncodeError:
        result = time.strftime(format_string.encode('unicode-escape').decode(), time.localtime())
        current_time = result.encode().decode('unicode-escape')
    return current_time


def get_age(format_string="%s年%s个月%s天", birthday="2018-1-1"):
    """ Get age with specific format_string.
        获取指定日期表示方式的年龄。

    Args:
        format_string: Specifies the format of time. 指定日期表示方式。
            Defaults to '{y}年{m}个月{d}天'.
    """
    assert isinstance(format_string, str), "The format_string must be a string."
    assert isinstance(birthday, str), "The birthday must be a string."
    
    # 方案1：根据每月具体天数计算具体时长
    mdays = [31, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30] # 从12（0）月到11月
    ct = get_current_time(format_string="%Y-%m-%d")
    st = [int(i) for i in birthday.split('-')]
    et = [int(i) for i in ct.split('-')]
    if st[1] < et[1]:
        year = et[0] - st[0]
        if st[2] < et[2]:
            month = et[1] - st[1]
            day = et[2] - st[2]
        else:
            month = et[1] - st[1] - 1
            day = et[2] + mdays[(et[1] - 1) % 12] - st[2]
    else:
        year = et[0] - st[0] - 1
        if st[2] < et[2]:
            month = et[1] + 12 - st[1]
            day = et[2] - st[2]
        else:
            month = et[1] + 12 - st[1] - 1
            day = et[2] + mdays[(et[1] - 1) % 12] - st[2]

    # 方案2：根据每月平均30天计算具体时长
    # start_time= datetime.datetime.strptime(birthday, "%Y-%m-%d")
    # end_time= datetime.datetime.strptime(ct, "%Y-%m-%d")

    # seconds = (end_time - start_time).seconds  
    # hours = (end_time - start_time).hours
    # days = (end_time - start_time).days
    # year = int(days / 365)
    # month = int(days % 365 / 30)
    # day = int(days % 365 % 30)

    age = format_string % (str(year), str(month), str(day))
    return age


def readlines(filepath, start=0, n=None):
    """ 按行读取从 start 位置开始的指定 n 行，当 n=None 时读取全部
    """
    with open(filepath, 'r', encoding='UTF-8') as f:
	    if start > 0:
	        for i in range(start):
	            f.readline()
	    cnt = 0
	    while True:
	        content = f.readline()
	        if content == '' or (cnt >= n and n != None):
	            break
	        cnt += 1
	        yield content


class AverageMeter(object):
    """ Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def remove_punc(text):
    """ 中文去标点 """
    text = not_zh.sub("", text)
    return text


def search(pattern, sequence):
    """从 sequence 中寻找子串 pattern
    如果找到，返回第一个下标；否则返回 -1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


def search_all(pattern, sequence):
    """从 sequence 中寻找子串 pattern
    如果找到，返回所有起始坐标；否则返回 []。
    """
    res = []
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            res.append([i, i+n-1])
    return res


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def num_common_words(query, title):
    """ 获取公共词的数量
    """
    query = set(query)
    title = set(title)
    return len(query & title)


def jaccard(query, title):
    """ 基于离散词的句子 jaccard 距离
    """
    query = set(query)
    title = set(title)
    intersection_len = len(query & title)
    union_len = len(query | title)
    return intersection_len / union_len


def word_based_similarity(query, title, mode='cosine'):
    """ 基于离散词的句子相似度
    """
    try:
        counter = CountVectorizer(analyzer='word', token_pattern=u"(?u)\\b\\w+\\b")
        counter.fit([query, title])
        result = counter.transform([query, title]).toarray()
        vec1, vec2 = result[0], result[1]
        if mode == 'euclid':
            return np.linalg.norm(vec1 - vec2)
        elif mode == 'manhattan':
            return np.sum(np.abs(vec1 - vec2))
        else:
            return np.sum(vec1 * vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    except Exception as e:
        print('Meet a outlier sample!')
        return 0