# -*- coding: utf-8 -*-
""" ai.helper.vis """

from pyecharts.charts import Bar
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot


def plot_cnt(name, items, values, mode='html', img_path='res.png'):
    """ 绘制直方图
    Note:
        从 pyecharts v1 版本开始，add_yaxis 需要传入两个参数，第一个参数为 str 类型
    	关于 snapshot: snapshot-selenium 是 pyecharts + selenium 渲染图片的扩展，使用 selenium 需要配置 browser driver。
    	可以参考 selenium-python 相关介绍，推荐使用 Chrome 浏览器，可以开启 headless 模式。目前支持 Chrome, Safari。
    	Download: https://selenium-python.readthedocs.io/installation.html#drivers
    	Make sure it’s in your PATH, e. g., place it in /usr/bin or /usr/local/bin.
    """
    bar = Bar()
    bar.add_xaxis(items)
    bar.add_yaxis(name, values)
    if mode == 'html':
        return bar.render()
    elif mode == 'notebook':
        return bar.render_notebook()
    elif mode == 'img':
        make_snapshot(snapshot, bar.render(), img_path)
    else:
        raise ValueError("mode must in ['html', 'notebook', 'img']")