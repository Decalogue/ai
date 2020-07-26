# -*- coding: utf-8 -*-
""" ai.vis """
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
from pyecharts.charts import Bar
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot


def plot_histogram(name, items, values, mode='html', img_path='res.png'):
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


def summary(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    result, params_info = summary_string(model, input_size, batch_size, device, dtypes)
    print(result)
    return params_info


def summary_string(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor] * len(input_size)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(*x)

    # remove hooks
    for h in hooks:
        h.remove()

    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ())) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.)) # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "================================================================" + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params - trainable_params) + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    # return summary
    return summary_str, (total_params, trainable_params)