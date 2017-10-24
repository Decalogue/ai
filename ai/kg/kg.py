# coding: utf-8

# # Create KG with XMind

# ## 导入依赖项

# In[117]:

import os
import string
import xlrd
from py2neo import Graph, Node, Relationship, NodeSelector


# ## 读取 Excel 文件

# In[118]:

def read_excel(filepath):
    """Get excel source

    Args:
        filepath: The full path of excel file. excel文件完整路径。

    Returns:
        data: Data of excel. excel数据。
    """
    is_valid = False
    try:
        if os.path.isfile(filepath):
            filename = os.path.basename(filepath)
            if filename.split('.')[1] == 'xls':
                is_valid = True
        data = None
        if is_valid:
            data = xlrd.open_workbook(filepath, formatting_info=True)
    except Exception as xls_error:
        raise TypeError("Can't get data from excel!") from xls_error
    return data


# In[119]:

print(read_excel("./sample/kg.xls"))


# ## 定义知识图谱

# In[122]:

class KG():
    """Knowledge Graph.
    知识图谱。
    """
    def __init__(self, password="train", userid="userid", is_admin=True):
        self.is_admin = is_admin
        self.graph = Graph("http://localhost:7474/db/data", password=password)
        self.selector = NodeSelector(self.graph)
        self.xmind = {}

    def merge(self, filepath=None, custom_sheets=None):
        assert filepath is not None, "The merge filepath can not be None."
        data = read_excel(filepath)
        data_sheets = data.sheet_names()
        if custom_sheets:
            sheet_names = list(set(data_sheets).intersection(set(custom_sheets)))
        else:
            sheet_names = data_sheets
        # print(sheet_names)
        
        for sheet_name in sheet_names:
            # 1.Select specified table
            table = data.sheet_by_name(sheet_name)
            # table = data.sheet_by_index(0)
            if table:
                # 2.Select specified column
                col_format = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
                try:
                    nrows = table.nrows
                    ncols = table.ncols
                    str_upcase = [i for i in string.ascii_uppercase]
                    i_upcase = range(len(str_upcase))
                    ncols_dir = dict(zip(str_upcase, i_upcase))
                    col_index = [ncols_dir.get(i) for i in col_format]
                    
                    # 显示全部子主题
                    # for i in range(2, nrows):
                        # for j in range(0, ncols):
                            # print(table.cell(i, col_index[j]).value)

                    merge_cell_index = []
                    for (rlow,rhigh,clow,chigh) in table.merged_cells:
                        merge_cell_index.append((rlow,clow))
                    print("merge_cells:\n", table.merged_cells)
                    print("merge_cell_index:\n", merge_cell_index)
                            
                    # 第一行：中心主题
                    main_topic = table.cell(0, col_index[0]).value
                    print("main_topic:", main_topic)
                    # 第二行：层级数目
                    max_level = table.cell(1, col_index[ncols-1]).value
                    print("max_level:", max_level)

                    # step_1: xls to dict
                    temp = self.xmind
                    temp[main_topic] = {}
                    temp = temp[main_topic] 
                    # print("xmind:", self.xmind)
                    # print("temp:", temp)
                    
                    rlow = 2
                    rhigh = nrows
                    clow = 0
                    
                    def mindmap(rlow, rhigh, clow, temp):
                        # print(rlow, rhigh, clow, temp)
                        i = rlow
                        j = clow
                        reset = temp # 保存初始主题以便处理完子主题后重置
                        while i < rhigh:
                            # 无合并单元格（无子主题或只有单个子主题）
                            if (i, j) not in merge_cell_index:
                                for j in range(clow, ncols):
                                    key = table.cell(i, col_index[j]).value
                                    if not key:
                                        break
                                    temp[key] = {}
                                    temp = temp[key]
                                i += 1
                                j = clow
                                temp = reset
                            # 有合并单元格（有多个子主题）
                            else:
                                key = table.cell(i, col_index[j]).value
                                temp[key] = {}
                                temp = temp[key]
                                for (rl,rh,cl,ch) in table.merged_cells:
                                    if i==rl and j==cl:
                                        sub_rhigh = rh
                                mindmap(i, sub_rhigh, j+1, temp)
                                i = sub_rhigh
                                temp = reset

                    mindmap(rlow, rhigh, clow, temp)
                    # print("xmind:", self.xmind)
                                
                    # step_2: dict to graph
                    
                except Exception as error:
                    print('Error: %s' % error)
                    return None
            else:
                print('Error! Data of %s is empty!' % sheet_name)
                return None


# ## 分析 XMind 导出的 xls 文件格式
# ### 由输出格式可以看出对于合并的单元格内容可以用低位索引

# In[124]:

kg = KG(password="train")
kg.merge(filepath="./sample/kg.xls")
print(kg.xmind)


# In[ ]:



