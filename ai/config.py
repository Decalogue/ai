import os
import yaml
from configparser import ConfigParser


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_yaml_config(filepath):
    """使用 .yaml 文件配置
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return AttrDict(yaml.load(f, Loader=yaml.FullLoader))


def get_conf_config(filepath):
    """使用 .conf 文件配置
    """
    conf = ConfigParser()
    conf.read(filepath, encoding="UTF-8")
    return conf


class Dict(dict):
    """重写 dict, 支持 “.” 方式的属性调用
    """
    def __init__(self, *args, **kwargs):
        super(Dict, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        try:
            value = self[key]
            if isinstance(value, dict):
                value = Dict(value)
            return value
        except KeyError:
            return None

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = Dict(value)
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            return None

    def __call__(self, key):
        try:
            return self[key]
        except KeyError:
            return None


class Config(object):
    """支持链式调用的 Config
    """
    def __init__(self, filepath=None, auto_save=True):
        """
        Args:
            filepath: 配置文件路径，如果为 None 则使用默认路径
            auto_save: 是否自动保存，如果为 False 则需要手动调用 save()
        """
        if filepath:
            self.filepath = filepath
        else:
            cur_dir = os.path.split(os.path.realpath(__file__))[0]
            self.filepath = os.path.join(cur_dir, "self.conf")
        # 如果文件不存在，创建空配置
        if not os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'w', encoding='utf-8') as f:
                    f.write('')
            except (IOError, OSError) as e:
                raise IOError(f"无法创建配置文件 {self.filepath}: {e}")
        self.auto_save = auto_save
        self._dirty = False  # 标记配置是否已修改但未保存
        self.config = get_conf_config(self.filepath)
        self.d = Dict()
        for s in self.config.sections():
            value = Dict()
            for k in self.config.options(s):
                value[k] = self.config.get(s, k)
            self.d[s] = value

    def add(self, section):
        """添加配置节，支持链式调用"""
        self.config.add_section(section)
        self.d[section] = Dict()
        self._dirty = True
        if self.auto_save:
            self._write_to_file()
        return self

    def set(self, section, key, value):
        """设置配置项，支持链式调用"""
        # 如果 section 不存在，自动创建
        if section not in self.config.sections():
            self.config.add_section(section)
            self.d[section] = Dict()
        self.config.set(section, key, str(value))
        self.d[section][key] = value
        self._dirty = True
        if self.auto_save:
            self._write_to_file()
        return self

    def get(self, section, key):
        return self.config.get(section, key, fallback=None)

    def remove_section(self, section):
        """删除配置节，支持链式调用"""
        if section in self.config.sections():
            self.config.remove_section(section)
            if section in self.d:
                del self.d[section]
            self._dirty = True
            if self.auto_save:
                self._write_to_file()
        return self

    def remove_option(self, section, key):
        """删除配置项，支持链式调用"""
        if section in self.config.sections() and self.config.has_option(section, key):
            self.config.remove_option(section, key)
            if section in self.d and key in self.d[section]:
                del self.d[section][key]
            self._dirty = True
            if self.auto_save:
                self._write_to_file()
        return self

    def _write_to_file(self):
        """内部方法：将配置写入文件"""
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                self.config.write(f)
            self._dirty = False
        except (IOError, OSError) as e:
            raise IOError(f"无法写入配置文件 {self.filepath}: {e}")

    def save(self):
        """保存配置到文件，支持链式调用"""
        # 同步内存中的配置到 ConfigParser
        for s in self.d:
            if s not in self.config.sections():
                self.config.add_section(s)
            for k in self.d[s]:
                try:
                    v = self.get(s, k)
                except Exception:
                    v = None
                # 将值转换为字符串（ConfigParser 只存储字符串）
                current_value = str(self.d[s][k]) if self.d[s][k] is not None else ''
                if v != current_value:
                    self.config.set(s, k, current_value)
        # 写入文件
        self._write_to_file()
        return self

    def __getattr__(self, name):
        if name not in self.__dict__:
            try:
                return self.d[name]
            except KeyError:
                self.d[name] = Dict()
                return self.d[name]
        return self.__dict__[name]
