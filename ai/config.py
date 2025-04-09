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
	return AttrDict(yaml.load(open(filepath, 'r'), Loader=yaml.FullLoader))


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
		except KeyError as k:
			return None

	def __setattr__(self, key, value):
		if isinstance(value, dict):
			value = Dict(value)
		self[key] = value

	def __delattr__(self, key):
		try:
			del self[key]
		except KeyError as k:
			return None

	def __call__(self, key):
		try:
			return self[key]
		except KeyError as k:
			return None


class Config(object):
	"""支持链式调用的 Config
	"""
	def __init__(self, filepath=None):
		if filepath:
			self.filepath = filepath
		else:
			cur_dir = os.path.split(os.path.realpath(__file__))[0]
			self.filepath = os.path.join(cur_dir, "self.conf")
		self.config = get_conf_config(self.filepath)
		self.d = Dict()
		for s in self.config.sections():
			value = Dict()
			for k in self.config.options(s):
				value[k] = self.config.get(s, k)
			self.d[s] = value

	def add(self, section):
		self.config.add_section(section)
		self.d[section] = Dict()
		with open(self.filepath, 'w', encoding="UTF-8") as f:
			self.config.write(f)

	def set(self, section, key, value):
		self.config.set(section, key, value)
		self.d[section][key] = value
		with open(self.filepath, 'w', encoding="UTF-8") as f:
			self.config.write(f)

	def get(self, section, key):
		return self.config.get(section, key, default=None)

	def remove_section(self, section):
		self.config.remove_section(section)
		del self.d[section]
		with open(self.filepath, 'w', encoding="UTF-8") as f:
			self.config.write(f)

	def remove_option(self, section, key):
		self.config.remove_option(section, key)
		del self.d[section][key]
		with open(self.filepath, 'w', encoding="UTF-8") as f:
			self.config.write(f)

	def save(self):
		for s in self.d:
			if s not in self.config.sections():
				self.add(s)
			for k in self.d[s]:
				try:
					v = self.get(s, k)
				except:
					v = None
				if self.d[s][k] != v:
					self.set(s, k, self.d[s][k])

	def __getattr__(self, name):
		if name not in self.__dict__:
			try:
				return self.d[name]
			except KeyError as k:
				self.d[name] = Dict()
				return self.d[name]
		return self.__dict__[name]
