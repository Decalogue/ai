"""
Config 模块的单元测试和使用示例

运行测试：
    pytest tests/test_config.py -v
    python -m pytest tests/test_config.py -v
"""
import os
import tempfile
import pytest
from ai.config import Config, Dict, AttrDict, get_yaml_config, get_conf_config


class TestDict:
    """测试 Dict 类的属性访问功能"""
    
    def test_dict_attribute_access(self):
        """测试通过属性访问字典值"""
        d = Dict({'key1': 'value1', 'key2': {'nested': 'value'}})
        assert d.key1 == 'value1'
        assert d.key2.nested == 'value'
    
    def test_dict_nonexistent_key(self):
        """测试访问不存在的键返回 None"""
        d = Dict()
        assert d.nonexistent is None
    
    def test_dict_set_attribute(self):
        """测试通过属性设置值"""
        d = Dict()
        d.new_key = 'new_value'
        assert d['new_key'] == 'new_value'
        assert d.new_key == 'new_value'
    
    def test_dict_callable(self):
        """测试 Dict 可调用"""
        d = Dict({'key': 'value'})
        assert d('key') == 'value'
        assert d('nonexistent') is None


class TestConfig:
    """测试 Config 类的核心功能"""
    
    @pytest.fixture
    def temp_config_file(self):
        """创建临时配置文件"""
        fd, path = tempfile.mkstemp(suffix='.conf')
        os.close(fd)
        yield path
        # 清理
        if os.path.exists(path):
            os.remove(path)
    
    def test_config_initialization(self, temp_config_file):
        """测试配置初始化"""
        config = Config(temp_config_file)
        assert config.filepath == temp_config_file
        assert config.auto_save is True
    
    def test_config_auto_save_default(self, temp_config_file):
        """测试默认自动保存"""
        config = Config(temp_config_file)
        assert config.auto_save is True
    
    def test_config_manual_save(self, temp_config_file):
        """测试手动保存模式"""
        config = Config(temp_config_file, auto_save=False)
        assert config.auto_save is False
    
    def test_add_section(self, temp_config_file):
        """测试添加配置节"""
        config = Config(temp_config_file)
        config.add('database')
        assert 'database' in config.config.sections()
        assert 'database' in config.d
    
    def test_set_option(self, temp_config_file):
        """测试设置配置项"""
        config = Config(temp_config_file)
        config.set('database', 'host', 'localhost')
        assert config.get('database', 'host') == 'localhost'
        assert config.d.database.host == 'localhost'
    
    def test_set_auto_create_section(self, temp_config_file):
        """测试设置时自动创建 section"""
        config = Config(temp_config_file)
        # section 不存在时应该自动创建
        config.set('new_section', 'key', 'value')
        assert 'new_section' in config.config.sections()
        assert config.get('new_section', 'key') == 'value'
    
    def test_get_option(self, temp_config_file):
        """测试获取配置项"""
        config = Config(temp_config_file)
        config.set('test', 'key', 'value')
        assert config.get('test', 'key') == 'value'
        assert config.get('test', 'nonexistent') is None
    
    def test_remove_section(self, temp_config_file):
        """测试删除配置节"""
        config = Config(temp_config_file)
        config.add('test_section').set('test_section', 'key', 'value')
        config.remove_section('test_section')
        assert 'test_section' not in config.config.sections()
        assert 'test_section' not in config.d
    
    def test_remove_option(self, temp_config_file):
        """测试删除配置项"""
        config = Config(temp_config_file)
        config.set('test', 'key1', 'value1').set('test', 'key2', 'value2')
        config.remove_option('test', 'key1')
        assert config.get('test', 'key1') is None
        assert config.get('test', 'key2') == 'value2'


class TestConfigChaining:
    """测试链式调用功能"""
    
    @pytest.fixture
    def temp_config_file(self):
        """创建临时配置文件"""
        fd, path = tempfile.mkstemp(suffix='.conf')
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)
    
    def test_chaining_add_set(self, temp_config_file):
        """测试链式调用：add -> set"""
        config = Config(temp_config_file)
        result = config.add('database').set('database', 'host', 'localhost')
        assert result is config  # 返回 self
        assert config.get('database', 'host') == 'localhost'
    
    def test_chaining_multiple_sets(self, temp_config_file):
        """测试链式调用：多个 set"""
        config = Config(temp_config_file)
        config.set('db', 'host', 'localhost').set('db', 'port', '3306').set('db', 'user', 'admin')
        assert config.get('db', 'host') == 'localhost'
        assert config.get('db', 'port') == '3306'
        assert config.get('db', 'user') == 'admin'
    
    def test_chaining_complex_operations(self, temp_config_file):
        """测试复杂链式操作"""
        config = Config(temp_config_file)
        config.remove_section('old').add('new').set('new', 'key1', 'val1').set('new', 'key2', 'val2')
        assert 'old' not in config.config.sections()
        assert config.get('new', 'key1') == 'val1'
        assert config.get('new', 'key2') == 'val2'
    
    def test_chaining_with_save(self, temp_config_file):
        """测试链式调用包含 save"""
        config = Config(temp_config_file, auto_save=False)
        result = config.add('test').set('test', 'key', 'value').save()
        assert result is config
        # 验证文件已保存
        config2 = Config(temp_config_file)
        assert config2.get('test', 'key') == 'value'


class TestConfigAutoSave:
    """测试自动保存和延迟保存功能"""
    
    @pytest.fixture
    def temp_config_file(self):
        """创建临时配置文件"""
        fd, path = tempfile.mkstemp(suffix='.conf')
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)
    
    def test_auto_save_enabled(self, temp_config_file):
        """测试自动保存模式（立即写入）"""
        config1 = Config(temp_config_file, auto_save=True)
        config1.set('test', 'key', 'value')
        
        # 立即读取应该能看到值
        config2 = Config(temp_config_file)
        assert config2.get('test', 'key') == 'value'
    
    def test_auto_save_disabled(self, temp_config_file):
        """测试延迟保存模式（批量写入）"""
        config1 = Config(temp_config_file, auto_save=False)
        config1.set('test', 'key1', 'value1').set('test', 'key2', 'value2')
        
        # 未保存前，新实例看不到值
        config2 = Config(temp_config_file)
        assert config2.get('test', 'key1') is None
        
        # 保存后可见
        config1.save()
        config3 = Config(temp_config_file)
        assert config3.get('test', 'key1') == 'value1'
        assert config3.get('test', 'key2') == 'value2'
    
    def test_dirty_flag(self, temp_config_file):
        """测试 _dirty 标志"""
        config = Config(temp_config_file, auto_save=False)
        assert config._dirty is False
        config.set('test', 'key', 'value')
        assert config._dirty is True
        config.save()
        assert config._dirty is False


class TestConfigAttributeAccess:
    """测试属性访问功能"""
    
    @pytest.fixture
    def temp_config_file(self):
        """创建临时配置文件"""
        fd, path = tempfile.mkstemp(suffix='.conf')
        os.close(fd)
        yield path
        if os.path.exists(path):
            os.remove(path)
    
    def test_attribute_access(self, temp_config_file):
        """测试通过属性访问配置"""
        config = Config(temp_config_file)
        config.set('database', 'host', 'localhost')
        assert config.database.host == 'localhost'
    
    def test_attribute_auto_create(self, temp_config_file):
        """测试访问不存在的 section 自动创建"""
        config = Config(temp_config_file)
        # 访问不存在的 section 会自动创建
        section = config.new_section
        assert isinstance(section, Dict)
        assert 'new_section' in config.d
    
    def test_attribute_set(self, temp_config_file):
        """测试通过属性设置值"""
        config = Config(temp_config_file)
        config.database.host = 'localhost'
        config.database.port = '3306'
        # 需要手动保存
        config.save()
        assert config.get('database', 'host') == 'localhost'
        assert config.get('database', 'port') == '3306'


class TestConfigErrorHandling:
    """测试错误处理"""
    
    def test_invalid_file_path(self):
        """测试无效文件路径"""
        invalid_path = '/nonexistent/directory/config.conf'
        with pytest.raises(IOError):
            Config(invalid_path)
    
    def test_remove_nonexistent_section(self):
        """测试删除不存在的 section（应该安全）"""
        fd, path = tempfile.mkstemp(suffix='.conf')
        os.close(fd)
        try:
            config = Config(path)
            # 删除不存在的 section 不应该报错
            config.remove_section('nonexistent')
            assert True  # 如果到这里说明没有报错
        finally:
            if os.path.exists(path):
                os.remove(path)
    
    def test_remove_nonexistent_option(self):
        """测试删除不存在的 option（应该安全）"""
        fd, path = tempfile.mkstemp(suffix='.conf')
        os.close(fd)
        try:
            config = Config(path)
            config.add('test')
            # 删除不存在的 option 不应该报错
            config.remove_option('test', 'nonexistent')
            assert True  # 如果到这里说明没有报错
        finally:
            if os.path.exists(path):
                os.remove(path)


# ==================== 使用示例 ====================

def example_basic_usage():
    """基本使用示例"""
    print("\n" + "="*60)
    print("基本使用示例")
    print("="*60)
    
    # 创建临时配置文件
    fd, path = tempfile.mkstemp(suffix='.conf')
    os.close(fd)
    
    try:
        # 创建配置对象
        config = Config(path)
        
        # 方式1：使用链式调用
        config.add('database').set('database', 'host', 'localhost').set('database', 'port', '3306')
        
        # 方式2：使用属性访问
        config.app.name = 'MyApp'
        config.app.version = '1.0.0'
        config.save()  # 属性设置需要手动保存
        
        # 读取配置
        print(f"数据库主机: {config.get('database', 'host')}")
        print(f"数据库端口: {config.get('database', 'port')}")
        print(f"应用名称: {config.app.name}")
        print(f"应用版本: {config.app.version}")
        
    finally:
        if os.path.exists(path):
            os.remove(path)


def example_chaining():
    """链式调用示例"""
    print("\n" + "="*60)
    print("链式调用示例")
    print("="*60)
    
    fd, path = tempfile.mkstemp(suffix='.conf')
    os.close(fd)
    
    try:
        config = Config(path)
        
        # 链式调用：一次性配置多个选项
        config.add('database').set('database', 'host', 'localhost').set('database', 'port', '3306').set('database', 'user', 'admin')
        
        # 复杂链式操作
        config.remove_section('old').add('new').set('new', 'key1', 'val1').set('new', 'key2', 'val2').save()
        
        print("链式调用完成！")
        
    finally:
        if os.path.exists(path):
            os.remove(path)


def example_auto_save():
    """自动保存 vs 延迟保存示例"""
    print("\n" + "="*60)
    print("自动保存 vs 延迟保存示例")
    print("="*60)
    
    fd, path = tempfile.mkstemp(suffix='.conf')
    os.close(fd)
    
    try:
        # 自动保存模式（默认）
        print("\n1. 自动保存模式（每次修改立即写入）:")
        config1 = Config(path, auto_save=True)
        config1.set('auto', 'key1', 'value1')  # 立即保存
        config1.set('auto', 'key2', 'value2')    # 立即保存
        
        # 延迟保存模式（批量写入，性能更好）
        print("\n2. 延迟保存模式（批量写入）:")
        config2 = Config(path, auto_save=False)
        config2.set('manual', 'key1', 'value1')  # 不保存
        config2.set('manual', 'key2', 'value2')  # 不保存
        config2.set('manual', 'key3', 'value3')  # 不保存
        config2.save()  # 一次性保存所有修改
        
        print("延迟保存完成！")
        
    finally:
        if os.path.exists(path):
            os.remove(path)


def example_attribute_access():
    """属性访问示例"""
    print("\n" + "="*60)
    print("属性访问示例")
    print("="*60)
    
    fd, path = tempfile.mkstemp(suffix='.conf')
    os.close(fd)
    
    try:
        config = Config(path)
        
        # 通过属性访问和设置
        config.database.host = 'localhost'
        config.database.port = '3306'
        config.app.name = 'MyApp'
        config.app.version = '1.0.0'
        config.save()
        
        # 读取
        print(f"数据库主机: {config.database.host}")
        print(f"数据库端口: {config.database.port}")
        print(f"应用名称: {config.app.name}")
        print(f"应用版本: {config.app.version}")
        
    finally:
        if os.path.exists(path):
            os.remove(path)


if __name__ == '__main__':
    """运行示例"""
    print("\n" + "="*60)
    print("Config 模块使用示例")
    print("="*60)
    
    example_basic_usage()
    example_chaining()
    example_auto_save()
    example_attribute_access()
    
    print("\n" + "="*60)
    print("运行测试: pytest tests/test_config.py -v")
    print("="*60)
