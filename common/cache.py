# 各类缓存： Memory/File/Redis

import os, json, hashlib, redis
import zlib
from typing import Any, Optional, Dict

# 内存缓存类
class InMemoryCache:
    """内存缓存类"""
    def __init__(self) -> None:
        self.cache = {}

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        self.cache[key] = value


# 基本的文件缓存类
class FileCache:
    """文件缓存类，读取、写入和缓存检查功能，使用 JSON 文件来存储缓存数据"""
    def __init__(self, cache_file: str = "cache/cache.json") -> None:
        self.cache_file = cache_file
        self.cache: Dict[str, Any] = {}
        self._load()

    def _load(self):
        """从 JSON 文件加载缓存到内存"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                # 如果文件损坏，则将 self.cache 重置为 {}
                self.cache = {}

    def _save(self):
        """将内存缓存写入到 JSON 文件"""
        # 小技巧： 
        # 使用临时文件 + 替换的方法，防止写入过程中程序崩溃导致文件损坏
        temp_filepath = f"{self.cache_file}.tmp"
        try:
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=4)
            os.replace(temp_filepath, self.cache_file)  # 原子替换
        except IOError as e:
            print(f"缓存文件保存失败：{e}")

    def get(self, key):
        """获取缓存，这里直接从 self.cache 获取，因为在初始化时，已经使用 _load() 进行全量加载了 """
        return self.cache.get(key)

    def set(self, key: str, value: Any):
        """写入缓存，并立即持久化到 JSON 文件"""
        self.cache[key] = value
        self._save()


# 简单的 Redis 缓存类：
class SimpleRedisCache:
    def __init__(self, host='localhost', port=6379, db=0) -> None:
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)

    def get(self, key):
        """
        从 Redis 获取缓存数据
        :param key: 键
        """
        cached_value = self.client.get(key)
        if cached_value:
            # 返回解码后的 JSON 数据：
            return json.loads(cached_value)
        return None

    def set(self, key, value, expiration=3600):
        """
        将数据以 json 格式存储到 Redis，设置过期时间
        :param key: 键
        :param value: 值
        :param expiration: 过期时间
        """
        self.client.setex(key, expiration, json.dumps(value))

    def delete(self, key):
        """
        删除指定的缓存
        :param key: 键
        """
        self.client.delete(key)
        print("指定消息已清除")


# Redis 优化策略：
# 1. 合理的过期时间； 2. LRU(Least Recently Used) 清理策略； 3. 数据压缩(zlib)
class OptimizedRedisCache(SimpleRedisCache):
    def __init__(self, host='localhost', port=6379, db=0, default_ttl: int = 86400) -> None:
        # super().__init__(host, port, db)
        self.default_ttl = default_ttl
        # 因为要进行 zlib 压缩存二进制，所以必须设置 decode_responses = False
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=False)  

        # 2. LRU 清理策略：这行代码会在连接时强制设置 Redis 的淘汰策略为 allkeys-lru
        # （前提是 Redis 配置文件中设置了 maxmemory）
        try:
            self.client.config_set('maxmemory-policy', 'allkeys-lru')
        except Exception:
            pass    # 如果无权限修改，忽略即可，需要再 redis.conf 中手动配置

    def set(self, key, value, expiration: int =3600):
        """
        数据序列化，压缩后存储到 Redis ，并设置过期时间
        """
        # 1. 序列化为 json 字符串
        json_str = json.dumps(value)
        # 2. 编码为 utf-8，并 zlib 压缩为 bytes
        compressed_value = zlib.compress(json_str.encode('utf-8'), level=6)
        # 3. 使用 setex 写入并设置过期时间(依然继承 TTL 机制)
        self.client.setex(key, expiration, compressed_value)

    def get(self, key):
        """
        从 Redis 中获取数据并解压
        """
        compressed_value = self.client.get(key)
        if compressed_value:
            if isinstance(compressed_value, bytes):
                try:
                    # 解压 bytes -> 解码 utf-8 -> 反序列化 JSON
                    decompressed = zlib.decompress(compressed_value)
                    return json.loads(decompressed.decode('utf-8'))
                except Exception:
                    return None
        return None
