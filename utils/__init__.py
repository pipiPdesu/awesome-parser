import time
import hashlib

def get_current_time_str():
    """获取当前时间字符串"""
    current_time_stamp = time.time()   # 获取当前时间戳
    # 将时间戳转换为本地时间并格式化为字符串
    current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time_stamp))
    return current_time_str  # 输出类似于 "2023-03-17 15:22:01"

def gen_token(msg):
    """生成token"""
    s = get_current_time_str() + msg
    _md5 = hashlib.md5(s.encode()).hexdigest()
    return _md5