import logging

# 创建全局 logger
logger = logging.getLogger("global_logger")
logger.setLevel(logging.INFO)  # 全局级别 INFO

# 创建控制台 handler 并设置级别
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)  # handler 也设为 INFO

# 日志输出格式
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - "
    "%(filename)s:%(lineno)d - %(funcName)s() - %(message)s"
)
ch.setFormatter(formatter)

# 先清空已存在的 handler，避免重复
logger.handlers.clear()

# 添加 handler
logger.addHandler(ch)
