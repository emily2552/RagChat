import time
import threading


class SnowflakeIDGenerator:
    def __init__(self, machine_id=1, datacenter_id=1):
        """
        初始化雪花算法ID生成器

        Args:
            machine_id: 机器ID (0-31)
            datacenter_id: 数据中心ID (0-31)
        """
        self.machine_id = machine_id & 0x1F  # 限制在5位内
        self.datacenter_id = datacenter_id & 0x1F  # 限制在5位内
        self.sequence = 0
        self.last_timestamp = -1
        self.lock = threading.Lock()

        # 配置位移量
        self.MACHINE_ID_BITS = 5
        self.DATACENTER_ID_BITS = 5
        self.SEQUENCE_BITS = 12

        self.MACHINE_LEFT = self.SEQUENCE_BITS
        self.DATACENTER_LEFT = self.SEQUENCE_BITS + self.MACHINE_ID_BITS
        self.TIMESTAMP_LEFT = self.SEQUENCE_BITS + self.MACHINE_ID_BITS + self.DATACENTER_ID_BITS

        # 最大值限制
        self.SEQUENCE_MASK = -1 ^ (-1 << self.SEQUENCE_BITS)  # 4095

        # 起始时间戳 (2023-01-01)
        self.EPOCH = 1672531200000

    def _wait_next_millis(self, last_timestamp):
        """等待下一毫秒"""
        timestamp = self._get_timestamp()
        while timestamp <= last_timestamp:
            timestamp = self._get_timestamp()
        return timestamp

    def _get_timestamp(self):
        """获取当前时间戳(毫秒)"""
        return int(time.time() * 1000)

    def generate_id(self):
        """
        生成唯一的ID

        Returns:
            int: 64位唯一ID
        """
        with self.lock:
            timestamp = self._get_timestamp()

            # 如果当前时间小于上次时间戳，说明发生了时钟回拨
            if timestamp < self.last_timestamp:
                raise Exception("时钟回拨异常")

            # 如果同一毫秒内生成多个ID，则递增序列号
            if timestamp == self.last_timestamp:
                self.sequence = (self.sequence + 1) & self.SEQUENCE_MASK
                # 同一毫秒的序列号用完，等待下一毫秒
                if self.sequence == 0:
                    timestamp = self._wait_next_millis(self.last_timestamp)
            else:
                # 不同毫秒则重置序列号
                self.sequence = 0

            self.last_timestamp = timestamp

            # 组装ID
            new_id = ((timestamp - self.EPOCH) << self.TIMESTAMP_LEFT) | \
                     (self.datacenter_id << self.DATACENTER_LEFT) | \
                     (self.machine_id << self.MACHINE_LEFT) | \
                     self.sequence

            return new_id


# 全局实例
_snowflake_generator = SnowflakeIDGenerator()


def generate_unique_id(machine_id=1, datacenter_id=1):
    """
    生成雪花算法ID的便捷函数

    Args:
        machine_id: 机器ID
        datacenter_id: 数据中心ID

    Returns:
        str: 字符串形式的唯一ID
    """
    global _snowflake_generator
    if _snowflake_generator.machine_id != machine_id or _snowflake_generator.datacenter_id != datacenter_id:
        _snowflake_generator = SnowflakeIDGenerator(machine_id, datacenter_id)

    return str(_snowflake_generator.generate_id())


# 批量生成示例
def generate_batch_ids(count, machine_id=1, datacenter_id=1):
    """
    批量生成ID

    Args:
        count: 生成数量
        machine_id: 机器ID
        datacenter_id: 数据中心ID

    Returns:
        list: ID列表
    """
    ids = []
    generator = SnowflakeIDGenerator(machine_id, datacenter_id)
    for _ in range(count):
        ids.append(str(generator.generate_id()))
    return ids

if __name__ == "__main__":
    # 单个生成示例
    print(generate_unique_id())

