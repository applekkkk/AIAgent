import os
from utils.logger_handler import logger
from langchain_core.tools import tool
from rag.rag_service import RagService
import random
from utils.config_handler import agent_conf
from utils.path_handler import get_abs_path

rag = RagService()

user_ids = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010", ]
month_arr = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", ]

external_data = {}


@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return rag.summarize(query)


@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city: str) -> str:
    return f"城市{city}天气为晴天，气温26摄氏度，空气湿度50%，南风1级，AQI21，最近6小时降雨概率极低"


@tool(description="获取用户所在城市的名称，以纯字符串形式返回")
def get_user_location() -> str:
    return random.choice(["深圳", "合肥", "杭州"])


@tool(description="获取用户的ID，以纯字符串形式返回")
def get_user_id() -> str:
    return random.choice(user_ids)


@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month() -> str:
    return random.choice(month_arr)


def generate_external_data():
    if not external_data:
        data_path = get_abs_path(agent_conf['external_data_path'])
        if not os.path.exists(path=data_path):
            raise FileNotFoundError(f"外部数据文件{data_path}不存在")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines()[1:]:
                record = line.strip().split(',')
                user_id: str = record[0].replace('"', "")
                feature: str = record[1].replace('"', "")
                efficiency: str = record[2].replace('"', "")
                consumables: str = record[3].replace('"', "")
                comparison: str = record[4].replace('"', "")
                time: str = record[5].replace('"', "")
                if user_id not in external_data:
                    external_data[user_id] = {}
                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparison,
                }


@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回，如果未检索到返回空字符串")
def fetch_external_data(user_id, time_):
    generate_external_data()
    try:
        return external_data[user_id][time_]
    except KeyError:
        logger.warning(f"[fetch_external_data]未能检索到用户：{user_id}在{time_}的使用记录数据")
        return ""


@tool(description="状态注入")
def fill_context_for_report():
    return ''
