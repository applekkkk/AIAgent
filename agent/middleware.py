from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest

from utils.logger_handler import logger
from utils.prompt_loader import load_report_prompts, load_system_prompts


@wrap_tool_call
def monitor_tool(request,handler):
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")
    try:
        result=handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")
        if request.tool_call['name'] == 'fill_context_for_report':
            request.runtime.context['report']=True
        return result
    except Exception as e:
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e


@before_model
def log_before_model(
        state,         # 整个Agent智能体中的状态记录
        runtime         # 记录了整个执行过程中的上下文信息
):         # 在模型执行前输出日志
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息。")

    logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")

    return None


@dynamic_prompt
def report_prompt_switch(request:ModelRequest):
    if request.runtime.context['report']:
        return load_report_prompts()
    else:
        return load_system_prompts()
