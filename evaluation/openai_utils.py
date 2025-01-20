
SYSTEM_ROLL = "developer"
USER_ROLL = "user"
MODEL_ROLL = "assistant"

def _add_msg(messages : list[dict[str,str]], role : str, content : str):
    messages.append({"role": role, "content": content})

def add_sys_msg(messages : list[dict[str,str]], content : str):
    _add_msg(messages, SYSTEM_ROLL, content)

def add_user_msg(messages : list[dict[str,str]], content : str):
    _add_msg(messages, USER_ROLL, content)

def add_model_msg(messages : list[dict[str,str]], content : str):
    _add_msg(messages, MODEL_ROLL, content)

