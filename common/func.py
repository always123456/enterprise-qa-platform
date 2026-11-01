# 功能函数：
from typing import Optional, Union, Dict, List, Any

# Responses API:
def generate_text_with_responses(client, 
                                 instructions, 
                                 input, 
                                 model_name="deepseek-flash", 
                                 temperature=0.7, 
                                 max_output_tokens=4096,
                                 top_p=0.5):
    import openai
    try:
        # 注意：openai SDK 的 Responses API 挂在 client.responses 下（复数），
        # 写成 client.response.create 会抛 AttributeError
        response = client.responses.create(
            model=model_name,
            instructions=instructions,
            input=input,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            top_p=top_p,
        )
        return response.output_text
    except openai.OpenAIError as e:
        print("API 调用失败：", e)
        return None
    except Exception as e:
        print("发生未知错误：", e)
        return None


# Chat Completions API:
def generate_text_with_chat(client, 
                            prompt: str, 
                            system_prompt: Optional[Union[str, List[str], Dict[Any, Any]]] = None, 
                            model_name="deepseek-flash", 
                            temperature=0.7, 
                            max_tokens=4896,
                            top_p=0.5,
                            frequency_penalty=0.0):
    from openai import OpenAIError

    messages = []

    # 处理 system_prompt 类型转换 :
    processed_system_prompt = ""

    # 这里需要结合业务，按需解析 system_prompt，这里只是做了很简单的从 list/dict/str  ->  str 的处理：
    if system_prompt is not None:
        if isinstance(system_prompt, str):
            processed_system_prompt = system_prompt
        elif isinstance(system_prompt, list):
            processed_system_prompt = "\n".join([str(item) for item in system_prompt])
        elif isinstance(system_prompt, dict):
            processed_system_prompt = str(system_prompt.get("content"))

    # 注意：这里要用转换后的 processed_system_prompt（原代码误用了原始 system_prompt，
    # 若传入 list/dict 会导致 API 序列化报错）；为空时干脆不发 system 消息
    if processed_system_prompt:
        messages.append({"role": "system", "content": processed_system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
        )

        return response.choices[0].message.content or ""
    except OpenAIError as e:
        print("API 调用失败：", e)
        return None
    except Exception as e:
        print("发生未知错误：", e)
        return None