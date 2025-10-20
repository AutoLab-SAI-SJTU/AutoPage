import re
import io
import contextlib
import traceback
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE, MSO_SHAPE, MSO_AUTO_SHAPE_TYPE
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig, QwenConfig, VLLMConfig, OpenRouterConfig, GeminiConfig
import math
from urllib.parse import quote_from_bytes, quote
from PIL import Image
import os
import copy
import io
from utils.src.utils import ppt_to_images
from playwright.sync_api import sync_playwright
from pathlib import Path
from playwright.async_api import async_playwright
import asyncio

def get_agent_config(model_type):
    agent_config = {}
    if model_type == 'qwen':
        agent_config = {
            "model_type": ModelType.DEEPINFRA_QWEN_2_5_72B,
            "model_config": QwenConfig().as_dict(),
            "model_platform": ModelPlatformType.DEEPINFRA,
        }
    elif model_type == 'gemini':
        agent_config = {
            "model_type": ModelType.GEMINI_2_5_PRO,
            "model_config": GeminiConfig().as_dict(),
            "model_platform": ModelPlatformType.GEMINI,
            'max_images': 99
        }
    elif model_type == 'phi4':
        agent_config = {
            "model_type": ModelType.DEEPINFRA_PHI_4_MULTIMODAL,
            "model_config": QwenConfig().as_dict(),
            "model_platform": ModelPlatformType.DEEPINFRA,
        }
    elif model_type == 'qwq-plus':
        agent_config = {
            "model_type": ModelType.QWEN_QWQ_PLUS,
            "model_config": QwenConfig().as_dict(),
            'model_platform': ModelPlatformType.QWEN
        }
    elif model_type == 'qwen-vl-max':
        agent_config = {
            "model_type": ModelType.QWEN_VL_MAX,
            "model_config": QwenConfig().as_dict(),
            'model_platform': ModelPlatformType.QWEN,
            'max_images': 99
        }
    elif model_type == 'qwen-plus':
        agent_config = {
            "model_type": ModelType.QWEN_PLUS,
            "model_config": QwenConfig().as_dict(),
            'model_platform': ModelPlatformType.QWEN,
            'max_images': 99
        }
    elif model_type == 'qwen-max':
        agent_config = {
            "model_type": ModelType.QWEN_MAX,
            "model_config": QwenConfig().as_dict(),
            'model_platform': ModelPlatformType.QWEN,
        }
    elif model_type == 'qwen-long':
        agent_config = {
            "model_type": ModelType.QWEN_LONG,
            "model_config": QwenConfig().as_dict(),
            'model_platform': ModelPlatformType.QWEN,
        }
    elif model_type == 'llama-4-scout-17b-16e-instruct':
        agent_config = {
            'model_type': ModelType.ALIYUN_LLAMA4_SCOUT_17B_16E,
            'model_config': QwenConfig().as_dict(),
            'model_platform': ModelPlatformType.QWEN,
            'max_images': 99
        }
    elif model_type == 'qwen-2.5-vl-72b':
        agent_config = {
            'model_type': ModelType.QWEN_2_5_VL_72B,
            'model_config': QwenConfig().as_dict(),
            'model_platform': ModelPlatformType.QWEN,
            'max_images': 99
        }
    elif model_type == 'gemini-2.5-pro':
        agent_config = {
            'model_type': ModelType.GEMINI_2_5_PRO,
            'model_config': GeminiConfig().as_dict(),
            'model_platform': ModelPlatformType.GEMINI,
            'max_images': 99
        }
    elif model_type == 'gemini-2.5-flash':
        agent_config = {
            'model_type': ModelType.GEMINI_2_5_FLASH,
            'model_config': GeminiConfig().as_dict(),
            'model_platform': ModelPlatformType.GEMINI,
            'max_images': 99
        }
    elif model_type == 'gemma':
        agent_config = {
            "model_type": "google/gemma-3-4b-it",
            "model_platform": ModelPlatformType.VLLM,
            "model_config": VLLMConfig().as_dict(),
            "url": 'http://localhost:5555/v1',
            'max_images': 99
        }
    elif model_type == 'llava':
        agent_config = {
            "model_type": "llava-hf/llava-onevision-qwen2-7b-ov-hf",
            "model_platform": ModelPlatformType.VLLM,
            "model_config": VLLMConfig().as_dict(),
            "url": 'http://localhost:8000/v1',
            'max_images': 99
        }
    elif model_type == 'molmo-o':
        agent_config = {
            "model_type": "allenai/Molmo-7B-O-0924",
            "model_platform": ModelPlatformType.VLLM,
            "model_config": VLLMConfig().as_dict(),
            "url": 'http://localhost:8000/v1',
            'max_images': 99
        }
    elif model_type == 'qwen-2-vl-7b':
        agent_config = {
            "model_type": "Qwen/Qwen2-VL-7B-Instruct",
            "model_platform": ModelPlatformType.VLLM,
            "model_config": VLLMConfig().as_dict(),
            "url": 'http://localhost:8000/v1',
            'max_images': 99
        }
    elif model_type == 'vllm_phi4':
        agent_config = {
            "model_type": "microsoft/Phi-4-multimodal-instruct",
            "model_platform": ModelPlatformType.VLLM,
            "model_config": VLLMConfig().as_dict(),
            "url": 'http://localhost:8000/v1',
            'max_images': 99
        }
    elif model_type == 'o3-mini':
        agent_config = {
            "model_type": ModelType.O3_MINI,
            "model_config": ChatGPTConfig().as_dict(),
            "model_platform": ModelPlatformType.OPENAI,
        }
    elif model_type == 'gpt-4.1':
        agent_config = {
            "model_type": ModelType.GPT_4_1,
            "model_config": ChatGPTConfig().as_dict(),
            "model_platform": ModelPlatformType.OPENAI,
        }
    elif model_type == 'gpt-4.1-mini':
        agent_config = {
            "model_type": ModelType.GPT_4_1_MINI,
            "model_config": ChatGPTConfig().as_dict(),
            "model_platform": ModelPlatformType.OPENAI,
        }
    elif model_type == '4o':
        agent_config = {
            "model_type": ModelType.GPT_4O,
            "model_config": ChatGPTConfig().as_dict(),
            "model_platform": ModelPlatformType.OPENAI,
            # "model_name": '4o'
        }
    elif model_type == '4o-mini':
        agent_config = {
            "model_type": ModelType.GPT_4O_MINI,
            "model_config": ChatGPTConfig().as_dict(),
            "model_platform": ModelPlatformType.OPENAI,
        }
    elif model_type == 'o1':
        agent_config = {
            "model_type": ModelType.O1,
            "model_config": ChatGPTConfig().as_dict(),
            "model_platform": ModelPlatformType.OPENAI,
            # "model_name": 'o1'
        }
    elif model_type == 'o3':
        agent_config = {
            "model_type": ModelType.O3,
            "model_config": ChatGPTConfig().as_dict(),
            "model_platform": ModelPlatformType.OPENAI,
        }
    elif model_type == 'vllm_qwen_vl':
        agent_config = {
            "model_type": "Qwen/Qwen2.5-VL-7B-Instruct",
            "model_platform": ModelPlatformType.VLLM,
            "model_config": VLLMConfig().as_dict(),
            "url": 'http://localhost:7000/v1'
        }
    elif model_type == 'vllm_qwen':
        agent_config = {
            "model_type": "Qwen/Qwen2.5-7B-Instruct",
            "model_platform": ModelPlatformType.VLLM,
            "model_config": VLLMConfig().as_dict(),
            "url": 'http://localhost:8000/v1',
        }
    elif model_type == 'openrouter_qwen_vl_72b':
        agent_config = {
            'model_type': ModelType.OPENROUTER_QWEN_2_5_VL_72B,
            'model_platform': ModelPlatformType.OPENROUTER,
            'model_config': OpenRouterConfig().as_dict(),
        }
    elif model_type == 'openrouter_qwen_vl_7b':
        agent_config = {
            'model_type': ModelType.OPENROUTER_QWEN_2_5_VL_7B,
            'model_platform': ModelPlatformType.OPENROUTER,
            'model_config': OpenRouterConfig().as_dict(),
        }
    elif model_type == 'openrouter_grok-4-fast':
        agent_config = {
            'model_type': ModelType.OPENROUTER_GROK_4_FAST,
            'model_platform': ModelPlatformType.OPENROUTER,
            'model_config': OpenRouterConfig().as_dict(),
        }
    elif model_type == 'openrouter_gemini-2.5-flash':
        agent_config = {
            'model_type': ModelType.OPENROUTER_GEMINI_2_5_FLASH,
            'model_platform': ModelPlatformType.OPENROUTER,
            'model_config': OpenRouterConfig().as_dict(),
        }
    elif model_type == 'openrouter_gpt-4o-mini':
        agent_config = {
            'model_type': ModelType.OPENROUTER_GPT_4O_MINI,
            'model_platform': ModelPlatformType.OPENROUTER,
            'model_config': OpenRouterConfig().as_dict(),
        }
    elif model_type == 'openrouter_qwen-plus':
        agent_config = {
            'model_type': ModelType.OPENROUTER_QWEN_PLUS,
            'model_platform': ModelPlatformType.OPENROUTER,
            'model_config': OpenRouterConfig().as_dict(),
        }
    elif model_type == 'openrouter_qwen-vl-max':
        agent_config = {
            'model_type': ModelType.OPENROUTER_QWEN_VL_MAX,
            'model_platform': ModelPlatformType.OPENROUTER,
            'model_config': OpenRouterConfig().as_dict(),
        }
    elif model_type == 'openrouter_openai/o3':
        agent_config = {
            'model_type': ModelType.OPENROUTER_o3,
            'model_platform': ModelPlatformType.OPENROUTER,
            'model_config': OpenRouterConfig().as_dict(),
        }
    elif model_type =='openrouter_claude-sonnet-4.5':
        agent_config = {
            'model_type': ModelType.OPENROUTER_CLAUDE_SONNET_4_5,
            'model_platform': ModelPlatformType.OPENROUTER,
            'model_config': OpenRouterConfig().as_dict(),
        }
    else:
        agent_config = {
            'model_type': model_type,
            'model_platform': ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
            'model_config': None
        }
    
    return agent_config

def account_token(response):
    input_token = response.info['usage']['prompt_tokens']
    output_token = response.info['usage']['completion_tokens']

    return input_token, output_token
