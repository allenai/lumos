"""From https://github.com/zeno-ml/zeno-build/blob/main/zeno_build/models/providers/openai_utils.py."""
"""Tools to generate from OpenAI prompts."""

import asyncio
import logging
import os
from typing import Any

import aiolimiter
import openai
import openai.error
from aiohttp import ClientSession
from tqdm.asyncio import tqdm_asyncio

from zeno_build.models import lm_config
from zeno_build.prompts import chat_prompt


async def _throttled_openai_completion_acreate(
    engine: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    async with limiter:
        for _ in range(6):
            try:
                return await openai.Completion.acreate(
                    engine=engine,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except openai.error.RateLimitError:
                await asyncio.sleep(20)
            except asyncio.exceptions.TimeoutError:
                await asyncio.sleep(20)
            except openai.error.InvalidRequestError:
                logging.warning("OpenAI API Invalid Request: Prompt was filtered")
                return {
                    "choices": [
                        {"message": {"content": "Invalid Request: Prompt was filtered"}}
                    ]
                }
            except openai.error.APIConnectionError:
                logging.warning(
                    "OpenAI API Connection Error: Error Communicating with OpenAI"
                )
                await asyncio.sleep(20)
            except openai.error.Timeout:
                logging.warning("OpenAI APITimeout Error: OpenAI Timeout")
                await asyncio.sleep(20)
            except openai.error.ServiceUnavailableError as e:
                logging.warning(f"OpenAI service unavailable error: {e}")
                await asyncio.sleep(20)
            except openai.error.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                await asyncio.sleep(20)
        return {"choices": [{"message": {"content": ""}}]}


async def generate_from_openai_completion(
    full_contexts: list[chat_prompt.ChatMessages],
    prompt_template: chat_prompt.ChatMessages,
    model_config: lm_config.LMConfig,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    requests_per_minute: int = 100,
) -> list[str]:
    """Generate from OpenAI Completion API.

    Args:
        full_contexts: List of full contexts to generate from.
        prompt_template: Prompt template to use.
        model_config: Model configuration.
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.organization = os.environ["OPENAI_ORG"]
    openai.aiosession.set(ClientSession())
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_completion_acreate(
            engine=model_config.model,
            prompt=prompt_template.to_text_prompt(
                full_context=full_context.limit_length(context_length),
                name_replacements=model_config.name_replacements,
            ),
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for full_context in full_contexts
    ]
    responses = await tqdm_asyncio.gather(*async_responses)
    # Note: will never be none because it's set, but mypy doesn't know that.
    await openai.aiosession.get().close()  # type: ignore
    return [x["choices"][0]["text"] for x in responses]


async def _throttled_openai_chat_completion_acreate(
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    limiter: aiolimiter.AsyncLimiter,
) -> dict[str, Any]:
    async with limiter:
        for _ in range(3):
            try:
                return await openai.ChatCompletion.acreate(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
            except openai.error.RateLimitError:
                # logging.warning(
                #     "OpenAI API rate limit exceeded. Sleeping for 10 seconds."
                # )
                await asyncio.sleep(5)
            except asyncio.exceptions.TimeoutError:
                # logging.warning("OpenAI API timeout. Sleeping for 10 seconds.")
                await asyncio.sleep(5)
            except openai.error.InvalidRequestError:
                logging.warning("OpenAI API Invalid Request: Prompt was filtered")
                return {
                    "choices": [
                        {"message": {"content": "Invalid Request: Prompt was filtered"}}
                    ]
                }
            except openai.error.APIConnectionError:
                logging.warning(
                    "OpenAI API Connection Error: Error Communicating with OpenAI"
                )
                await asyncio.sleep(10)
            except openai.error.Timeout:
                logging.warning("OpenAI APITimeout Error: OpenAI Timeout")
                await asyncio.sleep(10)
            except openai.error.ServiceUnavailableError as e:
                logging.warning(f"OpenAI service unavailable error: {e}")
                await asyncio.sleep(10)
            except openai.error.APIError as e:
                logging.warning(f"OpenAI API error: {e}")
                await asyncio.sleep(10)
        return {"choices": [{"message": {"content": ""}}]}


async def generate_from_openai_chat_completion(
    full_contexts: list[chat_prompt.ChatMessages],
    model_config: lm_config.LMConfig,
    temperature: float,
    max_tokens: int,
    top_p: float = 1,
    requests_per_minute: int = 60,
    tqdm: bool = True,
) -> list[str]:
    """Generate from OpenAI Chat Completion API.

    Args:
        full_contexts: List of full contexts to generate from.
        prompt_template: Prompt template to use.
        model_config: Model configuration.
        temperature: Temperature to use.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use.
        context_length: Length of context to use.
        requests_per_minute: Number of requests per minute to allow.

    Returns:
        List of generated responses.
    """
    openai.api_key = os.environ["OPENAI_API_KEY"]
    openai.organization = os.environ["OPENAI_ORG"]
    openai.aiosession.set(ClientSession())
    limiter = aiolimiter.AsyncLimiter(requests_per_minute)
    async_responses = [
        _throttled_openai_chat_completion_acreate(
            model=model_config.model,
            messages=full_context.messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            limiter=limiter,
        )
        for full_context in full_contexts
    ]
    if tqdm:
        responses = await tqdm_asyncio.gather(*async_responses)
    else:
        responses = await asyncio.gather(*async_responses)
    await openai.aiosession.get().close()
    return [x["choices"][0]["message"]["content"].strip() for x in responses]