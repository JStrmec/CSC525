from time import time
from typing import List, Optional

import torch
import logging as logger
from llm_classes import (
    LLMOutput,
    PenaltyLLMModelConfig,
    VLLMInput,
    build_tokens_used,
)
from starlette.requests import Request
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams, TokensPrompt
from .constants import DEFAULT_CHAT_MODEL

SEED = 72


class GenerativeLLM(object):
    def __init__(self) -> None:
        ##############################################################
        # PARAMS PARSING
        ##############################################################
        self.logger = logger
        # Model params parsing
        self.model_uri = DEFAULT_CHAT_MODEL
        self.tokenizer_uri = DEFAULT_CHAT_MODEL
        self.trust_remote_code = False
        self.max_model_len = 64000
        self.dtype = "auto"
        self.quantization = None
        self.gpu_memory_utilization = 0.90
        self.max_batch_size = 16

        ##############################################################
        # MODEL INITIALIZATION
        ##############################################################
        _init_time = time()
        self.logger.info("Loading model...")
        self.logger.info(f"CUDA available: {torch.cuda.is_available()}")

        # Set the seeds for reproducibility
        torch.cuda.manual_seed(SEED)
        torch.manual_seed(SEED)
        model_path = self.model_uri
        tokenizer_path = self.tokenizer_uri

        # Loading LLM model and tokenizer
        self.engine_args = AsyncEngineArgs(
            task="generate",
            disable_log_requests=True,
            disable_log_stats=False,
            model=model_path,
            tokenizer=tokenizer_path,
            trust_remote_code=self.trust_remote_code,
            max_model_len=self.max_model_len,
            dtype=self.dtype,
            quantization=self.quantization,
            gpu_memory_utilization=self.gpu_memory_utilization,
            seed=SEED,
        )
        self.model = AsyncLLMEngine.from_engine_args(engine_args=self.engine_args)
        self.tokenizer = self.model.engine.tokenizer.tokenizer
        self.tokenizer_ids_vocab = set(self.tokenizer.get_vocab().values())
        self.max_model_len = self.model.engine.model_config.max_model_len

        # Default Sampling Params
        self.default_sampling_params = PenaltyLLMModelConfig(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            max_tokens=512,
        )

        # Initialization timings
        self.logger.info(
            f"Model initialization finished in {(time() - _init_time) * 1000:.0f}ms"
        )
        self.logger.info(f"Model details: \n{self}")

    async def predict(
        self,
        request_id: str,
        request: Request,
        llm_input: VLLMInput,
        model_config: Optional[PenaltyLLMModelConfig] = None,
    ) -> LLMOutput:
        context_logger = self.logger

        try:
            _start_time = time()
            ######################################################
            # LLM MODEL CONFIG PARSING
            ######################################################
            if model_config is not None:
                context_logger.debug(f"Custom model_config: {model_config}")

                # Merging default and custom sampling params into sampling params
                sampling_params = PenaltyLLMModelConfig(
                    **{
                        **self.default_sampling_params.model_dump(exclude_none=True),
                        **model_config.model_dump(exclude_none=True),
                    }
                )
                context_logger.debug(f"Merged model_config: {sampling_params}")
            else:
                sampling_params = self.default_sampling_params
            sampling_params = SamplingParams(
                **sampling_params.model_dump(exclude_none=True)
            )
            context_logger.debug(f"{sampling_params}")

            ######################################################
            # TOKENIZATION
            ######################################################
            _tokenization_time = None
            if llm_input.prompt_token_ids is None:
                if llm_input.prompt is None:
                    raise ValueError(
                        "At least one of prompt or prompt_token_ids must be provided!"
                    )

                llm_input.prompt_token_ids = self.tokenizer(
                    text=llm_input.prompt,
                    max_length=self.max_model_len - sampling_params.max_tokens,
                    truncation=True,
                )["input_ids"]
            else:
                if not self.check_token_ids(token_ids=llm_input.prompt_token_ids):
                    raise ValueError("Request prompt_token_ids are not valid!")
            _tokenization_time = time()

            ######################################################
            # LLM INFERENCE
            ######################################################

            # Submitting generation request
            results_generator = self.model.generate(
                request_id=request_id,
                prompt=TokensPrompt(prompt_token_ids=llm_input.prompt_token_ids),
                sampling_params=sampling_params,
            )

            # Obtaining generation output
            request_output = None
            async for result in results_generator:
                if await request.is_disconnected():
                    context_logger.info(f"Aborting disconnected request: {request_id}")
                    await self.model.abort(request_id=request_id)
                    return LLMOutput(success=False, warning="Request aborted")
                request_output = result
            if request_output is None:
                raise RuntimeError("Failed to get prediction from AsyncLLMEngine")

            generated_output = request_output.outputs[0]
            generated_text = generated_output.text
            _generation_time = time()

            ######################################################
            # BUILDING LLM OUTPUT
            ######################################################
            tokens_used = build_tokens_used(
                prompt_tokens=len(request_output.prompt_token_ids),
                completion_tokens=len(generated_output.token_ids),
            )
            llm_output = LLMOutput(
                text=generated_text,
                tokens_used=tokens_used,
                success=True,
            )

            ######################################################
            # LOGGING TIMINGS AND TOKENS USED
            ######################################################
            infer_timings = {
                "tokenization": _tokenization_time - _start_time,
                "generation": _generation_time - _tokenization_time,
            }
            for k, v in infer_timings.items():
                infer_timings[k] = f"{v * 1000:.0f}ms"
            context_logger.info(
                f"Time taken to predict: {(time() - _start_time) * 1000:.0f}ms "
                f"{infer_timings | tokens_used.dict()}"
            )

            return llm_output
        except Exception as e:
            context_logger.error(
                f"Error during prediction: {e}",
                exc_info=True,
            )
            return LLMOutput(success=False, error=str(e))

    def check_token_ids(self, token_ids: List[int]) -> bool:
        return all([token_id in self.tokenizer_ids_vocab for token_id in token_ids])
