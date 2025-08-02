import json
from typing import List, Optional

from pydantic import Field

from .api_model import APIModel

from pydantic import AliasChoices


######################################################
# ARGUMENT KEY CONSTANTS
######################################################
BATCH_KEY = "batch"
USE_CASE_CONFIG_KEY = "use_case_config"
MODEL_CONFIG_KEY = "model_config"


######################################################
# OUTPUT MODELS
######################################################
class Output(APIModel, extra="forbid"):
    success: bool
    warning: Optional[str] = None


######################################################
# INPUT MODELS
######################################################
class TextInput(APIModel, extra="forbid"):
    text: str = Field(min_length=1, description="Text input.")


######################################################
# VARIETY OF CONFIGS MODELS
######################################################
class UseCaseConfig(APIModel, extra="forbid"):
    pass


class ModelConfig(APIModel, extra="forbid"):
    pass


class InternalConfig(APIModel, extra="allow"):
    pass


class RequestParams(APIModel, extra="forbid"):
    batch: List[TextInput] = Field(min_length=1, description="Batch of text inputs.")
    ml_config: Optional[ModelConfig] = Field(
        default=None,
        validation_alias=AliasChoices("modelConfig", "model_config"),
        serialization_alias="model_config",
    )
    use_case_config: Optional[UseCaseConfig] = None


class LLMInputTokensUsed(APIModel, extra="forbid"):
    prompt_tokens: int


class LLMInput(APIModel, extra="forbid"):
    prompt: str
    prefill: Optional[str] = None
    tokens_used: Optional[LLMInputTokensUsed] = None


class VLLMInput(LLMInput):
    prompt: Optional[str] = None
    prompt_token_ids: Optional[List[int]] = None


class LLMModelConfig(ModelConfig):
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="""Sampling temperature to use, between 0 and 2.
            A higher sampling temperature, like 0.8, will make the output more random,
            while a lower value, like 0.2, will make the output more focused and
            deterministic.""",
    )
    top_p: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="""A floating-point number that controls the cumulative 
            probability of the top tokens to consider. It must be in the range: (0, 1].
            Set it to 1 to consider all tokens.""",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=-1,
        description="""An integer that controls the number of top tokens to consider.
            Set top_k to -1 to consider all tokens.""",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of tokens to generate per output sequence",
    )


class PenaltyLLMModelConfig(LLMModelConfig):
    presence_penalty: Optional[float] = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description="""A floating-point number that penalizes new tokens based on
            whether they have already appeared in the text.
            A value > 0 encourages the model to use new tokens,
            while a value < 0 encourages the model to repeat existing tokens.""",
    )
    frequency_penalty: Optional[float] = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        description="""A floating-point number that penalizes new tokens based on 
            their frequency in the generated text. A value > 0 encourages the model to
            use new tokens, while a value < 0 encourages the model to repeat tokens.""",
    )


class LLMTokensUsed(LLMInputTokensUsed, extra="forbid"):
    completion_tokens: int
    total_tokens: int


class LLMOutput(Output, extra="forbid"):
    text: Optional[str] = Field(
        default=None,
        description="The generated text output from the model",
    )
    tokens_used: Optional[LLMTokensUsed] = Field(
        default=None,
        description="The number of tokens used in the generation process",
    )


def build_tokens_used(
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: Optional[int] = None,
) -> LLMTokensUsed:
    if total_tokens is None:
        total_tokens = prompt_tokens + completion_tokens

    return LLMTokensUsed(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def merge_tokens_used(left: LLMTokensUsed, right: LLMTokensUsed) -> LLMTokensUsed:
    return LLMTokensUsed(
        prompt_tokens=left.prompt_tokens + right.prompt_tokens,
        completion_tokens=left.completion_tokens + right.completion_tokens,
        total_tokens=left.total_tokens + right.total_tokens,
    )


# This function is used to parse a string value, from useCaseConfig or modelConfig,
# into a dictionary if it is a valid JSON string.
def parse_dict_config_variable(
    value: str,
    variable_name: str,
    config: str = "modelConfig",
) -> str:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            raise ValueError(
                f"Invalid {config} for {variable_name}: "
                f"'{value}' is not a valid JSON string."
            )
    return value
