import os


class Names:
    TEXT = "text"


# Constants

## Home directory setup
HOME_DIR = os.path.expanduser(os.path.join("~", "mental_health_retrieval_chatbot"))
HOME_DIR = (
    HOME_DIR if os.path.isdir(HOME_DIR) else os.path.join(os.getcwd(), "mental_health_retrieval_chatbot")
)
RESOURCE_DIR = os.path.join(HOME_DIR, "resources")

## Prompt paths
PROMPT_DIR = os.path.join(HOME_DIR, "prompts")
PROMPT_PATH = os.path.join(PROMPT_DIR, "custom_query_prompt.md")

## Model paths
DEFAULT_BASE_CHAT_MODEL = "microsoft/DialoGPT-small"
DEFAULT_CHAT_MODEL = os.path.join(".", RESOURCE_DIR, "empathetic-finetuned-chatbot")
DEFAULT_SVS_MODEL = "snowflake-arctic-embed-xs"


## Dataset paths
EMPATHETIC_DIALOGUES_PATH = os.path.join(RESOURCE_DIR, "empatheticdialogues/train.csv")
DAILY_DIALOG_PATH = os.path.join(RESOURCE_DIR, "dailydialog/dialogues_text.txt")

DEFAULT_CHAT_PROMPT = None

with open(PROMPT_PATH, "r") as f:
    DEFAULT_CHAT_PROMPT = f.read()

TOP_K = 2
