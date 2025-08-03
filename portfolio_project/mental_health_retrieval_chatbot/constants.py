import os


class Names:
    TEXT = "text"
    EOU = "__eou__"


# Constants

## Home directory setup
HOME_DIR = os.path.expanduser(os.path.join("~", "mental_health_retrieval_chatbot"))
HOME_DIR = (
    HOME_DIR
    if os.path.isdir(HOME_DIR)
    else os.path.join(os.getcwd(), "mental_health_retrieval_chatbot")
)
RESOURCE_DIR = os.path.join(HOME_DIR, "resources")

## Prompt paths
PROMPT_DIR = os.path.join(HOME_DIR, "prompts")
PROMPT_PATH = os.path.join(PROMPT_DIR, "chat_prompt.md")

## Model paths
DEFAULT_BASE_CHAT_MODEL = "microsoft/DialoGPT-small"
base_model = "empathetic-finetuned-chatbot"
DEFAULT_CHAT_MODEL = (
    DEFAULT_BASE_CHAT_MODEL  # = os.path.join(".", RESOURCE_DIR, base_model)
)

DEFAULT_SVS_MODEL = "Snowflake/snowflake-arctic-embed-l-v2.0"


## Dataset paths
EMPATHETIC_DIALOGUES_PATH = os.path.join(RESOURCE_DIR, "empatheticdialogues/train.csv")
DAILY_DIALOG_PATH = os.path.join(RESOURCE_DIR, "dailydialog/dialogues_text.txt")
DIALOGPT_JSON_PATH = os.path.join(RESOURCE_DIR, "dialoGPT_ready.jsonl")
MENTAL_HEALTH_CONVO_PATH = os.path.join(RESOURCE_DIR, "intents.json")
COUNSEL_CHAT_PATH = os.path.join(RESOURCE_DIR, "counselchat-data.csv")

SAVED_VECTOR_STORE = os.path.join(RESOURCE_DIR, "mental_health.index")

BASE_INSTRUCTIONS = None

with open(PROMPT_PATH, "r") as f:
    BASE_INSTRUCTIONS = f.read()

USE_FINE_TUNED = False

TOP_K = 1
CHAR_TRUNCATION = 5000
MAX_INPUT_TOKENS = 1024

CLEANING_MAP = {"\n": " ", "_comma_": ",", "_period_": ".", "&nbsp;": " ", " BOT:": ""}
