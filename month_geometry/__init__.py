"""Calendar-month residual-stream extraction for Qwen3.5-4B."""

MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
PROMPTS = {"bare": "{month}", "contextualized": "The month of the year is {month}"}
MODELS = ("Qwen/Qwen3.5-4B",)

LEADING_TOKEN = "<|endoftext|>"
