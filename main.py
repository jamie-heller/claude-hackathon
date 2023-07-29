from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import os

my_api_key = os.environ.get("ANTHROPIC_API_KEY")
anthropic = Anthropic(api_key=my_api_key)
stream = anthropic.completions.create(
    model="claude-2",
    max_tokens_to_sample=300,
    prompt=f"{HUMAN_PROMPT} How many toes do dogs have?{AI_PROMPT}",
    stream=True,
)
for completion in stream:
    print(completion.completion, end="", flush=True)

print("")

