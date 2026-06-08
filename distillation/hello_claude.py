from agents.backend import AnthropicBackend
import anthropic
from dotenv import load_dotenv
load_dotenv()

backend = AnthropicBackend(model="claude-sonnet-4-6")
messages = [
    [
        {"role": "system", "content": "You are playing Wordle. Please be diverse in your first word guess. Reply with <think>reasoning</think> then <guess>WORD</guess>.",
            "cache_control": {"type": "ephemeral"}},

        {"role": "user", "content": "Make your first guess."}
    ],
    [
        {"role": "system", "content": "You are playing Wordle. Please be diverse in your first word guess. Reply with <think>reasoning</think> then <guess>WORD</guess>.",
            "cache_control": {"type": "ephemeral"}},
        {"role": "user", "content": "Make your first guess."}
    ]
]

completions = backend.generate(messages)

for c in completions:
    print("."*50)
    print("TEXT:\n", c.text)
    print("\nFINISH:", c.finish_reason)

# client = anthropic.Anthropic()

# response = client.messages.create(
#     model="claude-haiku-4-5-20251001",
#     max_tokens=1024,
#     system=[
#         {
#             "type": "text",
#             "text": "You are an AI assistant tasked with analyzing literary works. Your goal is to provide insightful commentary on themes, characters, and writing style.",
#             "cache_control": {"type": "ephemeral"}  # ← inside the block
#         }
#     ],
#     messages=[
#         {"role": "user", "content": "Analyze the major themes in 'Pride and Prejudice'."}
#     ],
# )

# print(response.model_dump_json())
