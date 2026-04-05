import os
import json
import time
import requests
from minisweagent.agents.default import DefaultAgent, LimitsExceeded


class MemoryAgent(DefaultAgent):
    def __init__(self, memory_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_path = f"{memory_path}/memory.json"
        self.memorized_messages = []
        self.load_memory()

    def load_memory(self) -> None:
        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r") as f:
                    raw_memory = json.load(f)

                # Wrap memory nicely for the model
                self.memorized_messages = [
                    {
                        "role": "system",
                        "content": f"[MEMORY] Previous step:\n{m.get('content', '')}"
                    }
                    for m in raw_memory
                ]

                print(f"loaded {len(self.memorized_messages)} memory entries")

            except Exception:
                print("failed to load memory (corrupted file?)")
                self.memorized_messages = []

    def save_memory(self) -> None:
        useful_memory = []

        for msg in self.messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")

                # keep only meaningful outputs
                if "```bash" in content or "THOUGHT" in content:
                    useful_memory.append({
                        "role": "assistant",
                        "content": content,
                        "timestamp": time.time()
                    })

        # keep last 10 entries only
        useful_memory = useful_memory[-10:]

        try:
            with open(self.memory_path, "w") as f:
                json.dump(useful_memory, f, indent=2)

            print(f"saved {len(useful_memory)} filtered messages to memory")

        except Exception as e:
            print("failed to save memory:", e)

    def print_spend(self) -> None:
        u = self.model.config.model_kwargs["api_base"]
        h = {"Authorization": f'Bearer {self.model.config.model_kwargs["api_key"]}'}

        try:
            info = requests.get(f"{u}/user/info", headers=h, timeout=2).json()
            user_spend = f'{info["user_info"]["spend"]:.4f} / {info["user_info"]["max_budget"]:.4f}'
        except Exception:
            user_spend = "???"

        try:
            info = requests.get(f"{u}/key/info", headers=h, timeout=2).json()
            key_spend = f'{info["info"]["spend"]:.4f} / {info["info"]["max_budget"]:.4f}'
        except Exception:
            key_spend = "???"

        print(f"spend so far: key {key_spend}, user {user_spend}")

    def query(self) -> dict:
        if (
            0 < self.config.step_limit <= self.model.n_calls
            or 0 < self.config.cost_limit <= self.model.cost
        ):
            raise LimitsExceeded()

        self.print_spend()
        print(f"query llm: step {self.model.n_calls}")

        # only use recent memory
        recent_memory = self.memorized_messages[-5:]

        messages = [
            *self.messages[:1],
            *recent_memory,
            *self.messages[1:],
        ]

        response = self.model.query(messages)
        self.add_message("assistant", **response)
        return response