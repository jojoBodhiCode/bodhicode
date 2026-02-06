"""
Prepare training data for LoRA fine-tuning of the Buddhist scholar agent.

Exports approved drafts and generates additional Q&A pairs in ChatML/ShareGPT
format compatible with Qwen2.5's native template and Unsloth training.

Output format (JSONL):
  {"conversations": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]}

Usage:
  python prepare_training_data.py --output training_data.jsonl
"""

import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from prompts import SYSTEM_PROMPT


# ─── Configuration ───────────────────────────────────────────────────────────

CONFIG_DIR = Path.home() / ".config" / "dharma-agent"
DRAFTS_DIR = CONFIG_DIR / "drafts"


def collect_approved_drafts() -> list:
    """Collect all published/approved drafts as training examples."""
    examples = []

    if not DRAFTS_DIR.exists():
        print("  No drafts directory found.")
        return examples

    for draft_file in sorted(DRAFTS_DIR.glob("*.json")):
        try:
            with open(draft_file, encoding="utf-8") as f:
                draft = json.load(f)

            if draft.get("status") != "published":
                continue

            content = draft.get("content", "").strip()
            if not content or len(content) < 50:
                continue

            metadata = draft.get("metadata", {})
            draft_type = draft.get("type", "")

            if draft_type == "post":
                topic = metadata.get("topic_seed", metadata.get("title", ""))
                user_msg = f"""Write a Moltbook post about the following topic. \
Keep it under 300 words, scholarly but engaging.

Topic: {topic}"""

                # Reconstruct the full response with title
                title = metadata.get("title", "")
                if title:
                    assistant_msg = f"TITLE: {title}\nCONTENT: {content}"
                else:
                    assistant_msg = content

            elif draft_type == "comment":
                post_title = metadata.get("post_title", "")
                post_author = metadata.get("post_author", "")
                user_msg = f"""Write a thoughtful comment on this Moltbook post from \
your perspective as a Buddhist scholar. Keep it under 150 words.

Post title: {post_title}
Post by: {post_author}"""
                assistant_msg = content

            else:
                continue

            example = {
                "conversations": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ]
            }
            examples.append(example)

        except Exception as e:
            print(f"  ⚠️  Error reading {draft_file.name}: {e}")
            continue

    return examples


# ─── Q&A pair templates for additional training data ─────────────────────────

QA_TEMPLATES = [
    {
        "user": "How does Nagarjuna's concept of sunyata differ from nihilism?",
        "category": "Madhyamaka",
    },
    {
        "user": "What is the Theravada understanding of anatta (non-self)?",
        "category": "Theravada",
    },
    {
        "user": "Explain the difference between Svatantrika and Prasangika Madhyamaka.",
        "category": "Madhyamaka",
    },
    {
        "user": "What is tathagatagarbha (buddha-nature) and which traditions accept it?",
        "category": "cross-tradition",
    },
    {
        "user": "How does Dharmakirti's epistemology define valid cognition (pramana)?",
        "category": "Pramana",
    },
    {
        "user": "What are the key differences between Yogacara and Madhyamaka philosophy?",
        "category": "Indian philosophy",
    },
    {
        "user": "Explain dependent origination (pratityasamutpada) and its significance.",
        "category": "core doctrine",
    },
    {
        "user": "What is the two truths doctrine and how does it function in Madhyamaka?",
        "category": "Madhyamaka",
    },
    {
        "user": "How does Tsongkhapa's Lam Rim Chen Mo organize the Buddhist path?",
        "category": "Tibetan",
    },
    {
        "user": "What is the Bodhisattva ideal and how does it differ across traditions?",
        "category": "cross-tradition",
    },
]


def generate_qa_pairs_with_model(cfg: dict, templates: list) -> list:
    """
    Generate high-quality Q&A pairs using the existing model.

    This uses the local LLM to generate training examples from templates.
    Requires the model backend to be running.
    """
    examples = []

    try:
        import requests
    except ImportError:
        print("  ❌ requests library needed for model generation")
        return examples

    backend = cfg.get("backend", "ollama")
    if backend == "llama-server":
        url = f"{cfg['llama_server_url']}/v1/chat/completions"
    else:
        url = f"{cfg.get('ollama_base_url', 'http://localhost:11434')}/api/chat"

    for i, template in enumerate(templates):
        print(f"  Generating Q&A pair {i+1}/{len(templates)}: {template['user'][:50]}...")

        if backend == "llama-server":
            payload = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": template["user"]},
                ],
                "temperature": 0.3,
                "max_tokens": 1024,
            }
            try:
                r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=600)
                r.raise_for_status()
                response = r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"    ⚠️  Error: {e}")
                continue
        else:
            payload = {
                "model": cfg.get("ollama_model", "llama3.1:8b"),
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": template["user"]},
                ],
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 1024},
            }
            try:
                r = requests.post(url, json=payload, timeout=300)
                r.raise_for_status()
                response = r.json()["message"]["content"]
            except Exception as e:
                print(f"    ⚠️  Error: {e}")
                continue

        example = {
            "conversations": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": template["user"]},
                {"role": "assistant", "content": response},
            ]
        }
        examples.append(example)

    return examples


def main():
    parser = argparse.ArgumentParser(description="Prepare Buddhist scholar training data")
    parser.add_argument("--output", "-o", default="training_data.jsonl",
                        help="Output JSONL file path")
    parser.add_argument("--generate", "-g", action="store_true",
                        help="Also generate Q&A pairs using local model")
    parser.add_argument("--config", "-c", default=None,
                        help="Path to config.json (for model generation)")
    args = parser.parse_args()

    all_examples = []

    # 1. Collect approved drafts
    print("\n📝 Collecting approved drafts...")
    drafts = collect_approved_drafts()
    print(f"  Found {len(drafts)} published drafts")
    all_examples.extend(drafts)

    # 2. Optionally generate Q&A pairs
    if args.generate:
        print("\n🤖 Generating Q&A pairs with local model...")
        config_path = args.config or str(CONFIG_DIR / "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            qa_pairs = generate_qa_pairs_with_model(cfg, QA_TEMPLATES)
            print(f"  Generated {len(qa_pairs)} Q&A pairs")
            all_examples.extend(qa_pairs)
        else:
            print(f"  ⚠️  Config not found at {config_path}, skipping generation")

    # 3. Write output
    if not all_examples:
        print("\n  No training examples found. Approve some drafts first!")
        return

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"\n✅ Wrote {len(all_examples)} examples to {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
    print(f"\n💡 Upload this to Google Colab for QLoRA fine-tuning")


if __name__ == "__main__":
    main()
