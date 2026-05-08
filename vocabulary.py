"""
Seed vocabulary for the WBR agent.

Organized by category so the agent has broad conceptual coverage from the
start. New winners discovered during play are appended automatically when
Config.auto_expand_vocab is True.
"""

import json
import os
from typing import List


SEED_VOCABULARY: List[str] = [
    # ── Classic rock-paper-scissors extensions ────────────────────────────────
    "rock", "paper", "scissors", "fire", "water", "air", "earth",
    "lightning", "wind", "ice", "lava", "mud", "sand", "glass",

    # ── Forces of nature ──────────────────────────────────────────────────────
    "gravity", "electricity", "magnetism", "radiation", "nuclear explosion",
    "earthquake", "tsunami", "tornado", "hurricane", "volcano", "avalanche",
    "erosion", "rust", "decay", "entropy",

    # ── Cosmic scale ──────────────────────────────────────────────────────────
    "black hole", "supernova", "neutron star", "dark matter", "dark energy",
    "universe", "galaxy", "sun", "moon", "comet", "asteroid", "gamma ray burst",
    "heat death of the universe", "big bang",

    # ── Subatomic / physics ───────────────────────────────────────────────────
    "atom", "electron", "proton", "quark", "photon", "neutrino",
    "quantum mechanics", "nuclear fission", "antimatter",

    # ── Chemistry ─────────────────────────────────────────────────────────────
    "acid", "base", "oxidation", "rust", "corrosion", "solvent",
    "hydrofluoric acid", "thermite",

    # ── Materials ─────────────────────────────────────────────────────────────
    "diamond", "steel", "titanium", "graphene", "obsidian", "ceramic",
    "rubber", "wood", "plastic", "concrete", "rope", "chain",

    # ── Tools & weapons ───────────────────────────────────────────────────────
    "hammer", "drill", "saw", "laser", "gun", "sword", "bomb", "dynamite",
    "missile", "tank", "nuclear bomb", "jackhammer",

    # ── Biology / life ────────────────────────────────────────────────────────
    "bacteria", "virus", "fungus", "mold", "tree", "roots", "moss",
    "termites", "elephant", "whale", "human", "evolution",

    # ── Abstract concepts ─────────────────────────────────────────────────────
    "time", "infinity", "nothingness", "void", "chaos", "order",
    "love", "hate", "fear", "hope", "logic", "creativity", "knowledge",
    "ignorance", "truth", "lies", "paradox", "mathematics",

    # ── Human constructs ──────────────────────────────────────────────────────
    "science", "religion", "government", "law", "money", "capitalism",
    "democracy", "censorship", "internet", "social media",
    "artificial intelligence", "computer", "algorithm",

    # ── Mythological / pop-culture ────────────────────────────────────────────
    "god", "death", "dragon", "magic", "wizard", "curse",
    "kryptonite", "one ring", "thanos snap",

    # ── Processes ─────────────────────────────────────────────────────────────
    "melting", "evaporation", "freezing", "explosion", "implosion",
    "compression", "pressure", "heat", "cold", "vacuum",

    # ── Information / meta ────────────────────────────────────────────────────
    "nothing", "everything", "itself", "the concept of beating things",
]

# Deduplicate while preserving order
_seen = set()
SEED_VOCABULARY = [
    x for x in SEED_VOCABULARY if not (x in _seen or _seen.add(x))
]


def load_vocabulary(path: str) -> List[str]:
    """Load vocab from disk, seeding with SEED_VOCABULARY if file absent."""
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        vocab = data.get("vocabulary", [])
        # Always ensure seed items are present
        existing = set(vocab)
        for item in SEED_VOCABULARY:
            if item not in existing:
                vocab.append(item)
        return vocab
    return list(SEED_VOCABULARY)


def save_vocabulary(vocab: List[str], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"vocabulary": vocab}, f, indent=2)


def add_item(vocab: List[str], item: str, path: str) -> List[str]:
    """Add a new winner to the vocab and persist."""
    item = item.strip().lower()
    if item and item not in vocab:
        vocab.append(item)
        save_vocabulary(vocab, path)
    return vocab
