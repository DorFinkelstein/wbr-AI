from dataclasses import dataclass, field


@dataclass
class Config:
    # ── WBR API ──────────────────────────────────────────────────────────────
    # Find the real endpoint by opening whatbeatsrock.com in Chrome/Firefox,
    # opening DevTools → Network tab, playing one round, and copying the
    # POST request URL + body format.  Common patterns:
    #   POST /api/wbr          body: {"item1": "rock", "item2": "paper"}
    #   POST /api/guess        body: {"current": "rock", "guess": "paper"}
    api_base_url: str = "https://whatbeatsrock.com"
    api_path: str = "/api/wbr"
    # Field names in the request body (update after checking DevTools)
    api_field_item1: str = "item1"
    api_field_item2: str = "item2"
    # Field in the response that says whether item2 won
    api_field_wins: str = "item2wins"
    request_delay: float = 1.5  # seconds between API calls (be polite)

    # ── Offline judge (Claude) ────────────────────────────────────────────────
    # Used for pre-training before hitting the real API.
    # Requires ANTHROPIC_API_KEY env var.
    use_claude_judge: bool = True
    claude_model: str = "claude-sonnet-4-6"
    claude_judge_temperature: float = 0.2

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"  # 384-dim, fast, good quality
    embedding_dim: int = 384

    # ── Q-Network ─────────────────────────────────────────────────────────────
    hidden_dim: int = 256
    learning_rate: float = 1e-3
    gamma: float = 0.95          # discount factor for future chain length
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.997
    batch_size: int = 32
    replay_buffer_size: int = 20_000
    target_update_freq: int = 50  # steps between copying online→target network

    # ── Training ──────────────────────────────────────────────────────────────
    offline_episodes: int = 300    # pre-train with Claude judge
    online_episodes: int = 200     # fine-tune on real game API
    max_chain_length: int = 100    # safety cap per episode
    save_freq: int = 25            # save checkpoint every N episodes

    # ── Paths ─────────────────────────────────────────────────────────────────
    model_dir: str = "models"
    model_path: str = "models/agent.pt"
    db_path: str = "data/knowledge.db"
    vocab_path: str = "data/vocabulary.json"

    # ── Candidate generation ──────────────────────────────────────────────────
    # Number of vocabulary items scored by Q-network each step
    top_k_candidates: int = 30
    # Expand vocab with new winners discovered during play
    auto_expand_vocab: bool = True
