from dataclasses import dataclass, field


@dataclass
class Config:
    # ── WBR API ──────────────────────────────────────────────────────────────
    api_base_url: str = "https://whatbeatsrock.com"
    api_path: str = "/api/wbr"          # verify path in DevTools if it changes
    # Request body fields
    api_field_prev: str = "prev"         # current item to beat
    api_field_guess: str = "guess"       # our proposed item
    api_field_gid: str = "gid"           # game session UUID (client-generated)
    # Response: wins flag lives at response["data"]["guess_wins"]
    api_response_data_key: str = "data"
    api_field_wins: str = "guess_wins"
    request_delay: float = 1.5          # seconds between API calls (be polite)

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
