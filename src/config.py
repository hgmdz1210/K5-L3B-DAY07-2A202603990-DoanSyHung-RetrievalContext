"""Cấu hình chung, đọc từ .env ở thư mục gốc repo. Không có giá trị giả/mock."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / '.env')

DATA_DIR = ROOT / 'data'
RESULTS_DIR = ROOT / 'results'

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab')
DAY6_STRATEGY = os.getenv('DAY6_STRATEGY', 'B')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
EMBEDDING_DIM = {'text-embedding-3-small': 1536, 'text-embedding-3-large': 3072}[EMBEDDING_MODEL]
AGENT_MODEL = os.getenv('AGENT_MODEL', 'gpt-5.4-mini-2026-03-17')

# Giá USD / 1M token để ước tính chi phí. Kiểm tra lại trên trang giá OpenAI trước khi báo cáo;
# có thể ghi đè bằng biến môi trường.
EMBEDDING_PRICE_PER_1M = float(os.getenv('EMBEDDING_PRICE_PER_1M', '0.02'))
AGENT_PRICE_IN_PER_1M = float(os.getenv('AGENT_PRICE_IN_PER_1M', '0.75'))
AGENT_PRICE_OUT_PER_1M = float(os.getenv('AGENT_PRICE_OUT_PER_1M', '4.5'))


def require_api_key() -> str:
    key = os.getenv('OPENAI_API_KEY', '').strip()
    if not key:
        raise RuntimeError(f'OPENAI_API_KEY trống. Điền key vào {ROOT / ".env"} (xem .env.example).')
    return key
