-- Add migration script here
CREATE TABLE llm_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    prompt TEXT,
    response TEXT,
    llm_type TEXT,
    temperature FLOAT,
    max_tokens INTEGER,
    event_type TEXT
);