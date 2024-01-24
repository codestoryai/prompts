use std::path::Path;

use sqlx::SqlitePool;

use crate::{clients::types::LLMClientError, config::LLMBrokerConfiguration};

pub async fn init(config: LLMBrokerConfiguration) -> Result<SqlitePool, LLMClientError> {
    let data_dir = config.data_dir.to_string_lossy().to_owned();

    match connect(&data_dir).await {
        Ok(pool) => Ok(pool),
        Err(_) => {
            reset(&data_dir)?;
            connect(&data_dir).await
        }
    }
}

async fn connect(data_dir: &str) -> Result<SqlitePool, LLMClientError> {
    let url = format!("sqlite://{data_dir}/llm_data.data?mode=rwc");
    let pool = SqlitePool::connect(&url)
        .await
        .map_err(|_| LLMClientError::TokioMpscSendError)?;

    if let Err(e) = sqlx::migrate!().run(&pool).await {
        // We manually close the pool here to ensure file handles are properly cleaned up on
        // Windows.
        pool.close().await;
        Err(e).map_err(|_e| LLMClientError::SqliteSetupError)?
    } else {
        Ok(pool)
    }
}

fn reset(data_dir: &str) -> Result<(), LLMClientError> {
    let db_path = Path::new(data_dir).join("llm_data.data");
    let bk_path = db_path.with_extension("llm_data.bk");
    std::fs::rename(db_path, bk_path).map_err(|_| LLMClientError::SqliteSetupError)
}
