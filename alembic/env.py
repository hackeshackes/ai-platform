from logging.config import fileConfig
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from alembic import context
from backend.models import Base
from backend.core.config import db_config

config = context.config
config.set_main_option("sqlalchemy.url", db_config.url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

def run_migrations():
    with context.begin_transaction():
        context.run_migrations()

if context.is_offline_mode():
    with context.begin_transaction():
        context.run_migrations()
else:
    run_migrations()
