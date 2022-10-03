#!/bin/bash

export PYTHONUSERBASE="/home/ftuser/.local"

# SQLAlchemy support
export LIVE_DATABASE_URL=${LIVE_DATABASE_URL//postgres/postgresql}
export DRY_DATABASE_URL=${DRY_DATABASE_URL//postgres/postgresql}

envsubst < user_data/config.live-futures.json > live-futures.json
envsubst < user_data/config.live.json > live.json
envsubst < user_data/config.live-1m.json > live-1m.json
envsubst < user_data/config.dry.json > dry.json
envsubst < user_data/config.test.json > test.json
envsubst < user_data/config.hyper.json > hyper.json

echo "Env vars loaded into config"
exec "$@"
