#!/usr/bin/env bash
set -euo pipefail

# Run on the EC2 host after docker compose has started. This seeds the public
# demo database from the processed review and pseudo-label files bundled with
# the deployment archive. It is idempotent.

compose_file="${COMPOSE_FILE:-docker-compose.ec2.yml}"
COMPOSE=(docker compose --env-file .env.ec2 -f "$compose_file")

"${COMPOSE[@]}" up -d
"${COMPOSE[@]}" exec -T backend python clustering/embed.py --batch-size 64
"${COMPOSE[@]}" exec -T backend python backend/migrations/apply_migrations.py
"${COMPOSE[@]}" exec -T backend python clustering/dedupe.py --threshold 0.85
"${COMPOSE[@]}" exec -T backend python backend/scripts/import_pseudo_labels.py

echo "Demo data seeded. Open http://<instance-public-ip>/"
