# Feedback Lens

ML pipeline for ingesting, labeling, and analyzing product reviews.

## Run locally with Docker

With Docker Desktop running, start PostgreSQL (with pgvector), the FastAPI API,
and the React demo together:

```powershell
docker compose up --build
```

Open the demo at http://localhost:5173. The API is available at
http://localhost:8000 and its interactive documentation is at
http://localhost:8000/docs.

The API container connects to the Compose Postgres service automatically. The
live `/classify` demo uses the local LoRA adapter mounted from
`finetuning/checkpoints/`; run the fine-tuning step first if that directory is
empty. The base model downloads on its first classification request and is
cached in the named `hf-cache` Docker volume.

Stop the stack with:

```powershell
docker compose down
```
