# AWS EC2 GPU demo deployment

This deployment intentionally uses one GPU EC2 instance and Docker Compose.
It does not create RDS, an Elastic IP, a load balancer, a NAT gateway, or any
other always-on AWS resource.

## Before launching

- Configure the `feedback-lens-deployer` AWS CLI profile and obtain the EC2 G/VT quota.
- Create a cost budget and keep the instance stopped between demos.
- Run `powershell -ExecutionPolicy Bypass -File scripts/package_ec2_source.ps1`.
- Do not upload `.env`; the server creates `.env.ec2` with a unique database password.

## Deployment design

- `docker-compose.ec2.yml` exposes only port 80. PostgreSQL and FastAPI stay on
  Docker's private network.
- The production frontend proxies browser API requests from `/api/*` to FastAPI.
- `scripts/seed_ec2_demo_data.sh` embeds the bundled review corpus, imports
  pseudo-labels, and builds pgvector clusters after the first boot.
- No Elastic IP is used. The public IP changes after a stop/start cycle.

## CPU fallback deployment

If the GPU quota is unavailable, use `docker-compose.ec2.cpu.yml` on a
`t3.xlarge` (4 vCPUs, 16 GiB RAM). It runs the same FastAPI, PostgreSQL,
embedding, clustering, and LoRA inference flow without `gpus: all`.

CPU classification is materially slower than the GPU version, so it is suited
to small, occasional demonstrations rather than concurrent traffic. Seed it
with:

```bash
COMPOSE_FILE=docker-compose.ec2.cpu.yml ./scripts/seed_ec2_demo_data.sh
```

## EC2 requirements

- Region: `us-west-2`
- GPU instance: `g4dn.xlarge` (one NVIDIA T4 GPU, 16 GiB VRAM)
- CPU fallback: `t3.xlarge` (4 vCPUs, 16 GiB RAM)
- Root disk: 60 GiB gp3
- Ubuntu or an AWS Deep Learning Base GPU AMI
- Security group: SSH (22) and HTTP (80) from your current public IP only

## Stop versus terminate

Use **Stop** after a near-term demo: the database and Hugging Face cache remain
on the EBS disk, but EBS storage continues to cost a small amount.

Use **Terminate** and select **Delete attached EBS volumes** when you do not
expect to demo again soon. That removes all continuing EC2/EBS charges. The
next deployment uses the local source archive again and reseeds the database.
