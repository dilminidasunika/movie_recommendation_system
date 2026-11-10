# movie_recommendation_system

## Security & Deployment

### Local Development vs. Production

#### Local Development
```bash
# Starts database and backend; docker-compose.override.yml binds backend to 127.0.0.1:8000
docker compose up -d

# Run frontend with Vite dev server (proxies /api to http://localhost:8000)
cd frontend
npm run dev
```

#### Production Deployment
```bash
# 1. Build frontend distribution assets
cd frontend
npm run build
cd ..

# 2. Run compose in production mode (bypasses docker-compose.override.yml, launches Caddy proxy)
docker compose -f docker-compose.yml --profile prod up -d
```
In production, port 8000 is never exposed to the host network (`expose: ["8000"]`). Instead, Caddy listens on ports 80 and 443, serves the static frontend assets from `./frontend/dist`, applies security headers, and reverse proxies `/api/*` requests to the backend service.

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `APP_ENV` | Application environment (`dev` or `prod`). In `prod`, API documentation (`/docs`, `/openapi.json`) is disabled and `TrustedHostMiddleware` is activated. | `dev` |
| `CORS_ORIGINS` | JSON list of allowed origins for CORS. Explicit origins only; never set to `"*"`. | `["http://localhost:5173"]` |
| `ALLOWED_HOSTS` | JSON list of valid `Host` header values enforced in production mode. | `["localhost","127.0.0.1"]` |
| `FORWARDED_ALLOW_IPS` | IP/CIDR subnet of the trusted reverse proxy for `X-Forwarded-*` headers read by Uvicorn. In production, set to the proxy's IP/subnet; never set to `*` while port 8000 is reachable. | `127.0.0.1` |
| `SITE_ADDRESS` | Domain or IP address used by Caddy in `deploy/Caddyfile`. | `localhost` |

### Read-Only Database Role (`app_ro`)

For least-privilege security, create a dedicated read-only role using `database/create_app_role.sql`:
```bash
# Run once as the postgres superuser:
docker compose exec -i database psql -U $POSTGRES_USER -d $POSTGRES_DB < database/create_app_role.sql
```
Once created, configure `DATABASE_URL` to authenticate with the `app_ro` credentials.

### Architectural & Security Constraints
- **Never expose Postgres or port 8000 publicly**: The Postgres port (5432) has been removed from host port mappings. Use `docker compose exec database psql` for database access. Port 8000 is restricted to localhost in dev via `docker-compose.override.yml` and is unexposed to the host in production.
- **In-memory limits require a single worker**: Embedded ChromaDB, concurrency semaphores (`PIPELINE_SEM`, `LLM_SEM`), and in-memory rate limiting depend on a single process (`uvicorn --workers 1`). Do not configure `--workers > 1`.
