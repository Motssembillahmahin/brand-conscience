# Brand Conscience — API Keys

## Required API Keys

### 1. Meta Marketing API

**Variables**: `META_APP_ID`, `META_APP_SECRET`, `META_ACCESS_TOKEN`, `META_AD_ACCOUNT_ID`

**How to obtain**:
1. Create a Meta Developer account at developers.facebook.com
2. Create a new app (type: Business)
3. Add the Marketing API product
4. Generate a long-lived access token with `ads_management` and `ads_read` permissions
5. Find your ad account ID in Meta Business Suite

**Permissions needed**: `ads_management`, `ads_read`, `business_management`

### 2. Google Gemini API

**Variable**: `GEMINI_API_KEY`

**How to obtain**:
1. Go to Google AI Studio (aistudio.google.com)
2. Create an API key
3. Enable the Gemini API in your Google Cloud project

**Usage**: Image generation via Gemini's multimodal capabilities

### 3. Slack API

**Variable**: `SLACK_BOT_TOKEN`

**How to obtain**:
1. Create a Slack app at api.slack.com
2. Add bot scopes: `chat:write`, `channels:read`
3. Install the app to your workspace
4. Copy the Bot User OAuth Token

**Usage**: Notifications for circuit breaker events, approvals, daily summaries

### 4. OPIK

**Variable**: `OPIK_API_KEY` (optional — not needed for self-hosted)

**How to obtain**:
- For cloud: Sign up at comet.com/opik and generate an API key
- For self-hosted: No key needed, configure `OPIK_URL` to point to local instance

**Usage**: LLM and decision tracing

### 5. Database

**Variable**: `DATABASE_URL`

**Format**: `postgresql+psycopg://user:password@host:port/dbname`

**Default (Docker Compose)**: `postgresql+psycopg://brand_conscience:brand_conscience@localhost:5432/brand_conscience`

### 6. Redis

**Variable**: `REDIS_URL`

**Format**: `redis://host:port/db`

**Default (Docker Compose)**: `redis://localhost:6379/0`

## .env.example

All variables are documented in `.env.example` at the project root. Copy it to `.env` and fill in your values:

```bash
cp .env.example .env
```

## Security Notes

- Never commit `.env` files to git
- Use different credentials for development and production
- Meta access tokens expire — set up token refresh automation for production
- Store production secrets in a secrets manager (AWS Secrets Manager, Vault, etc.)
