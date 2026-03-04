# ClearNoteCheck Cloud API

Lightweight Flask API for ClearNoteCheck meeting summaries.

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| POST | `/summarize` | Generate meeting summary |
| POST | `/executive-summary` | Generate executive summary |
| POST | `/chat` | Chat with transcript |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for GPT-4o-mini |
| `PORT` | No | Server port (default: 5000) |

## Deploy to Railway

1. Push this folder to GitHub
2. Connect Railway to your GitHub repo
3. Add `OPENAI_API_KEY` in Railway dashboard → Variables
4. Deploy!

## Local Development

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your-key-here
python cloud_api_server.py
```

## API Usage

### Summarize
```bash
curl -X POST https://your-app.railway.app/summarize \
  -H "Content-Type: application/json" \
  -d '{"transcript": "Speaker 1: Hello everyone..."}'
```

### Executive Summary
```bash
curl -X POST https://your-app.railway.app/executive-summary \
  -H "Content-Type: application/json" \
  -d '{"transcript": "...", "meetingTitle": "Weekly Standup"}'
```
