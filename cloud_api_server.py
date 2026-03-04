"""
ClearNoteCheck - Cloud API Server
Lightweight Flask server for OpenAI-powered features (summaries, chat)

This is the CLOUD version - no Whisper, no heavy ML models.
Transcription is handled separately (locally or via cloud ASR service).

Endpoints:
    GET  /health              - Health check
    POST /summarize           - Generate AI summary from transcript
    POST /executive-summary   - Generate Manager/Executive summary
    POST /chat                - Chat with transcript

Environment variables:
    OPENAI_API_KEY - Required: OpenAI API key for GPT
"""

import os
import json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
openai_api_key = os.environ.get('OPENAI_API_KEY')
if not openai_api_key:
    print("[API Server] WARNING: OPENAI_API_KEY not set!")
    openai_client = None
else:
    openai_client = OpenAI(api_key=openai_api_key)
    print("[API Server] OpenAI client initialized!")


@app.route('/', methods=['GET'])
def root():
    """Root endpoint - API info"""
    return jsonify({
        'service': 'ClearNoteCheck Cloud API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': [
            'GET  /health',
            'POST /summarize',
            'POST /executive-summary', 
            'POST /chat'
        ]
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'ClearNoteCheck Cloud API',
        'openai': 'configured' if openai_client else 'not configured',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/summarize', methods=['POST'])
def summarize():
    """
    Generate AI summary from transcript using OpenAI GPT

    Request JSON: { "transcript": "Speaker 1: Hello...\nSpeaker 2: Hi..." }
    Response: { "success": true, "summary": { bulletPoints, actionItems, keyTopics, decisions } }
    """
    try:
        if not openai_client:
            return jsonify({
                'success': False,
                'error': 'OpenAI API not configured. Set OPENAI_API_KEY environment variable.'
            }), 503

        data = request.get_json()
        if not data or 'transcript' not in data:
            return jsonify({
                'success': False,
                'error': 'No transcript provided'
            }), 400

        transcript = data['transcript']
        print(f"[API Server] Generating summary ({len(transcript)} chars)")

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a meeting notes assistant. Given a transcript with speaker labels, produce a JSON response with:
1. "bulletPoints": array of 3-5 concise summary points
2. "actionItems": array of objects with "text" and "assignee" fields
3. "keyTopics": array of topic strings
4. "decisions": array of decisions made

Be concise. Use speaker names when attributing. Output ONLY valid JSON, no other text."""
                },
                {
                    "role": "user",
                    "content": f"Summarize this meeting transcript:\n\n{transcript}"
                }
            ],
            temperature=0.3,
            max_tokens=1024
        )

        summary_text = response.choices[0].message.content.strip()

        # Parse the JSON response
        try:
            if summary_text.startswith('```'):
                summary_text = summary_text.split('```')[1]
                if summary_text.startswith('json'):
                    summary_text = summary_text[4:]
            summary_text = summary_text.strip()
            summary = json.loads(summary_text)
        except json.JSONDecodeError:
            summary = {
                'bulletPoints': [summary_text[:200]],
                'actionItems': [],
                'keyTopics': [],
                'decisions': []
            }

        print(f"[API Server] Summary generated: {len(summary.get('bulletPoints', []))} bullet points")

        return jsonify({
            'success': True,
            'summary': summary
        })

    except Exception as e:
        print(f"[API Server] Summarization error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/executive-summary', methods=['POST'])
def executive_summary():
    """
    Generate Executive/Manager Summary from transcript using OpenAI GPT
    """
    try:
        if not openai_client:
            return jsonify({
                'success': False,
                'error': 'OpenAI API not configured. Set OPENAI_API_KEY environment variable.'
            }), 503

        data = request.get_json()
        if not data or 'transcript' not in data:
            return jsonify({
                'success': False,
                'error': 'No transcript provided'
            }), 400

        transcript = data['transcript']
        meeting_title = data.get('meetingTitle', 'Meeting')
        meeting_date = data.get('meetingDate', '')

        print(f"[API Server] Generating executive summary for: {meeting_title}")

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are an executive assistant preparing meeting summaries for busy managers.
Given a meeting transcript, produce a JSON response optimized for quick executive review:

{
    "executiveBrief": "2-3 sentence high-level summary of what was discussed and outcomes",
    "keyDecisions": [
        {"decision": "What was decided", "impact": "Brief impact/importance", "owner": "Who owns this"}
    ],
    "actionItems": [
        {"task": "What needs to be done", "assignee": "Who", "deadline": "When (if mentioned)", "priority": "high/medium/low"}
    ],
    "risksAndBlockers": [
        {"issue": "What's the risk/blocker", "severity": "high/medium/low", "mitigation": "Suggested action"}
    ],
    "nextSteps": ["Immediate next step 1", "Next step 2"],
    "followUpMeetings": [
        {"title": "Suggested meeting title", "purpose": "Why needed", "suggestedAttendees": ["Person 1"], "suggestedTimeframe": "This week/Next week/etc"}
    ],
    "keyMetrics": [
        {"metric": "Any numbers/KPIs mentioned", "value": "The value", "trend": "up/down/stable"}
    ]
}

Be concise and action-oriented. Focus on what matters to decision-makers.
If a field has no relevant content, use an empty array.
Output ONLY valid JSON, no other text."""
                },
                {
                    "role": "user",
                    "content": f"Meeting: {meeting_title}\nDate: {meeting_date}\n\nTranscript:\n{transcript}"
                }
            ],
            temperature=0.3,
            max_tokens=2048
        )

        summary_text = response.choices[0].message.content.strip()

        try:
            if summary_text.startswith('```'):
                summary_text = summary_text.split('```')[1]
                if summary_text.startswith('json'):
                    summary_text = summary_text[4:]
            summary_text = summary_text.strip()
            executive_summary = json.loads(summary_text)
        except json.JSONDecodeError:
            executive_summary = {
                'executiveBrief': summary_text[:500],
                'keyDecisions': [],
                'actionItems': [],
                'risksAndBlockers': [],
                'nextSteps': [],
                'followUpMeetings': [],
                'keyMetrics': []
            }

        print(f"[API Server] Executive summary generated")

        return jsonify({
            'success': True,
            'executiveSummary': executive_summary
        })

    except Exception as e:
        print(f"[API Server] Executive summary error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/chat', methods=['POST'])
def chat():
    """
    Chat with the transcript using OpenAI GPT
    """
    try:
        if not openai_client:
            return jsonify({
                'success': False,
                'error': 'OpenAI API not configured. Set OPENAI_API_KEY environment variable.'
            }), 503

        data = request.get_json()
        if not data or 'transcript' not in data or 'question' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing transcript or question'
            }), 400

        transcript = data['transcript']
        question = data['question']

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI assistant helping analyze a meeting transcript. Answer questions based on the transcript provided. Be concise and helpful."
                },
                {
                    "role": "user",
                    "content": f"Meeting Transcript:\n{transcript}\n\nQuestion: {question}"
                }
            ],
            temperature=0.3,
            max_tokens=512
        )

        answer = response.choices[0].message.content.strip()

        return jsonify({
            'success': True,
            'answer': answer
        })

    except Exception as e:
        print(f"[API Server] Chat error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("\n" + "="*50)
    print("ClearNoteCheck Cloud API Server")
    print("="*50)
    print(f"OpenAI: {'Configured' if openai_client else 'NOT CONFIGURED'}")
    print("Endpoints:")
    print("  GET  /              - API info")
    print("  GET  /health        - Health check")
    print("  POST /summarize     - Generate AI summary")
    print("  POST /executive-summary - Executive summary")
    print("  POST /chat          - Chat with transcript")
    print("="*50 + "\n")

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
