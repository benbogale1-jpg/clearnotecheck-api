"""
ClearNoteCheck - Cloud API Server
Lightweight Flask server for OpenAI-powered features (summaries, chat, transcription)

This is the CLOUD version - uses OpenAI Whisper API for transcription (no heavy local models).

Endpoints:
    GET  /health                      - Health check
    POST /transcribe                  - Transcribe audio using OpenAI Whisper
    POST /transcribe-with-diarization - Transcribe with basic speaker diarization
    POST /summarize                   - Generate AI summary from transcript
    POST /executive-summary           - Generate Manager/Executive summary
    POST /chat                        - Chat with transcript

Environment variables:
    OPENAI_API_KEY - Required: OpenAI API key for GPT and Whisper
"""

import os
import json
import tempfile
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
        'version': '2.0.0',
        'status': 'running',
        'endpoints': [
            'GET  /health',
            'POST /transcribe',
            'POST /transcribe-with-diarization',
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
        'whisper': 'available' if openai_client else 'not available',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Transcribe audio file using OpenAI Whisper API

    Request: multipart/form-data with 'audio' file
    Response: JSON with transcription segments
    """
    try:
        if not openai_client:
            return jsonify({
                'success': False,
                'error': 'OpenAI API not configured'
            }), 503

        if 'audio' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No audio file provided'
            }), 400

        audio_file = request.files['audio']

        # Save to temp file (OpenAI API needs a file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp:
            audio_file.save(temp.name)
            temp_path = temp.name

        print(f"[API Server] Transcribing with OpenAI Whisper: {audio_file.filename}")

        # Use OpenAI Whisper API with timestamps
        with open(temp_path, 'rb') as f:
            response = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )

        # Clean up temp file
        os.unlink(temp_path)

        # Extract segments from response
        segments = []
        if hasattr(response, 'segments') and response.segments:
            for seg in response.segments:
                segments.append({
                    'text': seg.text.strip(),
                    'startTime': seg.start,
                    'endTime': seg.end,
                    'confidence': 0.95
                })
        else:
            # Fallback: single segment
            segments.append({
                'text': response.text.strip(),
                'startTime': 0,
                'endTime': 0,
                'confidence': 0.95
            })

        print(f"[API Server] Transcription complete: {len(segments)} segments")

        return jsonify({
            'success': True,
            'transcription': response.text.strip(),
            'segments': segments,
            'language': getattr(response, 'language', 'en')
        })

    except Exception as e:
        print(f"[API Server] Transcription error: {str(e)}")
        # Clean up temp file if it exists
        try:
            os.unlink(temp_path)
        except:
            pass
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/transcribe-with-diarization', methods=['POST'])
def transcribe_with_diarization():
    """
    Transcribe with AI-powered speaker diarization using OpenAI Whisper + GPT

    Uses Whisper for transcription, then GPT to identify speaker changes
    based on conversation context, questions/answers, and dialogue patterns.
    """
    try:
        if not openai_client:
            return jsonify({
                'success': False,
                'error': 'OpenAI API not configured'
            }), 503

        if 'audio' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No audio file provided'
            }), 400

        audio_file = request.files['audio']

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp:
            audio_file.save(temp.name)
            temp_path = temp.name

        print(f"[API Server] Transcribing with diarization: {audio_file.filename}")

        # Use OpenAI Whisper API
        with open(temp_path, 'rb') as f:
            response = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )

        # Clean up temp file
        os.unlink(temp_path)

        # Extract raw segments
        raw_segments = []
        if hasattr(response, 'segments') and response.segments:
            for seg in response.segments:
                raw_segments.append({
                    'text': seg.text.strip(),
                    'startTime': seg.start,
                    'endTime': seg.end
                })
        else:
            raw_segments.append({
                'text': response.text.strip(),
                'startTime': 0,
                'endTime': 0
            })

        # Use GPT to identify speakers
        print(f"[API Server] Using GPT to identify speakers in {len(raw_segments)} segments...")

        # Build numbered segment list for GPT
        segment_list = "\n".join([f"{i}: {seg['text']}" for i, seg in enumerate(raw_segments)])

        gpt_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a speaker diarization assistant. Given a list of transcript segments, identify which speaker said each segment.

Analyze the conversation for:
- Questions vs answers (different speakers)
- "I" vs "you" references
- Topic/perspective changes
- Conversational turn-taking patterns
- Greetings and responses

Output ONLY a JSON array of speaker numbers (0, 1, 2, etc.) corresponding to each segment index.
Example: [0, 1, 0, 1, 2, 0] means segment 0 is speaker 0, segment 1 is speaker 1, etc.

Use the minimum number of speakers that makes sense for the conversation.
If it's clearly one person (monologue, no dialogue patterns), use [0, 0, 0, ...]."""
                },
                {
                    "role": "user",
                    "content": f"Identify speakers for each segment:\n\n{segment_list}"
                }
            ],
            temperature=0.1,
            max_tokens=1024
        )

        # Parse GPT response
        speaker_text = gpt_response.choices[0].message.content.strip()
        try:
            # Remove markdown if present
            if speaker_text.startswith('```'):
                speaker_text = speaker_text.split('```')[1]
                if speaker_text.startswith('json'):
                    speaker_text = speaker_text[4:]
            speaker_text = speaker_text.strip()
            speaker_assignments = json.loads(speaker_text)
        except json.JSONDecodeError:
            print(f"[API Server] Could not parse GPT speaker response, using fallback")
            # Fallback to pause-based
            speaker_assignments = []
            current_speaker = 0
            last_end = 0
            for seg in raw_segments:
                if seg['startTime'] - last_end > 1.5:
                    current_speaker = (current_speaker + 1) % 4
                speaker_assignments.append(current_speaker)
                last_end = seg['endTime']

        # Ensure we have assignments for all segments
        while len(speaker_assignments) < len(raw_segments):
            speaker_assignments.append(0)

        # Build final segments with speaker info
        segments = []
        diarization = []
        for i, seg in enumerate(raw_segments):
            speaker = speaker_assignments[i] if i < len(speaker_assignments) else 0
            segments.append({
                'text': seg['text'],
                'startTime': seg['startTime'],
                'endTime': seg['endTime'],
                'confidence': 0.95,
                'speaker': speaker
            })
            diarization.append({
                'speaker': speaker,
                'startTime': seg['startTime'],
                'endTime': seg['endTime']
            })

        num_speakers = len(set(seg['speaker'] for seg in segments))
        print(f"[API Server] Transcription complete: {len(segments)} segments, {num_speakers} speakers (GPT diarization)")

        return jsonify({
            'success': True,
            'transcription': response.text.strip(),
            'segments': segments,
            'diarization': diarization,
            'language': getattr(response, 'language', 'en'),
            'numSpeakers': num_speakers
        })

    except Exception as e:
        print(f"[API Server] Transcription error: {str(e)}")
        try:
            os.unlink(temp_path)
        except:
            pass
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


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
    print("ClearNoteCheck Cloud API Server v2.0")
    print("="*50)
    print(f"OpenAI: {'Configured' if openai_client else 'NOT CONFIGURED'}")
    print(f"Whisper: {'Available (OpenAI API)' if openai_client else 'NOT AVAILABLE'}")
    print("Endpoints:")
    print("  GET  /              - API info")
    print("  GET  /health        - Health check")
    print("  POST /transcribe    - Transcribe audio (Whisper)")
    print("  POST /transcribe-with-diarization - Transcribe + speakers")
    print("  POST /summarize     - Generate AI summary")
    print("  POST /executive-summary - Executive summary")
    print("  POST /chat          - Chat with transcript")
    print("="*50 + "\n")

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
