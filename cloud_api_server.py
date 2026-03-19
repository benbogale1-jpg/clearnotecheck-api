"""
ClearNoteCheck / ClearStudyCheck - Cloud API Server
Lightweight Flask server for OpenAI-powered features (summaries, chat, study guides, transcription)

This is the CLOUD version - uses AssemblyAI for transcription with real speaker diarization.
Supports 2+ hour recordings via chunked upload.

Endpoints:
    GET  /health                      - Health check
    POST /transcribe                  - Transcribe audio using AssemblyAI
    POST /transcribe-with-diarization - Transcribe with real voice-based speaker diarization
    POST /upload/init                 - Initialize chunked upload session
    POST /upload/chunk                - Upload a chunk
    POST /upload/complete             - Complete upload and transcribe
    POST /summarize                   - Generate AI summary from transcript
    POST /executive-summary           - Generate Manager/Executive summary (ClearNoteCheck)
    POST /study-guide                 - Generate student study guide (ClearStudyCheck)
    POST /chat                        - Chat with transcript

Environment variables:
    OPENAI_API_KEY - Required: OpenAI API key for GPT
    ASSEMBLYAI_API_KEY - Required: AssemblyAI API key for transcription + diarization
"""

import os
import json
import time
import tempfile
import requests
import uuid
import shutil
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# Chunked upload storage
UPLOAD_SESSIONS = {}  # session_id -> {chunks: [], created_at, total_chunks}
CHUNK_DIR = tempfile.mkdtemp(prefix='clearnote_chunks_')

# Initialize OpenAI client
openai_api_key = os.environ.get('OPENAI_API_KEY')
if not openai_api_key:
    print("[API Server] WARNING: OPENAI_API_KEY not set!")
    openai_client = None
else:
    openai_client = OpenAI(api_key=openai_api_key)
    print("[API Server] OpenAI client initialized!")

# Initialize AssemblyAI
assemblyai_api_key = os.environ.get('ASSEMBLYAI_API_KEY')
if not assemblyai_api_key:
    print("[API Server] WARNING: ASSEMBLYAI_API_KEY not set - diarization will be limited!")
else:
    print("[API Server] AssemblyAI configured for speaker diarization!")


@app.route('/', methods=['GET'])
def root():
    """Root endpoint - API info"""
    return jsonify({
        'service': 'ClearNoteCheck Cloud API',
        'version': '3.0.0',
        'status': 'running',
        'diarization': 'real voice-based (AssemblyAI)' if assemblyai_api_key else 'not configured',
        'endpoints': [
            'GET  /health',
            'POST /transcribe',
            'POST /transcribe-with-diarization',
            'POST /summarize',
            'POST /executive-summary',
            'POST /study-guide',
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
        'assemblyai': 'configured' if assemblyai_api_key else 'not configured',
        'assemblyai_key_prefix': assemblyai_api_key[:10] + '...' if assemblyai_api_key else 'not set',
        'diarization': 'real voice-based' if assemblyai_api_key else 'text-based fallback',
        'timestamp': datetime.now().isoformat()
    })


# ============================================
# Chunked Upload Endpoints (for 2+ hour recordings)
# ============================================

@app.route('/upload/init', methods=['POST'])
def upload_init():
    """
    Initialize a chunked upload session.
    Returns a session_id to use for uploading chunks.
    """
    try:
        data = request.get_json() or {}
        total_chunks = data.get('total_chunks', 0)
        file_extension = data.get('extension', 'm4a')

        session_id = str(uuid.uuid4())
        session_dir = os.path.join(CHUNK_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)

        UPLOAD_SESSIONS[session_id] = {
            'chunks_received': set(),
            'total_chunks': total_chunks,
            'extension': file_extension,
            'created_at': time.time(),
            'dir': session_dir
        }

        print(f"[Chunked Upload] Session created: {session_id}, expecting {total_chunks} chunks")

        return jsonify({
            'success': True,
            'session_id': session_id
        })
    except Exception as e:
        print(f"[Chunked Upload] Init error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/upload/chunk', methods=['POST'])
def upload_chunk():
    """
    Upload a single chunk.
    Expects: session_id, chunk_index, chunk data (multipart)
    """
    try:
        session_id = request.form.get('session_id')
        chunk_index = int(request.form.get('chunk_index', 0))

        if session_id not in UPLOAD_SESSIONS:
            return jsonify({'success': False, 'error': 'Invalid session_id'}), 400

        session = UPLOAD_SESSIONS[session_id]

        if 'chunk' not in request.files:
            return jsonify({'success': False, 'error': 'No chunk data'}), 400

        chunk_file = request.files['chunk']
        chunk_path = os.path.join(session['dir'], f'chunk_{chunk_index:04d}')
        chunk_file.save(chunk_path)

        session['chunks_received'].add(chunk_index)

        print(f"[Chunked Upload] Session {session_id[:8]}: chunk {chunk_index + 1}/{session['total_chunks']} received")

        return jsonify({
            'success': True,
            'chunks_received': len(session['chunks_received']),
            'total_chunks': session['total_chunks']
        })
    except Exception as e:
        print(f"[Chunked Upload] Chunk error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/upload/complete', methods=['POST'])
def upload_complete():
    """
    Finalize upload, combine chunks, and start transcription.
    """
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id')
        speaker_labels = data.get('speaker_labels', True)

        if session_id not in UPLOAD_SESSIONS:
            return jsonify({'success': False, 'error': 'Invalid session_id'}), 400

        session = UPLOAD_SESSIONS[session_id]

        # Check all chunks received
        if len(session['chunks_received']) < session['total_chunks']:
            return jsonify({
                'success': False,
                'error': f"Missing chunks: got {len(session['chunks_received'])}/{session['total_chunks']}"
            }), 400

        print(f"[Chunked Upload] Session {session_id[:8]}: combining {session['total_chunks']} chunks...")

        # Combine chunks into single file
        ext = session.get('extension', 'm4a')
        combined_path = os.path.join(session['dir'], f'combined.{ext}')

        with open(combined_path, 'wb') as outfile:
            for i in range(session['total_chunks']):
                chunk_path = os.path.join(session['dir'], f'chunk_{i:04d}')
                with open(chunk_path, 'rb') as chunk_file:
                    outfile.write(chunk_file.read())

        file_size = os.path.getsize(combined_path)
        print(f"[Chunked Upload] Combined file size: {file_size / (1024*1024):.2f} MB")

        # Transcribe with AssemblyAI
        print("[Chunked Upload] Starting transcription...")
        result = transcribe_with_assemblyai(combined_path, speaker_labels)

        # Clean up
        shutil.rmtree(session['dir'], ignore_errors=True)
        del UPLOAD_SESSIONS[session_id]

        # Process result same as regular endpoint
        segments = []
        diarization = []

        if 'utterances' in result and result['utterances']:
            for utt in result['utterances']:
                segments.append({
                    'text': utt['text'],
                    'start': utt['start'] / 1000,
                    'end': utt['end'] / 1000,
                    'speaker': utt.get('speaker', 'A')
                })
                diarization.append({
                    'speaker': utt.get('speaker', 'A'),
                    'start': utt['start'] / 1000,
                    'end': utt['end'] / 1000
                })
        elif 'words' in result and result['words']:
            current_segment = {'text': '', 'start': None, 'end': None, 'speaker': 'A'}
            for word in result['words']:
                if current_segment['start'] is None:
                    current_segment['start'] = word['start'] / 1000
                current_segment['text'] += word['text'] + ' '
                current_segment['end'] = word['end'] / 1000
                if word['text'].endswith(('.', '?', '!')):
                    segments.append({
                        'text': current_segment['text'].strip(),
                        'start': current_segment['start'],
                        'end': current_segment['end'],
                        'speaker': 'A'
                    })
                    current_segment = {'text': '', 'start': None, 'end': None, 'speaker': 'A'}
            if current_segment['text'].strip():
                segments.append({
                    'text': current_segment['text'].strip(),
                    'start': current_segment['start'],
                    'end': current_segment['end'],
                    'speaker': 'A'
                })
        else:
            segments.append({
                'text': result.get('text', ''),
                'start': 0,
                'end': result.get('audio_duration', 0),
                'speaker': 'A'
            })

        return jsonify({
            'success': True,
            'transcription': result.get('text', ''),
            'segments': segments,
            'diarization': diarization,
            'audio_duration': result.get('audio_duration', 0)
        })

    except Exception as e:
        print(f"[Chunked Upload] Complete error: {str(e)}")
        # Clean up on error
        if session_id and session_id in UPLOAD_SESSIONS:
            session = UPLOAD_SESSIONS[session_id]
            shutil.rmtree(session.get('dir', ''), ignore_errors=True)
            del UPLOAD_SESSIONS[session_id]
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/test-assemblyai', methods=['GET'])
def test_assemblyai():
    """Test AssemblyAI connection"""
    if not assemblyai_api_key:
        return jsonify({
            'success': False,
            'error': 'ASSEMBLYAI_API_KEY not configured'
        }), 503

    try:
        # Test the API key by checking account info
        response = requests.get(
            'https://api.assemblyai.com/v2/transcript',
            headers={'authorization': assemblyai_api_key},
            params={'limit': 1}
        )

        if response.status_code == 401:
            return jsonify({
                'success': False,
                'error': 'Invalid AssemblyAI API key (401 Unauthorized)',
                'key_prefix': assemblyai_api_key[:10] + '...'
            }), 401

        return jsonify({
            'success': True,
            'message': 'AssemblyAI API key is valid',
            'status_code': response.status_code,
            'key_prefix': assemblyai_api_key[:10] + '...'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def assemblyai_transcribe(audio_path, speaker_labels=False):
    """
    Transcribe audio using AssemblyAI with optional speaker diarization.
    Returns transcript with accurate timestamps and speaker labels.
    """
    headers = {
        'authorization': assemblyai_api_key,
        'content-type': 'application/json'
    }

    # Step 1: Upload the audio file
    print("[API Server] Uploading audio to AssemblyAI...")
    try:
        with open(audio_path, 'rb') as f:
            upload_response = requests.post(
                'https://api.assemblyai.com/v2/upload',
                headers={'authorization': assemblyai_api_key},
                data=f
            )

        print(f"[API Server] Upload response status: {upload_response.status_code}")
        upload_data = upload_response.json()

        if 'error' in upload_data:
            raise Exception(f"AssemblyAI upload error: {upload_data['error']}")

        if 'upload_url' not in upload_data:
            raise Exception(f"AssemblyAI upload failed: {upload_data}")

        upload_url = upload_data['upload_url']
        print(f"[API Server] Audio uploaded successfully")
    except Exception as e:
        print(f"[API Server] Upload error: {str(e)}")
        raise

    # Step 2: Request transcription
    transcript_request = {
        'audio_url': upload_url,
        'speaker_labels': speaker_labels,
        'speech_models': ['universal-3-pro'],  # Must be one of: universal-3-pro, universal-2
    }

    print(f"[API Server] Requesting transcription (speaker_labels={speaker_labels})...")
    try:
        transcript_response = requests.post(
            'https://api.assemblyai.com/v2/transcript',
            headers=headers,
            json=transcript_request
        )

        print(f"[API Server] Transcript response status: {transcript_response.status_code}")
        transcript_data = transcript_response.json()

        if 'error' in transcript_data:
            raise Exception(f"AssemblyAI transcript error: {transcript_data['error']}")

        if 'id' not in transcript_data:
            raise Exception(f"AssemblyAI transcript failed: {transcript_data}")

        transcript_id = transcript_data['id']
        print(f"[API Server] Transcription started: {transcript_id}")
    except Exception as e:
        print(f"[API Server] Transcript request error: {str(e)}")
        raise

    # Step 3: Poll for completion (with timeout)
    polling_url = f'https://api.assemblyai.com/v2/transcript/{transcript_id}'
    max_polls = 300  # Max 10 minutes (300 * 2 seconds) for longer recordings
    polls = 0

    while polls < max_polls:
        try:
            poll_response = requests.get(polling_url, headers=headers)
            poll_data = poll_response.json()
            status = poll_data.get('status', 'unknown')

            if status == 'completed':
                print("[API Server] Transcription completed!")
                return poll_data
            elif status == 'error':
                error_msg = poll_data.get('error', 'Unknown error')
                print(f"[API Server] AssemblyAI error: {error_msg}")
                raise Exception(f"AssemblyAI transcription error: {error_msg}")

            print(f"[API Server] Status: {status}, waiting... ({polls + 1}/{max_polls})")
            time.sleep(2)
            polls += 1
        except requests.exceptions.RequestException as e:
            print(f"[API Server] Polling error: {str(e)}")
            raise

    raise Exception("Transcription timed out after 10 minutes")


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Transcribe audio file using AssemblyAI

    Request: multipart/form-data with 'audio' file
    Response: JSON with transcription segments and timestamps
    """
    try:
        if not assemblyai_api_key:
            return jsonify({
                'success': False,
                'error': 'AssemblyAI API not configured'
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

        print(f"[API Server] Transcribing: {audio_file.filename}")

        # Use AssemblyAI for transcription
        result = assemblyai_transcribe(temp_path, speaker_labels=False)

        # Clean up temp file
        os.unlink(temp_path)

        # Extract segments with timestamps
        segments = []
        if 'words' in result and result['words']:
            # Group words into sentences/segments
            current_segment = {'text': '', 'startTime': None, 'endTime': None}
            for word in result['words']:
                if current_segment['startTime'] is None:
                    current_segment['startTime'] = word['start'] / 1000.0  # ms to seconds

                current_segment['text'] += word['text'] + ' '
                current_segment['endTime'] = word['end'] / 1000.0

                # End segment on sentence-ending punctuation
                if word['text'].rstrip()[-1:] in '.!?':
                    segments.append({
                        'text': current_segment['text'].strip(),
                        'startTime': current_segment['startTime'],
                        'endTime': current_segment['endTime'],
                        'confidence': result.get('confidence', 0.95)
                    })
                    current_segment = {'text': '', 'startTime': None, 'endTime': None}

            # Add remaining text as final segment
            if current_segment['text'].strip():
                segments.append({
                    'text': current_segment['text'].strip(),
                    'startTime': current_segment['startTime'],
                    'endTime': current_segment['endTime'],
                    'confidence': result.get('confidence', 0.95)
                })
        else:
            segments.append({
                'text': result.get('text', '').strip(),
                'startTime': 0,
                'endTime': 0,
                'confidence': result.get('confidence', 0.95)
            })

        print(f"[API Server] Transcription complete: {len(segments)} segments")

        return jsonify({
            'success': True,
            'transcription': result.get('text', '').strip(),
            'segments': segments,
            'language': result.get('language_code', 'en')
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


@app.route('/transcribe-with-diarization', methods=['POST'])
def transcribe_with_diarization():
    """
    Transcribe with REAL voice-based speaker diarization using AssemblyAI.

    This analyzes actual audio waveforms to identify different speakers,
    just like pyannote did locally. Each segment has accurate timestamps
    for audio sync and real speaker identification.
    """
    temp_path = None
    try:
        print(f"[API Server] /transcribe-with-diarization called")
        print(f"[API Server] AssemblyAI configured: {bool(assemblyai_api_key)}")
        print(f"[API Server] Request files: {list(request.files.keys())}")

        if not assemblyai_api_key:
            return jsonify({
                'success': False,
                'error': 'AssemblyAI API not configured. Add ASSEMBLYAI_API_KEY to Railway variables.'
            }), 503

        if 'audio' not in request.files:
            return jsonify({
                'success': False,
                'error': f'No audio file provided. Got files: {list(request.files.keys())}'
            }), 400

        audio_file = request.files['audio']
        print(f"[API Server] Audio file received: {audio_file.filename}, content_type: {audio_file.content_type}")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp:
            audio_file.save(temp.name)
            temp_path = temp.name

        # Check file size
        file_size = os.path.getsize(temp_path)
        print(f"[API Server] Saved temp file: {temp_path}, size: {file_size} bytes")

        if file_size == 0:
            raise Exception("Audio file is empty (0 bytes)")

        print(f"[API Server] Transcribing with speaker diarization: {audio_file.filename}")

        # Use AssemblyAI with speaker labels enabled
        result = assemblyai_transcribe(temp_path, speaker_labels=True)

        # Clean up temp file
        os.unlink(temp_path)

        # Extract utterances with speaker labels
        segments = []
        diarization = []

        if 'utterances' in result and result['utterances']:
            # AssemblyAI returns utterances grouped by speaker
            for utterance in result['utterances']:
                speaker_label = utterance.get('speaker', 'A')
                # Convert speaker letter to number (A=0, B=1, C=2, etc.)
                speaker_num = ord(speaker_label) - ord('A') if isinstance(speaker_label, str) else int(speaker_label)

                start_time = utterance['start'] / 1000.0  # ms to seconds
                end_time = utterance['end'] / 1000.0

                segments.append({
                    'text': utterance['text'].strip(),
                    'startTime': start_time,
                    'endTime': end_time,
                    'confidence': utterance.get('confidence', 0.95),
                    'speaker': speaker_num
                })

                diarization.append({
                    'speaker': speaker_num,
                    'startTime': start_time,
                    'endTime': end_time
                })
        else:
            # Fallback if no utterances
            segments.append({
                'text': result.get('text', '').strip(),
                'startTime': 0,
                'endTime': 0,
                'confidence': result.get('confidence', 0.95),
                'speaker': 0
            })
            diarization.append({
                'speaker': 0,
                'startTime': 0,
                'endTime': 0
            })

        num_speakers = len(set(seg['speaker'] for seg in segments))
        print(f"[API Server] Transcription complete: {len(segments)} segments, {num_speakers} speakers (voice-based diarization)")

        return jsonify({
            'success': True,
            'transcription': result.get('text', '').strip(),
            'segments': segments,
            'diarization': diarization,
            'language': result.get('language_code', 'en'),
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


@app.route('/study-guide', methods=['POST'])
def study_guide():
    """
    Generate comprehensive study guide from lecture transcript using OpenAI GPT.
    Used by ClearStudyCheck app to produce student-focused study materials.
    Returns: overview, keyConcepts, definitions, studyPoints, reviewQuestions,
             examTips, relatedTopics, keyTopics, enhancedTranscript
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
        lecture_title = data.get('lectureTitle', 'Lecture')
        lecture_date = data.get('lectureDate', '')
        subject = data.get('subject', 'general')
        class_name = data.get('className', '')

        print(f"[API Server] Generating study guide for: {lecture_title} (subject: {subject})")

        # Build subject-aware context
        subject_context = ""
        if subject and subject != 'general':
            subject_hints = {
                'biology': "Focus on biological processes, organisms, classifications, lab procedures, and scientific terminology. Format chemical/biological names properly.",
                'chemistry': "Focus on reactions, equations, molecular structures, mechanisms, and lab safety. Use proper chemical notation.",
                'physics': "Focus on formulas, laws, derivations, units, and problem-solving approaches. Include relevant equations.",
                'math_cs': "Focus on theorems, proofs, algorithms, code patterns, and problem-solving steps. Include formulas and notation.",
                'history': "Focus on dates, events, cause-and-effect relationships, key figures, and primary sources. Build chronological context.",
                'literature': "Focus on themes, motifs, characters, literary devices, author context, and textual analysis.",
                'psychology': "Focus on theories, key studies, researchers, statistical methods, and case examples.",
                'business': "Focus on models, frameworks, case studies, financial concepts, and market analysis.",
                'nursing_premed': "Focus on anatomy, pharmacology, clinical procedures, patient care, and medical terminology.",
                'engineering': "Focus on design principles, calculations, material properties, and system analysis.",
            }
            subject_context = subject_hints.get(subject, "Extract key academic concepts appropriate to this subject area.")

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are an expert professor and study guide creator. Your job is to transform raw lecture transcripts into COMPREHENSIVE, DETAILED study materials — the kind a top student would create after spending hours reviewing a lecture. Aim for the equivalent of 5-10 handwritten pages of study notes.

{f"Subject area: {subject}. {subject_context}" if subject_context else ""}

Given a lecture transcript, produce a JSON response. BE THOROUGH — write detailed explanations, not summaries. Each section should contain substantial content:

{{
    "overview": "A detailed 5-8 sentence overview of the lecture. Explain: (1) What the main topic was, (2) Why it matters in the broader context of the course, (3) The key arguments or ideas presented, (4) How different topics connected to each other, (5) What the professor emphasized as most important. This should give a student who missed class a solid understanding of what was covered.",

    "keyConcepts": [
        {{
            "concept": "Name of the concept",
            "explanation": "Write a DETAILED 3-5 sentence explanation. Include: what it is, why it matters, a concrete example from the lecture, and how it connects to other concepts discussed. Write as if explaining to a fellow student who missed class.",
            "importance": "Explain specifically why this matters: Is it foundational for later topics? Frequently tested? A common exam question? Connects to the textbook chapter?"
        }}
    ],

    "definitions": [
        {{
            "term": "Technical term or vocabulary word",
            "definition": "Write a clear 2-3 sentence definition. First sentence: concise definition. Second sentence: clarification or nuance. Third sentence (if needed): what it is NOT or common misconceptions.",
            "context": "How the professor used this term in the lecture — include the specific example or scenario they described"
        }}
    ],

    "studyPoints": [
        {{
            "point": "A specific fact, argument, or idea from the lecture that's worth studying",
            "details": "2-3 sentences of supporting detail: examples given, evidence mentioned, qualifications or exceptions the professor noted",
            "examRelevance": "high/medium/low"
        }}
    ],

    "reviewQuestions": [
        "Factual recall: What is X and why is it significant?",
        "Compare/contrast: How does X differ from Y in terms of Z?",
        "Application: Given scenario A, how would concept B apply?",
        "Critical thinking: Why might someone disagree with the argument that X?",
        "Synthesis: How do concepts X, Y, and Z work together to explain phenomenon W?"
    ],

    "examTips": [
        {{
            "tip": "Specific, actionable study advice — not generic. Reference actual content from this lecture.",
            "type": "memorization/understanding/application/comparison/essay-prep"
        }}
    ],

    "relatedTopics": [
        {{
            "topic": "Related topic or concept from the broader field",
            "connection": "Specifically how it connects — not just 'it's related' but explain the logical link",
            "suggestedReading": "Specific chapter, textbook section, Wikipedia article, or academic resource"
        }}
    ],

    "keyTopics": ["Topic 1", "Topic 2", "Topic 3", "Topic 4", "Topic 5"],

    "enhancedTranscript": "Rewrite the ENTIRE lecture content as clean, well-structured study notes. This is the LONGEST section and should be several paragraphs.\\n\\n## Section heading for first major topic\\n\\nRewrite what the professor said about this topic in clear, organized prose. Include all important details, examples, and explanations. Fix grammar, remove filler words (um, uh, like, you know), but preserve the substance.\\n\\n## Section heading for second major topic\\n\\nContinue with the next major topic...\\n\\nUse ## headers for major topic shifts. Use proper paragraphs. Preserve ALL important information — this should be comprehensive enough that a student could study entirely from this rewrite. The enhanced transcript should be at least 50% of your total output length."
}}

CRITICAL OUTPUT RULES:
- You MUST generate at least 5-8 key concepts with DETAILED 3-5 sentence explanations each
- You MUST generate at least 8-12 definitions with context
- You MUST generate at least 8-12 study points with details and exam relevance
- You MUST generate at least 8-10 review questions (mix: factual, analytical, application, synthesis)
- You MUST generate at least 5 exam tips with specific advice
- You MUST generate at least 3-4 related topics with specific reading suggestions
- The enhancedTranscript MUST be the longest section — rewrite ALL lecture content as organized study notes with ## section headers
- If the lecture is short (under 5 minutes), still extract maximum value but note it's a brief excerpt
- For a standard lecture (30-60+ minutes), output should be VERY comprehensive
- Output ONLY valid JSON, no other text or markdown fences
- Use \\n for newlines within string values"""
                },
                {
                    "role": "user",
                    "content": f"{'Class: ' + class_name + chr(10) if class_name else ''}Lecture: {lecture_title}\nDate: {lecture_date}\n\nTranscript:\n{transcript}"
                }
            ],
            temperature=0.3,
            max_tokens=16000
        )

        guide_text = response.choices[0].message.content.strip()

        # Parse JSON response
        try:
            if guide_text.startswith('```'):
                guide_text = guide_text.split('```')[1]
                if guide_text.startswith('json'):
                    guide_text = guide_text[4:]
            guide_text = guide_text.strip()

            # Try to extract JSON object if there's extra text
            json_match = None
            try:
                study_guide = json.loads(guide_text)
            except json.JSONDecodeError:
                import re
                match = re.search(r'\{[\s\S]*\}', guide_text)
                if match:
                    study_guide = json.loads(match.group())
                else:
                    raise ValueError("No JSON found in response")

        except (json.JSONDecodeError, ValueError) as parse_error:
            print(f"[API Server] JSON parse error, creating structured fallback: {parse_error}")
            study_guide = {
                'overview': guide_text[:500] if guide_text else 'Study guide generation failed.',
                'keyConcepts': [],
                'definitions': [],
                'studyPoints': [],
                'reviewQuestions': [],
                'examTips': [],
                'relatedTopics': [],
                'keyTopics': [],
                'enhancedTranscript': ''
            }

        # Ensure all expected fields exist with defaults
        study_guide.setdefault('overview', '')
        study_guide.setdefault('keyConcepts', [])
        study_guide.setdefault('definitions', [])
        study_guide.setdefault('studyPoints', [])
        study_guide.setdefault('reviewQuestions', [])
        study_guide.setdefault('examTips', [])
        study_guide.setdefault('relatedTopics', [])
        study_guide.setdefault('keyTopics', [])
        study_guide.setdefault('enhancedTranscript', '')

        print(f"[API Server] Study guide generated: {len(study_guide.get('keyConcepts', []))} concepts, "
              f"{len(study_guide.get('definitions', []))} terms, "
              f"{len(study_guide.get('reviewQuestions', []))} questions")

        return jsonify({
            'success': True,
            'studyGuide': study_guide
        })

    except Exception as e:
        print(f"[API Server] Study guide error: {str(e)}")
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
                    "content": """You are an intelligent study assistant for the ClearStudyCheck app. You help college students understand their lecture recordings.
                    Based on the lecture transcript provided, you can:
                      - Answer specific questions about what was taught
                      - Explain concepts from the lecture in simpler terms
                      - Provide additional context or examples for difficult topics
                      - Help students understand connections between different concepts
                      - Suggest study strategies for the material covered
                      - Create practice questions or flashcard-style Q&A
                      - Clarify confusing parts of the lecture
                      - Identify the most important points for exam preparation
                      - Reference specific things the professor/instructor said

                       Be helpful, clear, and student-friendly. Use examples and analogies.
                       If asked about topics completely unrelated to the lecture, politely redirect to the lecture content."""

                },
                {
                    "role": "user",
                    "content": f"Lecture Transcript:\n{transcript}\n\nStudent Question: {question}"
                }
            ],
            temperature=0.3,
            max_tokens=1024
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

# ============================================
# Math/Physics/Chemistry Solver (GPT-4o)
# Used by ClearStudyCheck Math tab
# ============================================
 
@app.route('/math', methods=['POST'])
def solve_math():
    """
    Solve math, physics, or chemistry problems using GPT-4o.
    Accepts either LaTeX text input or a base64-encoded image.
    Returns structured step-by-step solution with explanations.
    
    Request JSON: { "latex": "\\int x^2 + 3x \\, dx" }
           or:    { "image": "<base64 string>" }
    """
    try:
        if not openai_client:
            return jsonify({
                'success': False,
                'error': 'OpenAI API not configured. Set OPENAI_API_KEY environment variable.'
            }), 503
 
        data = request.json
        image_b64   = data.get('image', '')
        latex_input = data.get('latex', '')
 
        if not image_b64 and not latex_input:
            return jsonify({
                'success': False,
                'error': 'No input provided. Send either "latex" or "image" field.'
            }), 400
 
        SYSTEM_PROMPT = """You are an expert STEM tutor for college students.
Return ONLY a valid JSON object with no markdown, no backticks, nothing else:
{
  "subject": "math" | "physics" | "chemistry",
  "detected_problem": "clean readable problem string",
  "answer": "final answer as clean string",
  "difficulty": "beginner" | "intermediate" | "advanced",
  "steps": [
    {
      "step_number": 1,
      "title": "Short action title e.g. Apply the power rule to x²",
      "expression": "The actual math expression for this step e.g. x³/3",
      "explanation": "2-3 sentences. Name the exact rule or theorem used. Explain WHY it applies here — help the student understand the concept, not just the calculation.",
      "rule_used": "Name of the rule or theorem e.g. Power rule for integration"
    }
  ],
  "concept_explanation": "3-4 sentences of plain English explaining the underlying concept a student needs to understand to solve this class of problem.",
  "topics": ["topic1", "topic2", "topic3"]
}
CRITICAL RULES:
- Minimum 4 steps, aim for 5-6 for multi-part problems
- Every explanation must name the rule AND explain WHY it applies
- expression must show the actual math for that specific step
- Steps must build logically from each other
- concept_explanation must genuinely help a college student understand"""
 
        content = []
 
        if image_b64:
            print(f"[API Server] Math solver: processing image input ({len(image_b64)} chars)")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
            })
            content.append({
                "type": "text",
                "text": "Solve the math, physics, or chemistry problem shown in this image. Follow the JSON format in the system prompt exactly."
            })
        else:
            print(f"[API Server] Math solver: processing LaTeX input: {latex_input[:100]}")
            content.append({
                "type": "text",
                "text": f"Solve this problem: {latex_input}\nFollow the JSON format in the system prompt exactly."
            })
 
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": content}
            ],
            max_tokens=2000
        )
 
        result_text = response.choices[0].message.content.strip()
 
        # Clean markdown fences if GPT adds them
        if result_text.startswith('```'):
            result_text = result_text.split('```')[1]
            if result_text.startswith('json'):
                result_text = result_text[4:]
            result_text = result_text.strip()
 
        result = json.loads(result_text)
        print(f"[API Server] Math solver: {result.get('subject', 'unknown')} problem solved — {len(result.get('steps', []))} steps")
 
        return jsonify(result)
 
    except json.JSONDecodeError as e:
        print(f"[API Server] Math solver JSON parse error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to parse GPT-4o response as JSON'
        }), 500
    except Exception as e:
        print(f"[API Server] Math solver error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500        


if __name__ == '__main__':
    print("\n" + "="*50)
    print("ClearNoteCheck Cloud API Server v3.0")
    print("="*50)
    print(f"OpenAI: {'Configured' if openai_client else 'NOT CONFIGURED'}")
    print(f"AssemblyAI: {'Configured (real voice diarization)' if assemblyai_api_key else 'NOT CONFIGURED'}")
    print("Endpoints:")
    print("  GET  /              - API info")
    print("  GET  /health        - Health check")
    print("  POST /transcribe    - Transcribe audio (AssemblyAI)")
    print("  POST /transcribe-with-diarization - Transcribe + real speaker ID")
    print("  POST /summarize     - Generate AI summary")
    print("  POST /executive-summary - Executive summary")
    print("  POST /chat          - Chat with lecture transcript")
    print("  POST /study-guide   - Generate comprehensive study guide")
    print("="*50 + "\n")

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
