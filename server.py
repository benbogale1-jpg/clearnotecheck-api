"""
ClearLectureCheck - Cloud API Server
Flask server for OpenAI-powered lecture analysis and study material generation.

Endpoints:
    GET  /health                      - Health check
    POST /transcribe                  - Transcribe audio using AssemblyAI
    POST /transcribe-with-diarization - Transcribe with speaker diarization
    POST /summarize                   - Generate lecture summary
    POST /executive-summary           - Generate executive summary (legacy)
    POST /study-guide                 - Generate comprehensive study guide
    POST /study-materials             - Generate structured study materials
    POST /flashcards                  - Generate flashcards from transcript
    POST /quiz                        - Generate practice quiz questions
    POST /chat                        - Chat with transcript (RAG-ready)

Environment variables:
    OPENAI_API_KEY - Required: OpenAI API key for GPT
    ASSEMBLYAI_API_KEY - Required: AssemblyAI API key for transcription
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
UPLOAD_SESSIONS = {}
CHUNK_DIR = tempfile.mkdtemp(prefix='clc_chunks_')

# Initialize OpenAI client
openai_api_key = os.environ.get('OPENAI_API_KEY')
if not openai_api_key:
    print("[CLC API] WARNING: OPENAI_API_KEY not set!")
    openai_client = None
else:
    openai_client = OpenAI(api_key=openai_api_key)
    print("[CLC API] OpenAI client initialized!")

# Initialize AssemblyAI
assemblyai_api_key = os.environ.get('ASSEMBLYAI_API_KEY')
if not assemblyai_api_key:
    print("[CLC API] WARNING: ASSEMBLYAI_API_KEY not set!")
else:
    print("[CLC API] AssemblyAI configured!")


# ============================================
# Health & Info Endpoints
# ============================================

@app.route('/', methods=['GET'])
def root():
    """Root endpoint - API info"""
    return jsonify({
        'service': 'ClearNoteCheck Cloud API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': [
            'GET  /health',
            'POST /transcribe',
            'POST /transcribe-with-diarization',
            'POST /summarize',
            'POST /study-guide',
            'POST /study-materials',
            'POST /flashcards',
            'POST /quiz',
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
        'timestamp': datetime.now().isoformat()
    })


# ============================================
# Transcription Endpoints
# ============================================

def assemblyai_transcribe(audio_path, speaker_labels=False):
    """Transcribe audio using AssemblyAI with optional speaker diarization."""
    headers = {
        'authorization': assemblyai_api_key,
        'content-type': 'application/json'
    }

    # Step 1: Upload the audio file
    print("[CLC API] Uploading audio to AssemblyAI...")
    with open(audio_path, 'rb') as f:
        upload_response = requests.post(
            'https://api.assemblyai.com/v2/upload',
            headers={'authorization': assemblyai_api_key},
            data=f
        )

    upload_data = upload_response.json()
    if 'error' in upload_data:
        raise Exception(f"AssemblyAI upload error: {upload_data['error']}")
    if 'upload_url' not in upload_data:
        raise Exception(f"AssemblyAI upload failed: {upload_data}")

    upload_url = upload_data['upload_url']
    print("[CLC API] Audio uploaded successfully")

    # Step 2: Request transcription
    transcript_request = {
        'audio_url': upload_url,
        'speaker_labels': speaker_labels,
    }

    print(f"[CLC API] Requesting transcription (speaker_labels={speaker_labels})...")
    transcript_response = requests.post(
        'https://api.assemblyai.com/v2/transcript',
        headers=headers,
        json=transcript_request
    )

    transcript_data = transcript_response.json()
    if 'error' in transcript_data:
        raise Exception(f"AssemblyAI transcript error: {transcript_data['error']}")
    if 'id' not in transcript_data:
        raise Exception(f"AssemblyAI transcript failed: {transcript_data}")

    transcript_id = transcript_data['id']
    print(f"[CLC API] Transcription started: {transcript_id}")

    # Step 3: Poll for completion
    polling_url = f'https://api.assemblyai.com/v2/transcript/{transcript_id}'
    max_polls = 300  # Max 10 minutes

    for polls in range(max_polls):
        poll_response = requests.get(polling_url, headers=headers)
        poll_data = poll_response.json()
        status = poll_data.get('status', 'unknown')

        if status == 'completed':
            print("[CLC API] Transcription completed!")
            return poll_data
        elif status == 'error':
            raise Exception(f"AssemblyAI error: {poll_data.get('error', 'Unknown')}")

        print(f"[CLC API] Status: {status}, waiting... ({polls + 1}/{max_polls})")
        time.sleep(2)

    raise Exception("Transcription timed out after 10 minutes")


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Transcribe audio file using AssemblyAI"""
    temp_path = None
    try:
        if not assemblyai_api_key:
            return jsonify({'success': False, 'error': 'AssemblyAI not configured'}), 503

        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp:
            audio_file.save(temp.name)
            temp_path = temp.name

        result = assemblyai_transcribe(temp_path, speaker_labels=False)
        os.unlink(temp_path)

        # Extract segments with timestamps
        segments = []
        if 'words' in result and result['words']:
            current_segment = {'text': '', 'startTime': None, 'endTime': None}
            for word in result['words']:
                if current_segment['startTime'] is None:
                    current_segment['startTime'] = word['start'] / 1000.0

                current_segment['text'] += word['text'] + ' '
                current_segment['endTime'] = word['end'] / 1000.0

                if word['text'].rstrip()[-1:] in '.!?':
                    segments.append({
                        'text': current_segment['text'].strip(),
                        'startTime': current_segment['startTime'],
                        'endTime': current_segment['endTime'],
                        'confidence': result.get('confidence', 0.95)
                    })
                    current_segment = {'text': '', 'startTime': None, 'endTime': None}

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

        return jsonify({
            'success': True,
            'transcription': result.get('text', '').strip(),
            'segments': segments,
            'language': result.get('language_code', 'en')
        })

    except Exception as e:
        print(f"[CLC API] Transcription error: {str(e)}")
        if temp_path:
            try:
                os.unlink(temp_path)
            except:
                pass
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/transcribe-with-diarization', methods=['POST'])
def transcribe_with_diarization():
    """Transcribe with speaker diarization using AssemblyAI"""
    temp_path = None
    try:
        if not assemblyai_api_key:
            return jsonify({'success': False, 'error': 'AssemblyAI not configured'}), 503

        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp:
            audio_file.save(temp.name)
            temp_path = temp.name

        result = assemblyai_transcribe(temp_path, speaker_labels=True)
        os.unlink(temp_path)

        segments = []
        diarization = []

        if 'utterances' in result and result['utterances']:
            for utterance in result['utterances']:
                speaker_label = utterance.get('speaker', 'A')
                speaker_num = ord(speaker_label) - ord('A') if isinstance(speaker_label, str) else int(speaker_label)

                start_time = utterance['start'] / 1000.0
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
            segments.append({
                'text': result.get('text', '').strip(),
                'startTime': 0,
                'endTime': 0,
                'confidence': result.get('confidence', 0.95),
                'speaker': 0
            })

        num_speakers = len(set(seg['speaker'] for seg in segments))

        return jsonify({
            'success': True,
            'transcription': result.get('text', '').strip(),
            'segments': segments,
            'diarization': diarization,
            'language': result.get('language_code', 'en'),
            'numSpeakers': num_speakers
        })

    except Exception as e:
        print(f"[CLC API] Transcription error: {str(e)}")
        if temp_path:
            try:
                os.unlink(temp_path)
            except:
                pass
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================
# Summary Endpoints
# ============================================

@app.route('/summarize', methods=['POST'])
def summarize():
    """Generate lecture summary from transcript"""
    try:
        if not openai_client:
            return jsonify({'success': False, 'error': 'OpenAI not configured'}), 503

        data = request.get_json()
        if not data or 'transcript' not in data:
            return jsonify({'success': False, 'error': 'No transcript provided'}), 400

        transcript = data['transcript']
        print(f"[CLC API] Generating summary ({len(transcript)} chars)")

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are a lecture notes assistant for college students. Given a lecture transcript, produce a JSON response with:
1. "bulletPoints": array of 4-6 key points from the lecture
2. "actionItems": array of objects with "text" and "assignee" fields (study tasks, assignments mentioned)
3. "keyTopics": array of main topics covered
4. "decisions": array of important conclusions or takeaways

Focus on educational content. Be concise and student-friendly. Output ONLY valid JSON."""
                },
                {
                    "role": "user",
                    "content": f"Summarize this lecture transcript:\n\n{transcript}"
                }
            ],
            temperature=0.3,
            max_tokens=1024
        )

        summary_text = response.choices[0].message.content.strip()

        try:
            if summary_text.startswith('```'):
                summary_text = summary_text.split('```')[1]
                if summary_text.startswith('json'):
                    summary_text = summary_text[4:]
            summary = json.loads(summary_text.strip())
        except json.JSONDecodeError:
            summary = {
                'bulletPoints': [summary_text[:200]],
                'actionItems': [],
                'keyTopics': [],
                'decisions': []
            }

        return jsonify({'success': True, 'summary': summary})

    except Exception as e:
        print(f"[CLC API] Summary error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/executive-summary', methods=['POST'])
def executive_summary():
    """Generate executive summary (legacy endpoint)"""
    try:
        if not openai_client:
            return jsonify({'success': False, 'error': 'OpenAI not configured'}), 503

        data = request.get_json()
        if not data or 'transcript' not in data:
            return jsonify({'success': False, 'error': 'No transcript provided'}), 400

        transcript = data['transcript']
        meeting_title = data.get('meetingTitle', 'Lecture')
        meeting_date = data.get('meetingDate', '')

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are an academic assistant. Generate an executive summary for a lecture:

{
    "executiveBrief": "2-3 sentence overview of the lecture content",
    "keyDecisions": [{"decision": "Key concept", "impact": "Why it matters", "owner": "Topic area"}],
    "actionItems": [{"task": "Study task", "assignee": "Student", "deadline": "When", "priority": "high/medium/low"}],
    "risksAndBlockers": [{"issue": "Difficult concept", "severity": "high/medium/low", "mitigation": "Study approach"}],
    "nextSteps": ["Review notes", "Practice problems"],
    "followUpMeetings": [],
    "keyMetrics": []
}

Output ONLY valid JSON."""
                },
                {
                    "role": "user",
                    "content": f"Lecture: {meeting_title}\nDate: {meeting_date}\n\nTranscript:\n{transcript}"
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
            executive_summary_data = json.loads(summary_text.strip())
        except json.JSONDecodeError:
            executive_summary_data = {
                'executiveBrief': summary_text[:500],
                'keyDecisions': [],
                'actionItems': [],
                'risksAndBlockers': [],
                'nextSteps': [],
                'followUpMeetings': [],
                'keyMetrics': []
            }

        return jsonify({'success': True, 'executiveSummary': executive_summary_data})

    except Exception as e:
        print(f"[CLC API] Executive summary error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================
# Study Guide Endpoint
# ============================================

@app.route('/study-guide', methods=['POST'])
def study_guide():
    """Generate comprehensive study guide from lecture transcript"""
    try:
        if not openai_client:
            return jsonify({'success': False, 'error': 'OpenAI not configured'}), 503

        data = request.get_json()
        if not data or 'transcript' not in data:
            return jsonify({'success': False, 'error': 'No transcript provided'}), 400

        transcript = data['transcript']
        lecture_title = data.get('lectureTitle', 'Lecture')
        lecture_date = data.get('lectureDate', '')
        subject = data.get('subject', 'general')
        class_name = data.get('className', '')

        print(f"[CLC API] Generating study guide for: {lecture_title} ({subject})")

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are an expert academic tutor creating a comprehensive study guide for a {subject} lecture.
Generate a detailed JSON study guide with these sections:

{{
    "overview": "2-3 paragraph summary of the lecture content and learning objectives",
    "keyConcepts": [
        {{"concept": "Main concept name", "explanation": "Clear explanation", "importance": "Why this matters for exams/understanding"}}
    ],
    "studyPoints": [
        {{"point": "Key study point", "details": "Detailed explanation", "examRelevance": "high/medium/low"}}
    ],
    "definitions": [
        {{"term": "Technical term", "definition": "Clear definition", "context": "How it's used in this subject"}}
    ],
    "reviewQuestions": ["Question 1?", "Question 2?"],
    "relatedTopics": [
        {{"topic": "Related topic", "connection": "How it connects", "suggestedReading": "What to read"}}
    ],
    "examTips": [
        {{"tip": "Study tip", "type": "memorization/understanding/application"}}
    ],
    "keyTopics": ["Topic 1", "Topic 2"]
}}

Be thorough and student-focused. Include 5-8 key concepts, 6-10 study points, 8-12 definitions, and 5-8 review questions.
Output ONLY valid JSON."""
                },
                {
                    "role": "user",
                    "content": f"Class: {class_name}\nLecture: {lecture_title}\nDate: {lecture_date}\n\nTranscript:\n{transcript}"
                }
            ],
            temperature=0.3,
            max_tokens=4096
        )

        guide_text = response.choices[0].message.content.strip()

        try:
            if guide_text.startswith('```'):
                guide_text = guide_text.split('```')[1]
                if guide_text.startswith('json'):
                    guide_text = guide_text[4:]
            study_guide_data = json.loads(guide_text.strip())
        except json.JSONDecodeError as e:
            print(f"[CLC API] JSON parse error: {e}")
            study_guide_data = {
                'overview': guide_text[:500],
                'keyConcepts': [],
                'studyPoints': [],
                'definitions': [],
                'reviewQuestions': [],
                'relatedTopics': [],
                'examTips': [],
                'keyTopics': []
            }

        print(f"[CLC API] Study guide generated: {len(study_guide_data.get('keyConcepts', []))} concepts")
        return jsonify({'success': True, 'studyGuide': study_guide_data})

    except Exception as e:
        print(f"[CLC API] Study guide error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================
# Study Materials Endpoint
# ============================================

@app.route('/study-materials', methods=['POST'])
def study_materials():
    """Generate structured study materials from transcript"""
    try:
        if not openai_client:
            return jsonify({'success': False, 'error': 'OpenAI not configured'}), 503

        data = request.get_json()
        if not data or 'transcript' not in data:
            return jsonify({'success': False, 'error': 'No transcript provided'}), 400

        transcript = data['transcript']
        system_prompt = data.get('systemPrompt', '')
        lecture_title = data.get('lectureTitle', 'Lecture')

        print(f"[CLC API] Generating study materials for: {lecture_title}")

        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt or """You are an expert academic tutor. Generate comprehensive study materials:

{
    "structuredNotes": {
        "title": "Lecture title",
        "sections": [
            {"heading": "Section heading", "content": "Detailed content", "keyPoints": ["Point 1"], "importantTerms": ["Term 1"]}
        ],
        "examTips": ["Tip 1", "Tip 2"]
    },
    "keyTerms": [
        {"term": "Term", "definition": "Definition", "context": "Context"}
    ],
    "practiceQuestions": [
        {"type": "multiple_choice/short_answer/essay", "question": "Question?", "options": ["A", "B", "C", "D"], "correctAnswer": "A", "explanation": "Why", "difficulty": "easy/medium/hard"}
    ],
    "flashcards": [
        {"front": "Question/Term", "back": "Answer/Definition", "difficulty": "easy/medium/hard"}
    ]
}

Output ONLY valid JSON."""
                },
                {
                    "role": "user",
                    "content": f"Generate study materials for this lecture:\n\nTitle: {lecture_title}\n\nTranscript:\n{transcript}"
                }
            ],
            temperature=0.3,
            max_tokens=4096
        )

        materials_text = response.choices[0].message.content.strip()

        try:
            if materials_text.startswith('```'):
                materials_text = materials_text.split('```')[1]
                if materials_text.startswith('json'):
                    materials_text = materials_text[4:]
            study_materials_data = json.loads(materials_text.strip())
        except json.JSONDecodeError:
            study_materials_data = {}

        return jsonify({'success': True, 'studyMaterials': study_materials_data})

    except Exception as e:
        print(f"[CLC API] Study materials error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================
# Flashcards Endpoint
# ============================================

@app.route('/flashcards', methods=['POST'])
def flashcards():
    """Generate flashcards from lecture transcript"""
    try:
        if not openai_client:
            return jsonify({'success': False, 'error': 'OpenAI not configured'}), 503

        data = request.get_json()
        if not data or 'transcript' not in data:
            return jsonify({'success': False, 'error': 'No transcript provided'}), 400

        transcript = data['transcript']
        subject_category = data.get('subjectCategory', 'general')

        print(f"[CLC API] Generating flashcards ({subject_category})")

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a {subject_category} tutor creating flashcards for exam prep.
Generate 15-25 high-quality flashcards from this lecture.

Return JSON array:
[
    {{"front": "Question or term", "back": "Answer or definition", "difficulty": "easy/medium/hard"}}
]

Mix question types:
- Definitions (What is X?)
- Concepts (Explain Y)
- Applications (How would you Z?)
- Comparisons (Difference between A and B?)

Make cards specific and testable. Output ONLY valid JSON array."""
                },
                {
                    "role": "user",
                    "content": f"Create flashcards from this lecture:\n\n{transcript}"
                }
            ],
            temperature=0.4,
            max_tokens=2048
        )

        cards_text = response.choices[0].message.content.strip()

        try:
            if cards_text.startswith('```'):
                cards_text = cards_text.split('```')[1]
                if cards_text.startswith('json'):
                    cards_text = cards_text[4:]
            flashcards_data = json.loads(cards_text.strip())
        except json.JSONDecodeError:
            flashcards_data = []

        print(f"[CLC API] Generated {len(flashcards_data)} flashcards")
        return jsonify({'success': True, 'flashcards': flashcards_data})

    except Exception as e:
        print(f"[CLC API] Flashcards error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================
# Quiz Endpoint
# ============================================

@app.route('/quiz', methods=['POST'])
def quiz():
    """Generate practice quiz from lecture transcript"""
    try:
        if not openai_client:
            return jsonify({'success': False, 'error': 'OpenAI not configured'}), 503

        data = request.get_json()
        if not data or 'transcript' not in data:
            return jsonify({'success': False, 'error': 'No transcript provided'}), 400

        transcript = data['transcript']
        subject_category = data.get('subjectCategory', 'general')
        question_count = data.get('questionCount', 10)

        print(f"[CLC API] Generating {question_count} quiz questions ({subject_category})")

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"""You are a {subject_category} professor creating a practice quiz.
Generate exactly {question_count} questions from this lecture.

Return JSON array:
[
    {{
        "type": "multiple_choice",
        "question": "Question text?",
        "options": ["A) Option A", "B) Option B", "C) Option C", "D) Option D"],
        "correctAnswer": "A",
        "explanation": "Why A is correct",
        "difficulty": "easy/medium/hard"
    }},
    {{
        "type": "short_answer",
        "question": "Question text?",
        "correctAnswer": "Expected answer",
        "explanation": "Key points to include",
        "difficulty": "medium"
    }}
]

Mix 70% multiple choice, 30% short answer. Test understanding, not just memorization.
Output ONLY valid JSON array."""
                },
                {
                    "role": "user",
                    "content": f"Create a {question_count}-question quiz from this lecture:\n\n{transcript}"
                }
            ],
            temperature=0.4,
            max_tokens=3000
        )

        quiz_text = response.choices[0].message.content.strip()

        try:
            if quiz_text.startswith('```'):
                quiz_text = quiz_text.split('```')[1]
                if quiz_text.startswith('json'):
                    quiz_text = quiz_text[4:]
            questions_data = json.loads(quiz_text.strip())
        except json.JSONDecodeError:
            questions_data = []

        print(f"[CLC API] Generated {len(questions_data)} questions")
        return jsonify({'success': True, 'questions': questions_data})

    except Exception as e:
        print(f"[CLC API] Quiz error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================
# Chat Endpoint
# ============================================

@app.route('/chat', methods=['POST'])
def chat():
    """Chat with the lecture transcript"""
    try:
        if not openai_client:
            return jsonify({'success': False, 'error': 'OpenAI not configured'}), 503

        data = request.get_json()
        if not data or 'transcript' not in data or 'question' not in data:
            return jsonify({'success': False, 'error': 'Missing transcript or question'}), 400

        transcript = data['transcript']
        question = data['question']

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """You are an intelligent lecture assistant for the ClearLectureCheck app.
Based on the lecture transcript, you can:
- Answer questions about the content
- Explain concepts in different ways
- Provide examples and analogies
- Suggest study strategies
- Clarify confusing points
- Connect topics to broader context

Be helpful, educational, and reference specific content from the transcript.
If asked about unrelated topics, gently redirect to the lecture content."""
                },
                {
                    "role": "user",
                    "content": f"Lecture Transcript:\n{transcript}\n\nStudent Question: {question}"
                }
            ],
            temperature=0.4,
            max_tokens=1024
        )

        answer = response.choices[0].message.content.strip()

        return jsonify({'success': True, 'answer': answer})

    except Exception as e:
        print(f"[CLC API] Chat error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================
# Chunked Upload Endpoints
# ============================================

@app.route('/upload/init', methods=['POST'])
def upload_init():
    """Initialize a chunked upload session"""
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

        return jsonify({'success': True, 'session_id': session_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/upload/chunk', methods=['POST'])
def upload_chunk():
    """Upload a single chunk"""
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

        return jsonify({
            'success': True,
            'chunks_received': len(session['chunks_received']),
            'total_chunks': session['total_chunks']
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/upload/complete', methods=['POST'])
def upload_complete():
    """Finalize upload and transcribe"""
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id')
        speaker_labels = data.get('speaker_labels', True)

        if session_id not in UPLOAD_SESSIONS:
            return jsonify({'success': False, 'error': 'Invalid session_id'}), 400

        session = UPLOAD_SESSIONS[session_id]

        if len(session['chunks_received']) < session['total_chunks']:
            return jsonify({
                'success': False,
                'error': f"Missing chunks: got {len(session['chunks_received'])}/{session['total_chunks']}"
            }), 400

        # Combine chunks
        ext = session.get('extension', 'm4a')
        combined_path = os.path.join(session['dir'], f'combined.{ext}')

        with open(combined_path, 'wb') as outfile:
            for i in range(session['total_chunks']):
                chunk_path = os.path.join(session['dir'], f'chunk_{i:04d}')
                with open(chunk_path, 'rb') as chunk_file:
                    outfile.write(chunk_file.read())

        # Transcribe
        result = assemblyai_transcribe(combined_path, speaker_labels)

        # Clean up
        shutil.rmtree(session['dir'], ignore_errors=True)
        del UPLOAD_SESSIONS[session_id]

        # Process result
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
        if session_id and session_id in UPLOAD_SESSIONS:
            session = UPLOAD_SESSIONS[session_id]
            shutil.rmtree(session.get('dir', ''), ignore_errors=True)
            del UPLOAD_SESSIONS[session_id]
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================
# Main
# ============================================



# ============================================
# Math/Physics/Chemistry Solver (GPT-4o)
# Used by ClearStudyCheck Math tab
# ============================================

@app.route('/math', methods=['POST'])
def solve_math():
    try:
        if not openai_client:
            return jsonify({'success': False, 'error': 'OpenAI not configured'}), 503
        data = request.json
        image_b64 = data.get('image', '')
        latex_input = data.get('latex', '')
        if not image_b64 and not latex_input:
            return jsonify({'success': False, 'error': 'No input provided'}), 400
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
      "title": "Short action title",
      "expression": "The actual math expression for this step",
      "explanation": "2-3 sentences. Name the exact rule or theorem used. Explain WHY it applies here.",
      "rule_used": "Name of the rule or theorem"
    }
  ],
  "concept_explanation": "3-4 sentences of plain English explaining the underlying concept.",
  "topics": ["topic1", "topic2", "topic3"]
}
CRITICAL RULES:
- Minimum 4 steps, aim for 5-6 for multi-part problems
- Every explanation must name the rule AND explain WHY it applies
- expression must show the actual math for that specific step
- Steps must build logically from each other
- concept_explanation must genuinely help a college student understand"""
        content_list = []
        if image_b64:
            print(f"[API Server] Math solver: image input ({len(image_b64)} chars)")
            content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
            content_list.append({"type": "text", "text": "Solve the problem shown in this image. Follow the JSON format in the system prompt exactly."})
        else:
            print(f"[API Server] Math solver: LaTeX input: {latex_input[:100]}")
            content_list.append({"type": "text", "text": f"Solve this problem: {latex_input}\nFollow the JSON format in the system prompt exactly."})
        response = openai_client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content_list}], max_tokens=2000)
        result_text = response.choices[0].message.content.strip()
        if result_text.startswith("`"):
            result_text = result_text.split("`")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
            result_text = result_text.strip()
        result = json.loads(result_text)
        print(f"[API Server] Math solver: {result.get('subject', 'unknown')} solved")
        return jsonify(result)
    except json.JSONDecodeError as e:
        print(f"[API Server] Math JSON error: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to parse response'}), 500
    except Exception as e:
        print(f"[API Server] Math error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("ClearNoteCheck Cloud API Server v1.0")
    print("=" * 50)
    print(f"OpenAI: {'Configured' if openai_client else 'NOT CONFIGURED'}")
    print(f"AssemblyAI: {'Configured' if assemblyai_api_key else 'NOT CONFIGURED'}")
    print("Endpoints:")
    print("  GET  /health        - Health check")
    print("  POST /transcribe    - Transcribe audio")
    print("  POST /summarize     - Generate summary")
    print("  POST /study-guide   - Generate study guide")
    print("  POST /study-materials - Generate study materials")
    print("  POST /flashcards    - Generate flashcards")
    print("  POST /quiz          - Generate quiz")
    print("  POST /chat          - Chat with transcript")
    print("=" * 50 + "\n")

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

