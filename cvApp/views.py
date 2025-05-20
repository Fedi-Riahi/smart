from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.http import HttpResponse, FileResponse, JsonResponse
from .models import CV
import pdfplumber
import spacy
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
import logging
import requests
import json
from django.core.cache import cache

logger = logging.getLogger(__name__)

# Load the trained NER model
try:
    nlp = spacy.load(os.path.join(settings.BASE_DIR, 'ner_model'))
except Exception as e:
    logger.error(f"Failed to load NER model: {str(e)}")
    nlp = None

def generate_rule_based_suggestions(extracted_text, skills):
    """Generate rule-based suggestions for CV improvement"""
    suggestions = {
        'general': [],
        'professional_summary': [],
        'work_experience': [],
        'education': [],
        'skills': []
    }

    # General checks
    if len(extracted_text.split()) < 100:
        suggestions['general'].append("Your CV seems too short. Aim for 1-2 pages with detailed content.")
    if not any(keyword in extracted_text.lower() for keyword in ['email', 'phone', 'linkedin']):
        suggestions['general'].append("Include contact information (email, phone, LinkedIn) at the top.")

    # Professional Summary
    if 'summary' not in extracted_text.lower() and 'objective' not in extracted_text.lower():
        suggestions['professional_summary'].append("Add a concise professional summary (3-5 sentences) highlighting your career goals and strengths.")

    # Work Experience
    if 'experience' not in extracted_text.lower():
        suggestions['work_experience'].append("Include a 'Work Experience' section with job titles, companies, dates, and key achievements.")
    if not any(keyword in extracted_text.lower() for keyword in ['achieved', 'improved', 'increased', 'developed']):
        suggestions['work_experience'].append("Use action verbs (e.g., 'achieved', 'developed') and quantify achievements (e.g., 'increased sales by 20%').")

    # Education
    if 'education' not in extracted_text.lower():
        suggestions['education'].append("Add an 'Education' section with degrees, institutions, and graduation years.")

    # Skills
    if len(skills) < 3:
        suggestions['skills'].append("List at least 5 relevant technical or soft skills to strengthen your CV.")

    devops_skills = {'AWS', 'Docker', 'Kubernetes', 'CI/CD', 'Terraform'}
    web_dev_skills = {'React', 'Angular', 'Node.js', 'Django', 'Flask'}

    if not any(skill in devops_skills for skill in skills):
        suggestions['skills'].append("For DevOps roles, consider adding skills like AWS, Docker, or Kubernetes.")
    if not any(skill in web_dev_skills for skill in skills):
        suggestions['skills'].append("For web development, consider adding skills like React, Angular, or Node.js.")

    return suggestions

def generate_gemini_suggestions(extracted_text):
    """Generate AI-powered CV suggestions using Gemini API"""
    try:
        headers = {
            "Content-Type": "application/json",
        }

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                "You are an expert HR consultant. Provide JSON output with the following structure:\n"
                                "- general (overall improvements)\n"
                                "- professional_summary\n"
                                "- work_experience\n"
                                "- education\n"
                                "- skills\n"
                                "- skills_list (extracted skills)\n\n"
                                f"Analyze this CV:\n\n{extracted_text[:10000]}"  # Truncate to ~2000 words
                            )
                        }
                    ]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.7,
                "maxOutputTokens": 1500
            }
        }

        # Construct the API URL with the API key
        api_url = f"{settings.GEMINI_API_URL}?key={settings.GEMINI_API_KEY}"

        response = requests.post(
            api_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()

        # Cache rate limit (adjust based on Gemini's limits)
        cache.set("gemini_api_limit", True, timeout=60)

        # Parse response
        result = response.json()
        # Gemini response structure: content is under 'candidates[0].content.parts[0].text'
        content = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '{}')

        # Handle both string and dict responses
        if isinstance(content, str):
            return json.loads(content)
        return content

    except requests.exceptions.RequestException as e:
        logger.error(f"Gemini API request error: {str(e)}")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to parse Gemini API response: {str(e)}")
        return None
@login_required
@login_required
def upload_cv(request):
    error = None
    if request.method == 'POST':
        if 'cv' not in request.FILES:
            error = 'No file uploaded.'
        else:
            cv_file = request.FILES['cv']
            if not cv_file.name.endswith('.pdf'):
                error = 'File must be a PDF.'
            else:
                try:
                    cv = CV.objects.create(user=request.user, file=cv_file)
                    text = ''
                    with pdfplumber.open(cv_file) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text
                    if not text.strip():
                        error = 'No text extracted from the PDF.'
                        cv.delete()
                    else:
                        cv.extracted_text = text
                        ai_suggestions = generate_gemini_suggestions(text)
                        if isinstance(ai_suggestions, str):
                            try:
                                ai_suggestions = json.loads(ai_suggestions)
                            except json.JSONDecodeError:
                                ai_suggestions = None
                        if not ai_suggestions:
                            ai_suggestions = {
                                'general': ["AI analysis unavailable. Showing basic suggestions."],
                                'skills_list': []
                            }
                        # Debug the ai_suggestions structure
                        logger.debug(f"AI Suggestions: {ai_suggestions}")

                        # Normalize ai_suggestions to ensure fields are lists
                        def ensure_list(value):
                            if isinstance(value, dict):
                                # Convert dict to list (e.g., take values or specific key)
                                return list(value.values()) if value else []
                            elif isinstance(value, list):
                                return value
                            elif value is None:
                                return []
                            else:
                                return [value]

                        skills = ensure_list(ai_suggestions.get('skills_list', []))
                        if not skills and nlp:
                            doc = nlp(text)
                            skills = [ent.text for ent in doc.ents if ent.label_ == 'SKILL']
                        rule_suggestions = generate_rule_based_suggestions(text, skills)
                        combined_suggestions = {
                            'general': rule_suggestions['general'] + ensure_list(ai_suggestions.get('general', [])),
                            'professional_summary': rule_suggestions['professional_summary'] + ensure_list(ai_suggestions.get('professional_summary', [])),
                            'work_experience': rule_suggestions['work_experience'] + ensure_list(ai_suggestions.get('work_experience', [])),
                            'education': rule_suggestions['education'] + ensure_list(ai_suggestions.get('education', [])),
                            'skills': rule_suggestions['skills'] + ensure_list(ai_suggestions.get('skills', []))
                        }
                        cv.extracted_data = {
                            'skills': skills,
                            'suggestions': combined_suggestions
                        }
                        cv.save()
                        return redirect('cv_results', cv_id=cv.id)
                except Exception as e:
                    error = f'Error processing file: {str(e)}'
                    if 'cv' in locals():
                        cv.delete()
    return render(request, 'upload.html', {'error': error})

@login_required
def cv_results(request, cv_id):
    cv = get_object_or_404(CV, id=cv_id, user=request.user)
    return render(request, 'results.html', {'cv': cv})


@login_required
def modify_cv(request, cv_id):
    cv = get_object_or_404(CV, id=cv_id, user=request.user)
    if request.method == 'POST':
        modified_text = request.POST.get('modified_text')
        if modified_text:
            try:
                buffer = BytesIO()
                p = canvas.Canvas(buffer, pagesize=letter)
                y = 750
                for line in modified_text.split('\n'):
                    p.drawString(50, y, line[:100])
                    y -= 15
                    if y < 50:
                        p.showPage()
                        y = 750
                p.save()
                buffer.seek(0)
                from django.core.files import File
                cv.modified_file.save(f'modified_cv_{cv.id}.pdf', File(buffer))
                cv.extracted_text = modified_text
                doc = nlp(modified_text)
                skills = [ent.text for ent in doc.ents if ent.label_ == 'SKILL']
                rule_suggestions = generate_rule_based_suggestions(modified_text, skills)
                gemini_suggestions = generate_gemini_suggestions(modified_text)  # Updated here
                if isinstance(gemini_suggestions, str):
                    try:
                        gemini_suggestions = json.loads(gemini_suggestions)
                    except json.JSONDecodeError:
                        gemini_suggestions = None
                combined_suggestions = {
                    'general': rule_suggestions['general'] + (gemini_suggestions.get('general', []) if gemini_suggestions else []),
                    'professional_summary': rule_suggestions['professional_summary'] + (gemini_suggestions.get('professional_summary', []) if gemini_suggestions else []),
                    'work_experience': rule_suggestions['work_experience'] + (gemini_suggestions.get('work_experience', []) if gemini_suggestions else []),
                    'education': rule_suggestions['education'] + (gemini_suggestions.get('education', []) if gemini_suggestions else []),
                    'skills': rule_suggestions['skills'] + (gemini_suggestions.get('skills', []) if gemini_suggestions else [])
                }
                cv.extracted_data = {
                    'skills': skills,
                    'suggestions': combined_suggestions
                }
                cv.save()
                return redirect('cv_results', cv_id=cv.id)
            except Exception as e:
                return render(request, 'results.html', {'cv': cv, 'error': f'Error saving modifications: {str(e)}'})
    return render(request, 'results.html', {'cv': cv})


@login_required
def generate_optimized_cv(request, cv_id):
    cv = get_object_or_404(CV, id=cv_id, user=request.user)

    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            sections = data.get('sections', {})
            ats_feedback = data.get('ats_feedback', [])

            # Combine sections into optimized text
            optimized_text = "\n\n".join(
                f"{section.upper()}\n{content}"
                for section, content in sections.items()
            )

            # Add ATS feedback as comments
            optimized_text += "\n\n/* ATS OPTIMIZATION TIPS:\n" + "\n".join(ats_feedback) + "\n*/"

            # Generate PDF
            buffer = BytesIO()
            p = canvas.Canvas(buffer, pagesize=letter)
            y = 750
            for line in optimized_text.split('\n'):
                p.drawString(50, y, line[:100])
                y -= 15
                if y < 50:
                    p.showPage()
                    y = 750
            p.save()
            buffer.seek(0)

            # Save modified file
            cv.modified_file.save(f'optimized_cv_{cv.id}.pdf', File(buffer))
            cv.save()

            return JsonResponse({'success': True})

        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=500)

    return JsonResponse({'success': False, 'error': 'Invalid request'}, status=400)

@login_required
def download_modified_cv(request, cv_id):
    cv = get_object_or_404(CV, id=cv_id, user=request.user)
    if not cv.modified_file:
        return HttpResponse('No modified CV available', status=404)
    return FileResponse(cv.modified_file, as_attachment=True, filename=f'modified_cv_{cv.id}.pdf')

@login_required
def modify_cv_section(request, cv_id):
    cv = get_object_or_404(CV, id=cv_id, user=request.user)
    if request.method == 'POST':
        section_name = request.POST.get('section_name')
        section_content = request.POST.get('section_content')
        cv.extracted_text = section_content
        cv.save()
        return redirect('cv_results', cv_id=cv.id)
    return redirect('cv_results', cv_id=cv.id)

@login_required
def preview_cv(request, cv_id):
    cv = get_object_or_404(CV, id=cv_id, user=request.user)
    return render(request, 'preview.html', {'cv': cv})

@login_required
def generate_preview_pdf(request, cv_id):
    cv = get_object_or_404(CV, id=cv_id, user=request.user)
    if request.method == 'POST':
        text = request.POST.get('text', cv.extracted_text)
    else:
        text = cv.extracted_text
    try:
        buffer = BytesIO()
        p = canvas.Canvas(buffer, pagesize=letter)
        y = 750
        for line in text.split('\n'):
            p.drawString(50, y, line[:100])
            y -= 15
            if y < 50:
                p.showPage()
                y = 750
        p.save()
        buffer.seek(0)
        return FileResponse(buffer, content_type='application/pdf')
    except Exception as e:
        logger.error(f"Error generating preview PDF: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)
