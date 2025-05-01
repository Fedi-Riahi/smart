from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.conf import settings
from .models import CV
import pdfplumber
import spacy
import os

# Load the trained NER model
nlp = spacy.load(os.path.join(settings.BASE_DIR, 'ner_model'))

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
                    # Create CV instance
                    cv = CV.objects.create(user=request.user, file=cv_file)

                    # Extract text
                    text = ''
                    with pdfplumber.open(cv_file) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text += page_text
                    if not text.strip():
                        error = 'No text extracted from the PDF.'
                        cv.delete()  # Clean up if extraction fails
                    else:
                        # Extract skills
                        doc = nlp(text)
                        skills = [ent.text for ent in doc.ents if ent.label_ == 'SKILL']

                        # Generate suggestions
                        suggestions = []
                        if len(skills) < 3:
                            suggestions.append("Consider adding more technical skills to strengthen your CV.")

                        # Save extracted data
                        cv.extracted_data = {'skills': skills, 'suggestions': suggestions}
                        cv.save()
                        return redirect('cv_results', cv_id=cv.id)
                except Exception as e:
                    error = f'Error processing file: {str(e)}'
                    # Clean up CV instance if created
                    if 'cv' in locals():
                        cv.delete()

    return render(request, 'upload.html', {'error': error})

@login_required
def cv_results(request, cv_id):
    cv = get_object_or_404(CV, id=cv_id, user=request.user)
    return render(request, 'results.html', {'cv': cv})
