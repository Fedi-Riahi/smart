from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from .models import CV
import pdfplumber
import spacy

@login_required
def upload_cv(request):
    if request.method == 'POST':
        cv_file = request.FILES['cv']
        cv = CV.objects.create(user=request.user, file=cv_file)
        # Extract text
        with pdfplumber.open(cv_file) as pdf:
            text = ''.join(page.extract_text() for page in pdf.pages if page.extract_text())
        # Extract skills
        nlp = spacy.load("ner_model")
        doc = nlp(text)
        skills = [ent.text for ent in doc.ents if ent.label_ == "SKILL"]
        suggestions = []
        if len(skills) < 3:
            suggestions.append("Consider adding more technical skills to strengthen your CV.")
        cv.extracted_data = {'skills': skills, 'suggestions': suggestions}
        cv.save()
        return redirect('cv_results', cv_id=cv.id)
    return render(request, 'upload.html')

@login_required
def cv_results(request, cv_id):
    cv = get_object_or_404(CV, id=cv_id, user=request.user)
    return render(request, 'results.html', {'cv': cv})
