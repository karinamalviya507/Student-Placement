import os
import json
import pickle
import numpy as np
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib import messages


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'ml_models', 'models.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'ml_models', 'encoders.pkl')
RESULTS_PATH = os.path.join(BASE_DIR, 'ml_models', 'results.json')


with open(MODEL_PATH, 'rb') as f:
    MODELS = pickle.load(f)

with open(ENCODER_PATH, 'rb') as f:
    ENCODERS = pickle.load(f)

with open(RESULTS_PATH) as f:
    RESULTS_DATA = json.load(f)

FEATURE_COLUMNS = [
    'branch', 'college_tier', 'cgpa', 'backlogs',
    'coding_skills', 'dsa_score', 'aptitude_score', 'communication_skills',
    'ml_knowledge', 'system_design', 'internships', 'projects_count',
    'certifications', 'hackathons', 'open_source_contributions', 'extracurriculars'
]

BRANCH_CHOICES = ['CSE', 'ECE', 'EE', 'ME', 'CE', 'IT', 'Chemical']
TIER_CHOICES = ['Tier-1', 'Tier-2', 'Tier-3']



def landing(request):
    if request.user.is_authenticated:
        return redirect('home')
    return render(request, 'landing.html')


def register_view(request):
    if request.user.is_authenticated:
        return redirect('home')
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        email = request.POST.get('email', '').strip()
        password = request.POST.get('password', '')
        confirm = request.POST.get('confirm_password', '')

        if not username or not password:
            messages.error(request, 'Username and password are required.')
        elif password != confirm:
            messages.error(request, 'Passwords do not match.')
        elif User.objects.filter(username=username).exists():
            messages.error(request, 'Username already taken.')
        else:
            User.objects.create_user(username=username, email=email, password=password)
            messages.success(request, 'Account created! Please login.')
            return redirect('login')
    return render(request, 'register.html')


def login_view(request):
    if request.user.is_authenticated:
        return redirect('home')
    if request.method == 'POST':
        username = request.POST.get('username', '')
        password = request.POST.get('password', '')
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect('home')
        else:
            messages.error(request, 'Invalid username or password.')
    return render(request, 'login.html')


def logout_view(request):
    logout(request)
    return redirect('login')



@login_required
def home(request):
    return render(request, 'home.html', {
        'branch_choices': BRANCH_CHOICES,
        'tier_choices': TIER_CHOICES,
    })


@login_required
def predict(request):
    if request.method != 'POST':
        return redirect('home')

    try:
        branch = request.POST.get('branch')
        college_tier = request.POST.get('college_tier')
        cgpa = float(request.POST.get('cgpa'))
        backlogs = int(request.POST.get('backlogs'))
        coding_skills = float(request.POST.get('coding_skills'))
        dsa_score = float(request.POST.get('dsa_score'))
        aptitude_score = float(request.POST.get('aptitude_score'))
        communication_skills = float(request.POST.get('communication_skills'))
        ml_knowledge = float(request.POST.get('ml_knowledge'))
        system_design = float(request.POST.get('system_design'))
        internships = int(request.POST.get('internships'))
        projects_count = int(request.POST.get('projects_count'))
        certifications = int(request.POST.get('certifications'))
        hackathons = int(request.POST.get('hackathons'))
        open_source = int(request.POST.get('open_source_contributions'))
        extracurriculars = int(request.POST.get('extracurriculars'))

        branch_enc = ENCODERS['branch'].transform([branch])[0]
        tier_enc = ENCODERS['college_tier'].transform([college_tier])[0]

        features = np.array([[
            branch_enc, tier_enc, cgpa, backlogs,
            coding_skills, dsa_score, aptitude_score, communication_skills,
            ml_knowledge, system_design, internships, projects_count,
            certifications, hackathons, open_source, extracurriculars
        ]])

        predictions = {}
        confidences = {}
        for name, model in MODELS.items():
            pred = model.predict(features)[0]
            label = 'Placed ✅' if pred == 1 else 'Not Placed ❌'
            predictions[name] = label
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features)[0]
                conf = round(float(max(proba)) * 100, 2)
            else:
                conf = None
            confidences[name] = conf

        accuracies = RESULTS_DATA['accuracies']
        class_metrics = RESULTS_DATA.get('class_metrics', {})

        best_model = max(
            {k: v for k, v in confidences.items() if v is not None},
            key=lambda k: confidences[k]
        )

        best_pred = MODELS[best_model].predict(features)[0]
        final_result = 'Placed' if best_pred == 1 else 'Not Placed'
        is_placed = (best_pred == 1)

        outcome_key = 'placed' if is_placed else 'not_placed'
        per_class_acc = {}
        for name in MODELS:
            cm_data = class_metrics.get(name, {})
            per_class_acc[name] = cm_data.get(outcome_key + '_accuracy', accuracies.get(name))

        context = {
            'final_result': final_result,
            'is_placed': is_placed,
            'best_model': best_model,
            'predictions': predictions,
            'confidences': confidences,
            'accuracies': accuracies,
            'per_class_acc': per_class_acc,
            'outcome_label': 'Placed' if is_placed else 'Not Placed',
            'best_confidence': confidences.get(best_model),
            'input_data': {
                'Branch': branch,
                'College Tier': college_tier,
                'CGPA': cgpa,
                'Backlogs': backlogs,
                'Coding Skills': coding_skills,
                'DSA Score': dsa_score,
                'Aptitude Score': aptitude_score,
                'Communication': communication_skills,
                'ML Knowledge': ml_knowledge,
                'System Design': system_design,
                'Internships': internships,
                'Projects': projects_count,
                'Certifications': certifications,
                'Hackathons': hackathons,
                'Open Source': open_source,
                'Extracurriculars': extracurriculars,
            }
        }
        request.session['last_confidences'] = confidences
        request.session['last_predictions'] = {k: v.replace(' ✅','').replace(' ❌','') for k,v in predictions.items()}
        request.session['last_best_model'] = best_model
        request.session['last_result'] = final_result

        return render(request, 'result.html', context)

    except Exception as e:
        messages.error(request, f'Prediction error: {str(e)}')
        return redirect('home')


@login_required
def result(request):
    return redirect('home')


@login_required
def compare(request):
    import json as _json
    accuracies = RESULTS_DATA['accuracies']
    best_model = RESULTS_DATA['best']
    confusion_matrices = RESULTS_DATA.get('confusion_matrices', {})

    accuracies_with_width = {
        name: {'acc': acc, 'width': "{}%".format(acc)}
        for name, acc in accuracies.items()
    }
    last_confidences = request.session.get('last_confidences', {})
    last_predictions = request.session.get('last_predictions', {})
    last_best_model = request.session.get('last_best_model', '')
    last_result = request.session.get('last_result', '')

    return render(request, 'compare.html', {
        'accuracies': accuracies,
        'best_model': best_model,
        'confusion_matrices': confusion_matrices,
        'confusion_matrices_json': _json.dumps(confusion_matrices),
        'class_metrics': RESULTS_DATA.get('class_metrics', {}),
        'last_confidences': last_confidences,
        'last_predictions': last_predictions,
        'last_best_model': last_best_model,
        'last_result': last_result,
    })