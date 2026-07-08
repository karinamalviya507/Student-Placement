# 🎓 Student Placement Prediction — Django + ML

A full-stack web application that predicts campus placement outcomes using 4 Machine Learning algorithms.

---

## 📁 Project Structure

```
student_placement/
├── manage.py
├── train_model.py              ← Run this first
├── student_placement_synthetic.csv  ← Place your dataset here
├── ml_models/
│   ├── models.pkl              ← Trained ML models (auto-generated)
│   ├── encoders.pkl            ← Label encoders (auto-generated)
│   └── results.json            ← Accuracy results (auto-generated)
├── placement_project/          ← Django project settings
│   ├── settings.py
│   └── urls.py
└── placement_app/              ← Main Django app
    ├── views.py
    ├── urls.py
    ├── templates/
    │   ├── landing.html
    │   ├── register.html
    │   ├── login.html
    │   ├── home.html           ← Prediction Form
    │   ├── result.html         ← Prediction Result
    │   └── compare.html        ← Model Comparison
    └── templatetags/
        └── dict_filters.py
```

---

## ⚙️ Setup & Run

### 1. Install dependencies
```bash
pip install django scikit-learn pandas numpy
```

### 2. Place the dataset
Copy `student_placement_synthetic.csv` into the project root folder.

### 3. Train the ML models
```bash
python train_model.py
```
This creates the `ml_models/` folder with trained models.

### 4. Run migrations
```bash
python manage.py migrate
```

### 5. Start the server
```bash
python manage.py runserver
```

### 6. Open in browser
Visit: **http://127.0.0.1:8000/**

---

## 🤖 Algorithms Used

| Algorithm | Type | Accuracy |
|---|---|---|
| Logistic Regression | Linear | 69.84% ⭐ Best |
| Random Forest | Ensemble | 69.36% |
| KNN | Instance-based | 63.85% |
| Decision Tree | Tree-based | 59.12% |

---

## 📊 Dataset Features (16 inputs)

**Academic:** Branch, College Tier, CGPA, Backlogs  
**Technical:** Coding Skills, DSA Score, ML Knowledge, System Design  
**Performance:** Aptitude Score, Communication Skills  
**Activities:** Internships, Projects, Certifications, Hackathons, Open Source, Extracurriculars

**Target:** `placement_status` (1 = Placed, 0 = Not Placed)

---

## 🔧 Pages

| URL | Description |
|---|---|
| `/` | Landing page |
| `/register/` | User registration |
| `/login/` | User login |
| `/home/` | Prediction form (login required) |
| `/predict/` | Handles prediction (POST) |
| `/compare/` | Model accuracy comparison |
| `/logout/` | Logout |
