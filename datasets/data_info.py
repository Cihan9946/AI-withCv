import pandas as pd
import re

# CSV'yi oku
df = pd.read_csv('combined_dataset.csv')  # Dosya adınızı buraya yazın

# Yardımcı fonksiyonlar
def count_job_experiences(text):
    headers = ['work experience', 'experience', 'employment', 'professional experience']
    count = 0
    lines = text.lower().split('\n')
    for i, line in enumerate(lines):
        if any(header in line for header in headers):
            for j in range(i+1, min(i+10, len(lines))):
                if re.search(r'\b\d{4}\b', lines[j]):
                    count += 1
    return count

def count_educations(text):
    matches = re.findall(r'\b(university|college|b\.a|b\.sc|m\.a|m\.sc|ph\.d|education)\b', text.lower())
    return len(matches)

def count_skills(text):
    skill_section = re.search(r'(skills|technical skills)(.*?)\n\n', text.lower(), re.DOTALL)
    if skill_section:
        content = skill_section.group(2)
        return len(re.split(r',|\n', content.strip()))
    return 0

def count_certifications(text):
    matches = re.findall(r'\b(certification|certifications|certificate|certified)\b', text.lower())
    return len(matches)

def estimate_experience_years(text):
    years = re.findall(r'\b(19|20)\d{2}\b', text)
    years = list(map(int, years))
    if len(years) >= 2:
        return max(years) - min(years)
    return 0

def count_languages(text):
    match = re.search(r'(languages|fluent in)(.*)', text.lower())
    if match:
        line = match.group(2)
        return len(re.findall(r'[a-zA-Z]+', line))
    return 0

def count_projects(text):
    matches = re.findall(r'\b(project|projects|project management|project experience)\b', text.lower())
    return len(matches)

def has_contact_info(text):
    email = re.search(r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b', text)
    phone = re.search(r'(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}', text)
    address = re.search(r'\d{1,5} [\w\s]+,? [\w\s]+', text)
    return int(bool(email or phone or address))

# Boş değerleri engelle
df['text'] = df['text'].fillna('')

# Yeni sütunları oluştur
df['num_jobs'] = df['text'].apply(count_job_experiences)
df['num_educations'] = df['text'].apply(count_educations)
df['num_skills'] = df['text'].apply(count_skills)
df['num_certifications'] = df['text'].apply(count_certifications)
df['years_experience'] = df['text'].apply(estimate_experience_years)
df['num_languages'] = df['text'].apply(count_languages)
df['num_projects'] = df['text'].apply(count_projects)
df['has_contact_info'] = df['text'].apply(has_contact_info)

# Yeni CSV olarak kaydet
df.to_csv('final_Dataset.csv', index=False)
