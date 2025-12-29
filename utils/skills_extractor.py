import re
import json
from typing import List, Dict

class SkillsExtractor:
    def __init__(self):
        self.skills_db = {
            'programming': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'swift', 'kotlin'],
            'web': ['html', 'css', 'react', 'angular', 'vue', 'django', 'flask', 'express', 'node.js'],
            'databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'sqlite'],
            'cloud': ['aws', 'azure', 'google cloud', 'docker', 'kubernetes', 'terraform', 'jenkins'],
            'tools': ['git', 'github', 'gitlab', 'jira', 'confluence', 'figma', 'postman'],
            'data_science': ['pandas', 'numpy', 'tensorflow', 'pytorch', 'scikit-learn', 'tableau', 'power bi'],
            'soft_skills': ['leadership', 'communication', 'teamwork', 'problem solving', 'critical thinking']
        }
    
    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract skills from text"""
        text_lower = text.lower()
        found_skills = {}
        
        for category, skills in self.skills_db.items():
            found = [skill for skill in skills if skill in text_lower]
            if found:
                found_skills[category] = found
        
        return found_skills
    
    def compare_skills(self, resume_skills: Dict, job_skills: Dict) -> Dict:
        """Compare skills between resume and job"""
        missing = {}
        
        for category in job_skills:
            if category in resume_skills:
                missing[category] = [
                    skill for skill in job_skills[category] 
                    if skill not in resume_skills[category]
                ]
            else:
                missing[category] = job_skills[category]
        
        return missing