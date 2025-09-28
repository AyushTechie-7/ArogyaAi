from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import requests
import json

class ActionPubMedQA(Action):
    def name(self) -> Text:
        return "action_pubmed_qa"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        question = tracker.latest_message['text']
        print(f"Processing question: {question}")
        
        # Simple disease knowledge base
        disease_info = {
            "measles": {
                "symptoms": "fever, dry cough, runny nose, sore throat, inflamed eyes (conjunctivitis), skin rash",
                "transmission": "airborne through coughs and sneezes",
                "prevention": "MMR vaccine",
                "treatment": "rest, fluids, fever reducers"
            },
            "tuberculosis": {
                "symptoms": "persistent cough (more than 3 weeks), coughing up blood, chest pain, weight loss, fatigue, fever, night sweats",
                "transmission": "airborne droplets from coughs or sneezes",
                "prevention": "BCG vaccine, good ventilation",
                "treatment": "antibiotics for 6-9 months"
            },
            "cholera": {
                "symptoms": "severe diarrhea, dehydration, vomiting, muscle cramps",
                "transmission": "contaminated water or food",
                "prevention": "clean water, proper sanitation, oral cholera vaccine",
                "treatment": "oral rehydration solution, antibiotics"
            },
            "influenza": {
                "symptoms": "fever, chills, cough, sore throat, runny nose, muscle aches, headaches, fatigue",
                "transmission": "airborne droplets, direct contact",
                "prevention": "annual flu vaccine, hand hygiene",
                "treatment": "rest, fluids, antiviral medications"
            },
            "covid-19": {
                "symptoms": "fever, cough, tiredness, loss of taste or smell, difficulty breathing",
                "transmission": "airborne droplets, close contact",
                "prevention": "vaccination, masks, social distancing",
                "treatment": "symptomatic care, antiviral medications"
            },
            "malaria": {
                "symptoms": "fever, headache, chills, sweating, nausea",
                "transmission": "mosquito bites (Anopheles mosquito)",
                "prevention": "mosquito nets, insect repellent, antimalarial drugs",
                "treatment": "antimalarial medications"
            },
            "hiv": {
                "symptoms": "fever, fatigue, swollen lymph nodes, weight loss",
                "transmission": "unprotected sex, contaminated needles, mother to child",
                "prevention": "condoms, pre-exposure prophylaxis (PrEP), sterile needles",
                "treatment": "antiretroviral therapy (ART)"
            },
            "diabetes": {
                "symptoms": "increased thirst, frequent urination, hunger, fatigue, blurred vision",
                "types": "Type 1 (autoimmune), Type 2 (insulin resistance)",
                "prevention": "healthy diet, exercise, weight management (for Type 2)",
                "treatment": "insulin, oral medications, lifestyle changes"
            },
            "hypertension": {
                "symptoms": "often none (silent), sometimes headaches, shortness of breath, nosebleeds",
                "causes": "high salt intake, obesity, stress, genetics",
                "prevention": "healthy diet, exercise, weight control, limit alcohol",
                "treatment": "lifestyle changes, medications"
            },
            "asthma": {
                "symptoms": "wheezing, shortness of breath, chest tightness, coughing",
                "triggers": "allergens, exercise, cold air, respiratory infections",
                "prevention": "avoid triggers, use inhalers as prescribed",
                "treatment": "bronchodilators, corticosteroids"
            }
        }

        # Extract disease name from question
        question_lower = question.lower()
        found_disease = None
        
        for disease in disease_info.keys():
            if disease in question_lower:
                found_disease = disease
                break
        
        if not found_disease:
            # Try to find disease by synonyms
            disease_synonyms = {
                "flu": "influenza",
                "tb": "tuberculosis",
                "high blood pressure": "hypertension",
                "blood pressure": "hypertension",
                "aids": "hiv",
                "covid": "covid-19",
                "coronavirus": "covid-19"
            }
            
            for synonym, main_disease in disease_synonyms.items():
                if synonym in question_lower:
                    found_disease = main_disease
                    break

        if found_disease:
            info = disease_info[found_disease]
            
            # Answer based on question type
            if "symptom" in question_lower:
                answer = f"Symptoms of {found_disease} include: {info['symptoms']}"
            elif "spread" in question_lower or "transmit" in question_lower:
                answer = f"{found_disease.capitalize()} is spread through: {info.get('transmission', info.get('transmission', 'contact with infected individuals'))}"
            elif "prevent" in question_lower or "avoid" in question_lower:
                answer = f"To prevent {found_disease}: {info['prevention']}"
            elif "treat" in question_lower or "cure" in question_lower:
                answer = f"Treatment for {found_disease} includes: {info['treatment']}"
            elif "cause" in question_lower:
                answer = f"Causes of {found_disease}: {info.get('causes', info.get('transmission', 'various factors including pathogens or lifestyle factors'))}"
            else:
                # General information
                answer = f"About {found_disease}:\n"
                answer += f"- Symptoms: {info['symptoms']}\n"
                if 'transmission' in info:
                    answer += f"- Transmission: {info['transmission']}\n"
                answer += f"- Prevention: {info['prevention']}\n"
                answer += f"- Treatment: {info['treatment']}"
                
            dispatcher.utter_message(text=answer)
        else:
            dispatcher.utter_message(text="I can provide information about various diseases like measles, tuberculosis, cholera, influenza, COVID-19, malaria, HIV, diabetes, hypertension, and asthma. Please ask about a specific disease or its symptoms, transmission, prevention, or treatment.")

        return []