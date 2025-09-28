# test_pubmedbert.py

from transformers import pipeline
import time

# This is the official name of a popular PubMedBERT model on the Hugging Face Hub
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

# --- 1. Expanded Knowledge Base with ~50 Diseases ---
knowledge_base = {
    # Infectious Diseases
    "measles_fact_sheet.txt": (
        "Measles is a childhood infection caused by a virus. Once quite common, "
        "measles can now almost always be prevented with a vaccine. Measles signs "
        "and symptoms appear around 10 to 14 days after exposure to the virus. "
        "Signs and symptoms of measles typically include fever, dry cough, runny nose, "
        "sore throat and inflamed eyes (conjunctivitis)."
    ),
    "cholera_fact_sheet.txt": (
        "Cholera is a bacterial disease usually spread through contaminated water. "
        "Cholera causes severe diarrhea and dehydration. Left untreated, cholera can "
        "be fatal in a matter of hours, even in previously healthy people. Modern "
        "sewage and water treatment have virtually eliminated cholera in industrialized countries."
    ),
    "influenza_fact_sheet.txt": (
        "Influenza, commonly known as the flu, is a contagious respiratory illness caused by influenza viruses. "
        "It can cause mild to severe illness. Symptoms of the flu often come on suddenly and can include "
        "fever or feeling feverish/chills, cough, sore throat, runny or stuffy nose, muscle or body aches, "
        "headaches, and fatigue. Annual vaccination is the most effective way to prevent influenza."
    ),
    "tuberculosis_fact_sheet.txt": (
        "Tuberculosis (TB) is a potentially serious infectious disease that mainly affects the lungs. "
        "The bacteria that cause tuberculosis are spread from one person to another through tiny droplets "
        "released into the air via coughs and sneezes. Symptoms of active TB include a persistent cough that lasts "
        "more than three weeks, coughing up blood or mucus, chest pain, unintentional weight loss, fatigue, "
        "fever, and night sweats."
    ),
    "malaria_fact_sheet.txt": (
        "Malaria is a life-threatening disease caused by parasites that are transmitted to people through "
        "the bites of infected female Anopheles mosquitoes. It is preventable and curable. Symptoms usually "
        "appear 10–15 days after the infective mosquito bite and include fever, headache, and chills. "
        "If not treated within 24 hours, P. falciparum malaria can progress to severe illness, often leading to death."
    ),
    "covid-19_fact_sheet.txt": (
        "Coronavirus disease (COVID-19) is an infectious disease caused by the SARS-CoV-2 virus. "
        "Most people infected with the virus will experience mild to moderate respiratory illness and recover "
        "without requiring special treatment. Common symptoms include fever, cough, tiredness, and loss of taste or smell. "
        "Vaccines and booster shots are highly effective at preventing severe illness."
    ),
    "hiv_aids_fact_sheet.txt": (
        "HIV (Human Immunodeficiency Virus) is a virus that attacks the body's immune system. "
        "If HIV is not treated, it can lead to AIDS (Acquired Immunodeficiency Syndrome). "
        "HIV is spread through contact with certain body fluids from a person with HIV. "
        "There is currently no effective cure, but with proper medical care, HIV can be controlled."
    ),
    "hepatitis_b_fact_sheet.txt": (
        "Hepatitis B is a serious liver infection caused by the hepatitis B virus (HBV). "
        "It can cause both acute and chronic disease. The virus is transmitted through contact "
        "with the blood or other body fluids of an infected person. Vaccination is the most effective prevention."
    ),
    "hepatitis_c_fact_sheet.txt": (
        "Hepatitis C is a liver infection caused by the hepatitis C virus (HCV). "
        "Hepatitis C is spread through contact with blood from an infected person. "
        "Today, most people become infected with hepatitis C by sharing needles or other equipment to inject drugs."
    ),
    "dengue_fever_fact_sheet.txt": (
        "Dengue fever is a mosquito-borne tropical disease caused by the dengue virus. "
        "Symptoms typically begin three to fourteen days after infection and include high fever, "
        "headache, vomiting, muscle and joint pains, and a characteristic skin rash."
    ),
    "zika_virus_fact_sheet.txt": (
        "Zika virus is a mosquito-borne virus that can cause birth defects in babies born to infected mothers. "
        "Symptoms are generally mild and include fever, rash, conjunctivitis, muscle and joint pain."
    ),
    "ebola_fact_sheet.txt": (
        "Ebola virus disease is a severe, often fatal illness in humans. "
        "The virus is transmitted to people from wild animals and spreads in the human population "
        "through human-to-human transmission. Average case fatality rate is around 50%."
    ),
    "lyme_disease_fact_sheet.txt": (
        "Lyme disease is an infectious disease caused by Borrelia bacteria spread by ticks. "
        "The most common sign of infection is an expanding area of redness on the skin that appears "
        "at the site of the tick bite about a week after it occurred."
    ),
    "syphilis_fact_sheet.txt": (
        "Syphilis is a bacterial infection usually spread by sexual contact. "
        "The disease starts as a painless sore typically on the genitals, rectum or mouth. "
        "Syphilis spreads from person to person via skin or mucous membrane contact with these sores."
    ),
    "gonorrhea_fact_sheet.txt": (
        "Gonorrhea is a sexually transmitted infection caused by the bacterium Neisseria gonorrhoeae. "
        "Many people with gonorrhea develop no symptoms. When present, symptoms may include burning with urination, "
        "discharge from the penis or vaginal discharge."
    ),
    
    # Chronic Diseases
    "diabetes_fact_sheet.txt": (
        "Diabetes mellitus is a chronic metabolic disorder characterized by high blood sugar levels. "
        "Type 2 diabetes is the most common type and is often linked to obesity and a lack of exercise. "
        "Symptoms include increased thirst, frequent urination, unexplained weight loss, fatigue, and blurred vision. "
        "Management involves lifestyle changes, oral medications, or insulin."
    ),
    "hypertension_fact_sheet.txt": (
        "Hypertension, also known as high blood pressure, is a condition in which the force of the blood against "
        "the artery walls is too high. It is often called the 'silent killer' because it may have no symptoms. "
        "Uncontrolled high blood pressure increases the risk of serious health problems, including heart attack and stroke. "
        "It can be managed with lifestyle changes and medication."
    ),
    "asthma_fact_sheet.txt": (
        "Asthma is a condition in which your airways narrow and swell and may produce extra mucus. "
        "This can make breathing difficult and trigger coughing, a whistling sound (wheezing) when you breathe out, and shortness of breath. "
        "Asthma can't be cured, but its symptoms can be controlled."
    ),
    "copd_fact_sheet.txt": (
        "Chronic Obstructive Pulmonary Disease (COPD) is a chronic inflammatory lung disease that causes obstructed airflow from the lungs. "
        "Symptoms include breathing difficulty, cough, mucus production and wheezing. It's typically caused by long-term exposure to irritating gases or particulate matter, most often from cigarette smoke."
    ),
    "arthritis_fact_sheet.txt": (
        "Arthritis is the swelling and tenderness of one or more joints. The main symptoms of arthritis are joint pain and stiffness, "
        "which typically worsen with age. The two most common types are osteoarthritis and rheumatoid arthritis."
    ),
    "osteoporosis_fact_sheet.txt": (
        "Osteoporosis is a bone disease that occurs when the body loses too much bone, makes too little bone, or both. "
        "As a result, bones become weak and may break from a fall or, in serious cases, from sneezing or minor bumps."
    ),
    "alzheimer's_fact_sheet.txt": (
        "Alzheimer's disease is a progressive neurologic disorder that causes the brain to shrink and brain cells to die. "
        "It's the most common cause of dementia — a continuous decline in thinking, behavioral and social skills that affects a person's ability to function independently."
    ),
    "parkinson's_fact_sheet.txt": (
        "Parkinson's disease is a progressive nervous system disorder that affects movement. "
        "Symptoms start gradually, sometimes starting with a barely noticeable tremor in just one hand. "
        "Tremors are common, but the disorder also commonly causes stiffness or slowing of movement."
    ),
    "multiple_sclerosis_fact_sheet.txt": (
        "Multiple sclerosis (MS) is a potentially disabling disease of the brain and spinal cord. "
        "In MS, the immune system attacks the protective sheath that covers nerve fibers and causes communication problems "
        "between your brain and the rest of your body."
    ),
    "epilepsy_fact_sheet.txt": (
        "Epilepsy is a central nervous system (neurological) disorder in which brain activity becomes abnormal, "
        "causing seizures or periods of unusual behavior, sensations, and sometimes loss of awareness."
    ),
    "migraine_fact_sheet.txt": (
        "Migraine is a neurological condition that can cause multiple symptoms. It's frequently characterized by intense, "
        "debilitating headaches. Symptoms may include nausea, vomiting, difficulty speaking, numbness or tingling, and sensitivity to light and sound."
    ),
    
    # Cardiovascular Diseases
    "coronary_artery_disease_fact_sheet.txt": (
        "Coronary artery disease develops when the major blood vessels that supply your heart with blood, oxygen and nutrients become damaged or diseased. "
        "Cholesterol-containing deposits (plaques) in your arteries and inflammation are usually to blame for coronary artery disease."
    ),
    "heart_failure_fact_sheet.txt": (
        "Heart failure occurs when the heart muscle doesn't pump blood as well as it should. "
        "Certain conditions, such as narrowed arteries in your heart or high blood pressure, gradually leave your heart too weak or stiff to fill and pump efficiently."
    ),
    "stroke_fact_sheet.txt": (
        "A stroke occurs when the blood supply to part of your brain is interrupted or reduced, preventing brain tissue from getting oxygen and nutrients. "
        "Brain cells begin to die in minutes. A stroke is a medical emergency, and prompt treatment is crucial."
    ),
    "atrial_fibrillation_fact_sheet.txt": (
        "Atrial fibrillation is an irregular and often very rapid heart rhythm that can lead to blood clots in the heart. "
        "It increases the risk of stroke, heart failure and other heart-related complications."
    ),
    
    # Cancer Types
    "lung_cancer_fact_sheet.txt": (
        "Lung cancer is a type of cancer that begins in the lungs. Your lungs are two spongy organs in your chest that take in oxygen when you inhale and release carbon dioxide when you exhale. "
        "Lung cancer is the leading cause of cancer deaths worldwide, with smoking being the primary risk factor."
    ),
    "breast_cancer_fact_sheet.txt": (
        "Breast cancer is cancer that forms in the cells of the breasts. After skin cancer, breast cancer is the most common cancer diagnosed in women. "
        "Breast cancer can occur in both men and women, but it's far more common in women."
    ),
    "prostate_cancer_fact_sheet.txt": (
        "Prostate cancer is cancer that occurs in the prostate — a small walnut-shaped gland in men that produces the seminal fluid that nourishes and transports sperm. "
        "Prostate cancer is one of the most common types of cancer in men."
    ),
    "colorectal_cancer_fact_sheet.txt": (
        "Colorectal cancer is cancer that starts in the colon or the rectum. These cancers can also be called colon cancer or rectal cancer, "
        "depending on where they start. Colorectal cancer is the third most common cancer diagnosed in both men and women."
    ),
    "skin_cancer_fact_sheet.txt": (
        "Skin cancer — the abnormal growth of skin cells — most often develops on skin exposed to the sun. "
        "But this common form of cancer can also occur on areas of your skin not ordinarily exposed to sunlight. "
        "The three main types of skin cancer are basal cell carcinoma, squamous cell carcinoma and melanoma."
    ),
    
    # Autoimmune Diseases
    "lupus_fact_sheet.txt": (
        "Lupus is a systemic autoimmune disease that occurs when your body's immune system attacks your own tissues and organs. "
        "Inflammation caused by lupus can affect many different body systems including your joints, skin, kidneys, blood cells, brain, heart and lungs."
    ),
    "rheumatoid_arthritis_fact_sheet.txt": (
        "Rheumatoid arthritis is a chronic inflammatory disorder that can affect more than just your joints. "
        "In some people, the condition can damage a wide variety of body systems, including the skin, eyes, lungs, heart and blood vessels."
    ),
    "celiac_disease_fact_sheet.txt": (
        "Celiac disease is an immune reaction to eating gluten, a protein found in wheat, barley and rye. "
        "Over time, this reaction damages your small intestine's lining and prevents absorption of some nutrients."
    ),
    "crohn's_disease_fact_sheet.txt": (
        "Crohn's disease is a type of inflammatory bowel disease that may affect any part of the gastrointestinal tract from mouth to anus. "
        "Symptoms often include abdominal pain, diarrhea, fever, and weight loss."
    ),
    "ulcerative_colitis_fact_sheet.txt": (
        "Ulcerative colitis is an inflammatory bowel disease that causes inflammation and ulcers in your digestive tract. "
        "It affects the innermost lining of your large intestine and rectum."
    ),
    
    # Other Important Diseases
    "chronic_kidney_disease_fact_sheet.txt": (
        "Chronic kidney disease includes conditions that damage your kidneys and decrease their ability to keep you healthy. "
        "If kidney disease worsens, wastes can build to high levels in your blood and make you feel sick."
    ),
    "cirrhosis_fact_sheet.txt": (
        "Cirrhosis is late stage scarring of the liver caused by many forms of liver diseases and conditions, such as hepatitis and chronic alcoholism. "
        "Each time your liver is injured, it tries to repair itself, resulting in scar tissue formation."
    ),
    "osteoarthritis_fact_sheet.txt": (
        "Osteoarthritis is the most common form of arthritis, affecting millions of people worldwide. "
        "It occurs when the protective cartilage that cushions the ends of your bones wears down over time."
    ),
    "gout_fact_sheet.txt": (
        "Gout is a common and complex form of arthritis that can affect anyone. It's characterized by sudden, severe attacks of pain, "
        "swelling, redness and tenderness in one or more joints, most often in the big toe."
    ),
    "pneumonia_fact_sheet.txt": (
        "Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, "
        "causing cough with phlegm or pus, fever, chills, and difficulty breathing."
    ),
    "bronchitis_fact_sheet.txt": (
        "Bronchitis is an inflammation of the lining of your bronchial tubes, which carry air to and from your lungs. "
        "People who have bronchitis often cough up thickened mucus, which can be discolored."
    ),
    "sinusitis_fact_sheet.txt": (
        "Sinusitis is an inflammation of the tissues lining the sinuses. Healthy sinuses are filled with air. "
        "But when they become blocked and filled with fluid, germs can grow and cause an infection."
    ),
    "cataracts_fact_sheet.txt": (
        "A cataract is a clouding of the normally clear lens of your eye. For people who have cataracts, seeing through cloudy lenses is a bit like looking through a frosty or fogged-up window."
    ),
    "glaucoma_fact_sheet.txt": (
        "Glaucoma is a group of eye conditions that damage the optic nerve, which is vital for good vision. "
        "This damage is often caused by an abnormally high pressure in your eye."
    ),
}

# --- 2. Initialize the Question-Answering Pipeline ---
print("Initializing the PubMedBERT Question-Answering model...")
print("(This may take a few minutes and download the model if you're running it for the first time)...")
start_time = time.time()
qa_pipeline = pipeline("question-answering", model=MODEL_NAME)
end_time = time.time()
print(f"Model loaded successfully in {end_time - start_time:.2f} seconds.")

# --- 3. Ask a Question ---
question = "What spreads tuberculosis?"
print(f"\nSearching for an answer to the question: '{question}'")

# --- 4. Search the Knowledge Base for the Answer ---
best_answer = None
best_score = 0
best_context = ""

for doc_name, context in knowledge_base.items():
    print(f"-> Reading '{doc_name}'...")
    
    result = qa_pipeline(question=question, context=context)
    
    if result['score'] > best_score:
        best_answer = result['answer']
        best_score = result['score']
        best_context = doc_name

print("\n--- Search Complete ---")

if best_answer:
    print(f"Best Answer Found: '{best_answer}'")
    print(f"Confidence Score: {best_score:.4f}")
    print(f"Source Document: '{best_context}'")
else:
    print("Sorry, I could not find an answer in the provided documents.")