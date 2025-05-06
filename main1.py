import os
import json
import re
from typing import Any, Dict
import random
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import openai
from openai import OpenAI

# Charger les variables d'environnement
load_dotenv()

api_key = os.getenv("API_KEY")
replicate_token = os.getenv("REPLICATE_API_TOKEN")
openai.api_key = os.getenv("OPENAI_API_KEY")

if replicate_token:
    os.environ["REPLICATE_API_TOKEN"] = replicate_token
else:
    print("⚠️  REPLICATE_API_TOKEN n'est pas défini dans le fichier .env")

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=api_key,
)

# Initialisation de FastAPI
app = FastAPI()

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèles d'entrée
class OffreInput(BaseModel):
    poste: str
    description: str
    typeTravail: str
    niveauExperience: str
    responsabilite: str
    experience: str

class PoidsTraitsInput(BaseModel):
    ouverture: int
    conscience: int
    extraversion: int
    agreabilite: int
    stabilite: int

@app.post("/generate-test", response_model=Dict[str, Any])
async def generate_test(
    offre: OffreInput = Body(...),
    poids: PoidsTraitsInput = Body(...)
) -> Dict[str, Any]:
    poids_traits = {
        "ouverture": poids.ouverture,
        "conscience": poids.conscience,
        "extraversion": poids.extraversion,
        "agreabilite": poids.agreabilite,
        "stabilite": poids.stabilite
    }

    prompt = fr"""
Tu es un psychologue expert en recrutement et un rédacteur de tests professionnels. Crée un test de personnalité basé sur le modèle des Big Five (ouverture, conscience, extraversion, agréabilité, stabilité émotionnelle), conçu pour évaluer la compatibilité d’un candidat avec l’offre suivante :

### Informations sur l’offre :
- Poste : {offre.poste}
- Description : {offre.description}
- Type de travail : {offre.typeTravail}
- Niveau d’expérience requis : {offre.niveauExperience}
- Responsabilités principales : {offre.responsabilite}
- Expérience professionnelle attendue : {offre.experience}

### Instructions spécifiques :
- Le test contiendra **au maximum 15 questions**, réparties comme suit :
  - {poids.ouverture} questions sur l’ouverture
  - {poids.conscience} sur la conscience
  - {poids.extraversion} sur l’extraversion
  - {poids.agreabilite} sur l’agréabilité
  - {poids.stabilite} sur la stabilité émotionnelle

- Chaque question doit :
  - Être **contextualisée dans des situations de travail réelles ou techniques** liées à l’offre.
  - Avoir une **formulation unique**, avec un **contexte professionnel distinct pour chaque question**.
  - Employer un **langage technique ou professionnel adapté au domaine du poste**.
  - Être rédigée sous forme de **QCM à 4 réponses** (avec scores de 1 à 5), où chaque option est formulée comme une **réponse comportementale concrète et nuancée**.
  - Chaque option doit représenter un **comportement ou une attitude spécifique** face à la situation décrite.
  - Les options doivent rester **cohérentes avec l’intention du trait de personnalité évalué**, tout en étant **distinctes et plausibles**.

- Ne répète pas les contextes d’une question à l’autre.
- Ne sors pas du format JSON suivant, sans balises ni explications :

[
    {{
        "trait": "conscience", 
        "question": "Lorsque je travaille sur plusieurs projets à échéance courte, je suis capable de hiérarchiser mes tâches efficacement.", 
        "options": [
            {{"text": "Je préfère attendre les instructions claires de mon supérieur avant de commencer.", "score": 1}},
            {{"text": "Je commence le travail mais demande des clarifications au fur et à mesure.", "score": 2}},
            {{"text": "Je prends l’initiative en me basant sur mes expériences précédentes.", "score": 4}},
            {{"text": "Je planifie et lance le projet de manière autonome en anticipant les obstacles.", "score": 5}}
        ]
    }},
    ...
]
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt.strip()}],
            max_tokens=3000,
            temperature=0.7
        )
    except Exception as e:
        print("Erreur OpenAI:", e)
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'appel à OpenAI: {str(e)}")

    content = response.choices[0].message.content
    print("🟡 Réponse brute GPT :", content[:500])  # Affiche les 500 premiers caractères

    # Nettoyage du contenu JSON
    cleaned_content = re.sub(r"^```json\s*|```$", "", content.strip(), flags=re.MULTILINE)

    try:
        questions = json.loads(cleaned_content)
    except json.JSONDecodeError as json_error:
        print("Erreur JSON:", json_error)
        print("Contenu reçu:", cleaned_content)
        return JSONResponse(
            status_code=502,
            content={
                "error": "La réponse de l'IA n'est pas un JSON valide.",
                "raw": cleaned_content,
                "json_error": str(json_error),
            }
        )

    # Vérification du format
    if not isinstance(questions, list) or not all(
        isinstance(q, dict) and 'trait' in q and 'question' in q and 'options' in q for q in questions
    ):
        print("Format JSON incorrect:", questions)
        return JSONResponse(
            status_code=400,
            content={"error": "Le format des questions n'est pas correct.", "raw": cleaned_content}
        )
    for q in questions:
        random.shuffle(q['options'])
    return {
        "questions": questions,
    }



class Offre(BaseModel):
    poste: str
    description: str
    niveauExperience: str
    niveauEtude: str
    responsabilite: str
    experience: str
    pays: str
    ville: str

class MatchingScoreRequest(BaseModel):
    cv: str
    offre: Offre

@app.post("/match-cv-offre")
async def match_cv_offre(data: MatchingScoreRequest) -> Dict[str, Any]:
    offre = data.offre

    prompt = f"""
    Tu es un assistant RH expert en recrutement. Ta tâche est d'analyser le niveau de correspondance entre un CV et une offre d'emploi.

    CV :
    {data.cv}

    Offre d'emploi :
    Poste recherché: {offre.poste}
    Description: {offre.description}
    Niveau d'expérience: {offre.niveauExperience}
    Niveau d’étude: {offre.niveauEtude}
    Responsabilités: {offre.responsabilite}
    Expérience demandée: {offre.experience}
    Pays: {offre.pays}
    Ville: {offre.ville}

    Instructions obligatoires :
    1. Analyse si le poste du candidat correspond globalement au poste recherché, même si les mots sont différents.
       - Si les domaines sont totalement différents (exemple : comptable vs développeur), arrête l'analyse immédiatement.
       - Dans ce cas, donne un score de 0 et une explication rapide sans analyser les autres critères.
    2. Si le poste est similaire ou dans le même domaine, continue l'analyse :
       - Compare les expériences du candidat et les responsabilités demandées
       - Compare le niveau d’étude et d’expérience
       - Analyse les compétences techniques et comportementales

    Réponds uniquement au format JSON suivant :
    {{
        "score": 87,
        "evaluation": "Le profil est globalement adapté au poste, avec une bonne expérience en gestion de projet.",
        "points_forts": ["Expérience similaire", "Bonne communication"],
        "ecarts": ["Manque de certification demandée"]
    }}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt.strip()}],
            max_tokens=500
        )
    except Exception as e:
        return {"error": f"Erreur lors de l'appel à OpenAI: {str(e)}"}

    content = response.choices[0].message.content
    cleaned_content = re.sub(r"^```json\n?|```$", "", content.strip(), flags=re.MULTILINE)

    try:
        return json.loads(cleaned_content)
    except json.JSONDecodeError:
        return {"error": "La réponse de l'IA n'est pas un JSON valide", "raw": content}
