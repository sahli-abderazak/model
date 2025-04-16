import os
import json
import re
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import openai
from openai import OpenAI
import replicate

# === CONFIGURATION ===

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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === SCHEMAS ===

class ImageQuestionRequest(BaseModel):
    cv: str
    offre: str

class ImagePersonalityRequest(BaseModel):
    image_url: str
    image_prompt: str
    description: str

class Offre(BaseModel):
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

# === ROUTES ===



'''
def get_difficulty_level(niveau: str) -> str:
    niveau = niveau.lower().strip()
    if "aucune" in niveau or "0" in niveau or "1" in niveau:
        return "facile"
    elif any(x in niveau for x in ["2", "3", "4"]):
        return "moyenne"
    elif any(x in niveau for x in ["5", "6", "7", "8", "9", "10", "plus"]):
        return "difficile"
    else:
        return "moyenne"

# ===== Route principale =====
@app.post("/generate-test", response_model=Dict[str, Any])
async def generate_test(offre: OffreInput) -> Dict[str, Any]:
    selected_traits = ["Empathy", "Communication", "Self-Control", "Sociability", "Conscientiousness"]
    difficulty = get_difficulty_level(offre.niveauExperience)

    niveau_difficulte = {
        "facile": "- Les questions doivent être simples et accessibles.",
        "moyenne": "- Les questions doivent avoir une difficulté modérée, avec des situations classiques du poste.",
        "difficile": "- Les questions doivent être complexes, incluant des dilemmes, des cas concrets ou des analyses comportementales poussées."
    }[difficulty]

    prompt = f"""
    Tu es un expert en psychologie RH.

    Génère 10 questions de test de personnalité sous forme de QCM (Questions à Choix Multiples).
    Chaque deux questions doivent évaluer l’un des 5 traits suivants :
    {', '.join(selected_traits)}.

    Ces questions doivent être adaptées à l’offre suivante :
    - Poste : {offre.poste}
    - Description : {offre.description}
    - Type de travail : {offre.typeTravail}
    - Niveau d’expérience requis : {offre.niveauExperience}
    - Responsabilités principales : {offre.responsabilite}
    - Expérience professionnelle attendue : {offre.experience}

    {niveau_difficulte}

    Format requis en JSON :
    [
        {{
            "trait": "Empathy",
            "question": "Votre question ici",
            "options": [
                {{"text": "Réponse 1", "score": 1}},
                {{"text": "Réponse 2", "score": 2}},
                {{"text": "Réponse 3", "score": 4}},
                {{"text": "Réponse 4", "score": 5}}
            ]
        }}
    ]

    Ne réponds qu'au format JSON, sans explication, sans balises Markdown.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt.strip()}],
            max_tokens=2000,
            temperature=0.7
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'appel à l'API OpenAI: {str(e)}")

    content = response.choices[0].message.content
    cleaned_content = re.sub(r"^```json\n?|```$", "", content.strip(), flags=re.MULTILINE)

    try:
        questions = json.loads(cleaned_content)
    except json.JSONDecodeError as json_error:
        return JSONResponse(
            status_code=502,
            content={
                "error": "La réponse de l'IA n'est pas un JSON valide.",
                "raw": cleaned_content,
                "json_error": str(json_error),
            }
        )

    if not isinstance(questions, list) or not all(
        isinstance(q, dict) and 'trait' in q and 'question' in q and 'options' in q for q in questions
    ):
        return JSONResponse(
            status_code=400,
            content={"error": "Le format des questions n'est pas correct.", "raw": cleaned_content}
        )

    return {"questions": questions}
    
    '''
@app.post("/generate-image-question")
async def generate_image_question(data: ImageQuestionRequest) -> Dict[str, str]:
    prompt = f"""
    Génère une illustration simple représentant une situation professionnelle reflétant la personnalité d’un candidat.
    Pas de texte. Style clair, épuré.

    CV : {data.cv}
    Offre : {data.offre}

    Exemples :
    - Personne échangeant calmement en salle de réunion
    - Personne aidant un collègue à résoudre un problème
    - Personne concentrée seule dans un bureau
    """

    try:
        response = openai.images.generate(
            model="dall-e-3",
            prompt=prompt.strip(),
            n=1,
            size="1024x1024",
            response_format="url"
        )
        image_url = response.data[0].url
        return {"image_url": image_url, "description_auto": prompt.strip()}
    except Exception as e:
        return {"error": f"Erreur lors de la génération de l'image : {str(e)}"}

@app.post("/analyze-personality")
async def analyze_personality(data: ImagePersonalityRequest) -> Dict[str, str]:
    prompt = f"""
    Voici une image représentant une scène professionnelle : elle a été générée selon cette intention :
    "{data.image_prompt}"

    Le candidat a ensuite rédigé la description suivante :
    "{data.description}"

    Analyse de manière concise la personnalité du candidat, en te concentrant sur :
    - Les traits de personnalité principaux (ex : leader, analytique, orienté équipe, etc.)
    - Son approche du travail et de la collaboration.
    - Sa réaction probable face à une situation similaire à celle décrite.

    Réponds de manière brève et directe, en résumant les éléments clés de la personnalité du candidat.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt.strip()}],
            max_tokens=150
        )
        content = response.choices[0].message.content.strip()
        return {"personality_analysis": content}
    except Exception as e:
        return {"error": f"Erreur lors de l'analyse: {str(e)}"}

@app.post("/match-cv-offre")
async def match_cv_offre(data: MatchingScoreRequest) -> Dict[str, Any]:
    offre = data.offre

    prompt = f"""
    Tu es un assistant RH expert en recrutement. Ta tâche est d'analyser le niveau de correspondance entre un CV et une offre d'emploi.

    CV :
    {data.cv}

    Offre d'emploi :
    Description: {offre.description}
    Niveau d'expérience: {offre.niveauExperience}
    Niveau d’étude: {offre.niveauEtude}
    Responsabilités: {offre.responsabilite}
    Expérience demandée: {offre.experience}
    Pays: {offre.pays}
    Ville: {offre.ville}

    Analyse les similarités entre :
    - Les expériences du candidat et les responsabilités demandées
    - Le niveau d’étude et d’expérience
    - Les compétences techniques et comportementales

    Donne :
    - Un score de matching entre 0 et 100
    - Une évaluation brève de l'adéquation du profil
    - Les points forts et les écarts

    Réponds uniquement au format JSON :
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