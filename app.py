from flask import Flask, request, jsonify, render_template
import openai
import joblib
import pandas as pd
from datetime import datetime, timedelta
import fitz  
import os
import re
import requests
from bs4 import BeautifulSoup

# Initialize Flask application
app = Flask(__name__)

# OpenAI API key setup
openai.api_key = "sk-VVCFekEE3eM3cOxApXGV8I-iUJKBR5fbIeBz_SerndT3BlbkFJ3jNLRRNNqEoKzSkoV1Y09Kc-RL46oJGTKH7Ts3qzEA"

# Load trained model and columns for prediction
model = joblib.load('crop_yield_model.pkl')
X_train_columns = joblib.load('X_train_columns.pkl')

# Define constants
regions_senegal = [
    'Dakar', 'Diourbel', 'Fatick', 'Kaffrine', 'Kaolack', 
    'Kédougou', 'Kolda', 'Louga', 'Matam', 'Saint-Louis', 
    'Sédhiou', 'Tambacounda', 'Thiès', 'Ziguinchor'
]
cultures = ["Riz", "Mil", "Maïs", "Sorgho", "Arachide", "Coton"]

# Initialize user data and step tracker
user_data = {}
step = 0

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chat.html')


@app.route('/chat', methods=['POST'])
def chat():
    global user_data, step
    user_message = request.json['message']
    return jsonify(handle_user_input(user_message))

@app.route('/meteo')
def meteo():
    return render_template('meteo.html')

@app.route("/semences", methods=["GET"])
def semences_page():
    return render_template("semences.html")

@app.route("/get_seed_info", methods=["POST"])
def get_seed_info():
    data = request.json
    seed_name = data.get("seed")

    # Log message to ensure this part of code is reached
    print(f"Fetching information for seed: {seed_name}")

    try:
        # Call OpenAI API for seed description
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Décrivez la semence {seed_name}.",
            max_tokens=100
        )
        info = response.choices[0].text.strip()
        return jsonify({"info": info})
    except Exception as e:
        print(f"Erreur lors de la récupération des informations pour la semence {seed_name}: {e}")
        return jsonify({"info": "Erreur lors de la récupération de la description de la semence."})

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    # Obtenir le fichier audio du formulaire
    audio_file = request.files.get("audio")
    
    if audio_file:
        # Sauvegarder temporairement l'audio
        audio_path = "temp_audio.wav"
        audio_file.save(audio_path)
        
        try:
            # Appel à OpenAI pour transcrire en français
            with open(audio_path, "rb") as audio:
                transcript = openai.Audio.transcribe("whisper-1", audio, language="fr")

            # Supprimer le fichier temporaire après utilisation
            os.remove(audio_path)
            
            return jsonify({"transcription": transcript['text']})
        
        except Exception as e:
            return jsonify({"error": f"Erreur lors de la transcription : {str(e)}"}), 500
    else:
        return jsonify({"error": "Aucun fichier audio fourni."}), 400

# Chargement des articles depuis le fichier Excel
def load_articles():
    try:
        df = pd.read_excel("agriculture_articles.xlsx")
        articles = df.to_dict(orient="records")
        return articles
    except FileNotFoundError:
        print("Le fichier agriculture_articles.xlsx n'a pas été trouvé.")
        return []

@app.route("/agriculture", endpoint="agriculture")
def agriculture():
    articles = load_articles()
    return render_template("agriculture.html", articles=articles)


# Core function to handle chatbot interactions
def handle_user_input(user_message):
    global user_data, step
    print(f"Étape actuelle : {step}")

    # Step 0: Ask for region
    if step == 0:
        if user_message in regions_senegal:
            user_data['region'] = user_message
            step += 1
            return {"message": "Quelle culture souhaitez-vous planter ?"}
        else:
            return {"message": "Veuillez choisir une région parmi les suivantes : " + ', '.join(regions_senegal)}

    # Step 1: Ask for crop type
    elif step == 1:
        if user_message.capitalize() in cultures:
            user_data['crop'] = user_message.capitalize()
            step += 1
            return {"message": "Quelle est la surface de la parcelle en hectares ?"}
        else:
            return {"message": "Veuillez choisir une culture parmi : " + ', '.join(cultures)}

    # Step 2: Ask for surface area
    elif step == 2:
        try:
            user_data['area'] = float(user_message)
            step += 1
            return {"message": "Quelle saison de culture souhaitez-vous ? (Saison des Pluies, Saison sèche, Toute l'année)"}
        except ValueError:
            return {"message": "Veuillez entrer une surface valide (en hectares)."}

    # Step 3: Ask for season and detect rainfall automatically
    elif step == 3:
        user_data['season'] = user_message
        rainfall_message = get_rainfall_for_region(user_data['region'])
        
        if rainfall_message:
            user_data['annual_rainfall'] = rainfall_message
            step += 1
            return {"message": f"La pluviométrie détectée pour {user_data['region']} est de {rainfall_message}. Voulez-vous que je vous propose le montant de pesticides ? (oui/non)"}
        else:
            return {"message": "Je n'ai pas pu obtenir la pluviométrie, veuillez réessayer."}

    # Step 4: Ask for pesticide amount
    elif step == 4:
        if user_message.lower() == "oui":
            pesticide = propose_pesticide(user_data['crop'], user_data['season'], user_data['annual_rainfall'], user_data['region'], user_data['area'])
            if pesticide:
                user_data['pesticide'] = pesticide
                step += 1
                return {"message": f"Le montant proposé de pesticides est de {pesticide} kg. Voulez-vous l'utiliser ? (oui/non)"}
            else:
                return {"message": "Je n'ai pas pu déterminer la quantité de pesticides, veuillez entrer un montant manuel."}
        elif user_message.lower() == "non":
            return {"message": "Veuillez entrer la quantité de pesticides utilisé (en kg)."}
        else:
            try:
                user_data['pesticide'] = float(user_message)
                step += 1
                return {"message": "Souhaitez-vous que je vous propose le montant d'engrais ? (oui/non)"}
            except ValueError:
                return {"message": "Veuillez entrer une quantité valide de pesticides (en kg)."}

    # Step 5: Ask for fertilizer amount
    elif step == 5:
        if user_message.lower() == "oui":
            fertilizer = propose_fertilizer(user_data['crop'], user_data['season'], user_data['annual_rainfall'], user_data['region'], user_data['area'])
            if fertilizer:
                user_data['fertilizer'] = fertilizer
                step += 1
                return {"message": f"La quantité proposé d'engrais est de {fertilizer} kg. Voulez-vous l'utiliser ? (oui/non)"}
            else:
                return {"message": "Je n'ai pas pu déterminer une quantité d'engrais, veuillez entrer la quantité manuellement."}
        elif user_message.lower() == "non":
            return {"message": "Veuillez entrer la quantité d'engrais utilisé (en kg)."}
        else:
            try:
                user_data['fertilizer'] = float(user_message)
                step += 1
                return {"message": predict_yield() + " Voulez-vous entrer la date à laquelle vous souhaitez commencer la culture ? (oui/non)"}
            except ValueError:
                return {"message": "Veuillez entrer une quantité valide d'engrais (en kg)."}

    # Step 6: Predict yield and ask for cultivation date
    elif step == 6:
        yield_message = predict_yield()
        step += 1
        return {"message": yield_message + " Voulez-vous entrer la date à laquelle vous souhaitez commencer la culture ? (oui/non)"}

    # Step 7: Ask for cultivation date input
    elif step == 7:
        if user_message.lower() == "oui":
            step += 1
            return {"message": "Veuillez entrer la date souhaitée pour commencer la culture (format JJ/MM/AAAA)."}
        elif user_message.lower() == "non":
            return {"message": generate_technical_advice(user_data['crop'], user_data['region'], user_data['fertilizer'], user_data['pesticide'])}
        else:
            return {"message": "Veuillez répondre par 'oui' ou 'non'."}

    # Step 8: Finalize itinerary
    elif step == 8:
        if validate_date(user_message):
            user_data['cultivation_date'] = user_message
            itinerary_message = generate_technical_advice(user_data['crop'], user_data['region'], user_data['fertilizer'], user_data['pesticide'])
            step = 0
            user_data.clear()
            return {"message": itinerary_message + "<br><br>Souhaitez-vous refaire la prédiction de rendement ou l'itinéraire ? (oui/non)"}
        else:
            return {"message": "Date invalide. Veuillez entrer une date au format JJ/MM/AAAA."}
    elif step == 9:
        if user_message.lower() == "oui":
            step = 0
            user_data.clear()
            return {"message": "Veuillez choisir une région parmi les suivantes : " + ', '.join(regions_senegal)}
        elif user_message.lower() == "non":
            return {"message": "Merci d'avoir utilisé le service ! N'hésitez pas à revenir pour une nouvelle prédiction."}
        else:
            return {"message": "Veuillez répondre par 'oui' ou 'non'."}

# Helper functions (propose_pesticide, propose_fertilizer, extract_numeric_value, get_rainfall_for_region, predict_yield, generate_technical_advice, validate_date)
# (Implement helper functions as in your original code)



def propose_pesticide(crop, season, rainfall, region, area):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an agricultural assistant."},
                {"role": "user", "content": f"Proposez un montant de pesticides pour la culture de {crop} dans la région {region} pendant la {season} avec une pluviométrie de {rainfall} mm sur une superficie de {area} hectares."}
            ]
        )
        print(f"Réponse brute OpenAI pour pesticide : {response}")  # Debug print
        return extract_numeric_value(response['choices'][0]['message']['content']) * area  # Multiplication par la superficie en hectares
    except Exception as e:
        print(f"Erreur lors de l'appel à OpenAI pour les pesticides: {e}")
        return None

def propose_fertilizer(crop, season, rainfall, region, area):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an agricultural assistant."},
                {"role": "user", "content": f"Proposez un montant d'engrais pour la culture de {crop} dans la région {region} pendant la {season} avec une pluviométrie de {rainfall} mm sur une superficie de {area} hectares."}
            ]
        )
        print(f"Réponse brute OpenAI pour fertilizer : {response}")  # Debug print
        return extract_numeric_value(response['choices'][0]['message']['content']) * area  # Multiplication par la superficie en hectares
    except Exception as e:
        print(f"Erreur lors de l'appel à OpenAI pour les engrais: {e}")
        return None


def extract_numeric_value(text):
    import re
    # Chercher toutes les valeurs numériques dans le texte
    matches = re.findall(r'\d+', text)
    if matches:
        print(f"Valeurs numériques extraites : {matches}")  # Afficher toutes les valeurs extraites
        return float(matches[0])  # Retourner la première valeur numérique trouvée
    else:
        print("Aucune valeur numérique trouvée.")
        return None


def get_rainfall_for_region(region):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You provide rainfall data."},
                {"role": "user", "content": f"Quelle est la pluviométrie moyenne annuelle pour la région {region} au Sénégal ?"}
            ]
        )
        full_message = response['choices'][0]['message']['content']
        print(f"Réponse OpenAI pour la pluie: {full_message}")  # Debug pour voir la réponse dans le terminal
        
        # Retourner le texte complet de la réponse au lieu d'une simple valeur numérique
        return full_message  # Retourner la réponse entière pour l'afficher dans l'interface
    except Exception as e:
        print(f"Erreur lors de la récupération des données de pluviométrie: {e}")
        return "Erreur lors de la récupération de la pluviométrie"


def predict_yield():
    region = user_data['region']
    crop = user_data['crop']
    area = user_data['area']
    rainfall = user_data['annual_rainfall']
    fertilizer = user_data['fertilizer']
    pesticide = user_data['pesticide']
    cultivation_date = user_data.get('cultivation_date', 'N/A')  # Ajout de la date de culture si disponible

    # Créer un DataFrame pour la prédiction
    data = {'Crop': [crop], 'Crop_Year': [2024], 'Season': [user_data['season']], 'State': [region],
            'Area': [area], 'Annual_Rainfall': [rainfall], 'Fertilizer': [fertilizer], 'Pesticide': [pesticide],
            'Cultivation_Date': [cultivation_date]}  # Inclure la date dans les données

    df = pd.DataFrame(data)
    df = pd.get_dummies(df, drop_first=True)

    # Ajouter les colonnes manquantes
    missing_cols = set(X_train_columns) - set(df.columns)
    for col in missing_cols:
        df[col] = 0

    df = df[X_train_columns]
    predicted_yield = model.predict(df)[0]

    return f"Le rendement prédit pour {crop} dans la région {region} est de {predicted_yield:.2f} tonnes par hectare."



# Folder containing PDF documentation
pdf_folder_path = "Itinéraire Technique Documentation"

def extract_text_from_pdfs():
    combined_text = ""
    if os.path.exists(pdf_folder_path) and os.path.isdir(pdf_folder_path):
        for pdf_file in os.listdir(pdf_folder_path):
            if pdf_file.endswith(".pdf"):
                with fitz.open(os.path.join(pdf_folder_path, pdf_file)) as pdf:
                    for page_num in range(pdf.page_count):
                        page = pdf.load_page(page_num)
                        combined_text += page.get_text()
    else:
        print("Le dossier 'Itinéraire Technique Documentation' est introuvable ou vide.")
    if not combined_text:
        print("Aucun contenu extrait des fichiers PDF.")
    return combined_text


def generate_technical_advice(crop, region, fertilizer, pesticide):
    try:
        start_date_str = user_data.get('cultivation_date', '01/01/2024')
        start_date = datetime.strptime(start_date_str, "%d/%m/%Y")

        pdf_content = extract_text_from_pdfs()
        if not pdf_content:
            return "Erreur : Aucun contenu pertinent trouvé dans les documents PDF. Veuillez vérifier le dossier 'Itinéraire Technique Documentation'."

        climate_info = f"Climat de {region} propice à la culture de {crop}."

        # Instructions étape par étape avec dates calculées
        advice_steps = [
            {
                "step": "Préparation du sol",
                "days_after_start": 0,
                "details": (
                    f"- Labourer le sol en profondeur pour favoriser l'aération et le drainage. "
                    f"Incorporer les engrais recommandés ({fertilizer} kg) pour enrichir le sol."
                ),
            },
            {
                "step": "Semis du riz",
                "days_after_start": 30,  # Exemple de décalage de 1 mois
                "details": (
                    "- Préparer les semences et les répartir uniformément sur la parcelle. "
                    "Assurer une irrigation régulière pour favoriser la germination."
                ),
            },
            {
                "step": "Entretien du riz",
                "days_after_start": 60,  # Exemple de décalage de 2 mois
                "details": (
                    f"- Effectuer des binages réguliers et appliquer les pesticides ({pesticide} kg) "
                    "pour protéger les plants des ravageurs."
                ),
            },
            {
                "step": "Récolte du riz",
                "days_after_start": 120,  # Exemple de décalage de 4 mois
                "details": (
                    "- Observer la maturité du riz pour déterminer le moment propice à la récolte. "
                    "Couper les épis et les laisser sécher."
                ),
            },
            {
                "step": "Stockage et commercialisation",
                "days_after_start": 150,  # Exemple de décalage de 5 mois
                "details": (
                    "- Sécher et stocker le riz dans des conditions appropriées. "
                    "Préparer la commercialisation du riz."
                ),
            },
        ]

        # Construire l'itinéraire en ajoutant les dates calculées
        itinerary_with_dates = f"Itinéraire technique pour la culture de {crop} dans la région de {region}.\n\n"
        for step in advice_steps:
            step_date = start_date + timedelta(days=step["days_after_start"])
            itinerary_with_dates += f"{step['step']} (à partir du {step_date.strftime('%d/%m/%Y')}) : {step['details']}\n\n"

        # Ajouter des balises HTML pour formatage correct dans l'interface web
        return itinerary_with_dates.replace('\n', '<br>')
    except Exception as e:
        print(f"Erreur lors de la génération de l'itinéraire technique : {e}")
        return "Erreur lors de la génération de l'itinéraire technique. Veuillez réessayer."


import re

def validate_date(date_text):
    # Expression régulière pour le format de la date JJ/MM/AAAA
    return re.match(r'\d{2}/\d{2}/\d{4}', date_text) is not None


if __name__ == '__main__':
    app.run(debug=True)
