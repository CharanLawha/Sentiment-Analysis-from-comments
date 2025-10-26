import streamlit as st
import pickle
import numpy as np

import re
from langdetect import detect_langs, LangDetectException
from spellchecker import SpellChecker # Import SpellChecker

# =============================
# Initialize SpellChecker
# =============================
spell = SpellChecker()

# =============================
# Load model files
# =============================
try:
    model = pickle.load(open("sentiment_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
except FileNotFoundError:
    st.error("Model files (sentiment_model.pkl, vectorizer.pkl, label_encoder.pkl) not found! Please make sure they are in the same directory.")
    st.stop()

from deep_translator import GoogleTranslator



# =============================
# UI
# =============================
st.set_page_config(page_title="Sentiment Analysis App", page_icon="💬", layout="centered")
st.title("💬 Sentiment Analysis (English / Urdu / Roman Urdu)")
st.write("Enter a comment in English, Urdu (اردو) or Roman Urdu (mujhe ye acha laga). App will detect, translate (if needed) and predict sentiment.")

user_input = st.text_area("✍️ Type your comment here:", height=160)

# =============================
# Helpers: roman-urdu list + detection
# (Keep all your existing helper functions here: contains_urdu_script, roman_urdu_score, is_english_like, safe_langdetect_probs)
# =============================

_ROMAN_URDU_WORDS = {
    # Pronouns
    "main","mein","me","mai","mujhe","mujhay","mujhko","mujko","mera","meri","mere","tum","tu","aap","ap","aapka","aapki","aapke",
    "us","use","usay","unko","unka","unki","unke","ham","hum","humein","humko","hamara","hamari","hamare",
    "woh","wo","vo","wohi","unka","unki","unhon","unho","inhon","inho",
    
    # Basic verbs
    "hona","hun","hoon","hain","hai","tha","thi","the","tha","thi","tha","thay",
    "karna","karta","kartay","karti","karti hoon","kartay ho","kar raha","kar rahi","kar rahe","kiya","kya","karein","karo",
    "aana","aaya","aayi","aaye","aata","aati","aate","jaana","gaya","gayi","jate","jata","jati","jaa","jana","aaja","chalo",
    "bolna","bola","boli","bolay","bol","sunna","suna","suni","sunay","sochna","socha","sochi","sochay","dekhna","dekha","dekhi","dekhe",
    "samajhna","samjha","samjhi","samjhe","likhna","likha","likhi","likhe","parhna","parha","parhi","parhe","chahiye","dena","diya","di","diyah",
    "lena","liya","li","liye","rakha","rakhi","rakhe","mangna","manga","mangi","mangna","mangta","mangti","mangtay",

    # Common connectors
    "aur","ya","lekin","magar","par","agar","to","phir","jab","tab","kyunki","isliye","ke","mein","may","kay","ka","ki","ke","kaun","kis","kisko","kuch","sab","sabhi",
    "ye","yeh","yah","woh","wo","voh","ab","tab","idhar","udhar","yahan","wahan","ahan","ahan","haan","han","nahi","nai","nhi","bilkul","zaroor","shayad","kabhi","hamesha",

    # Emotions / reactions
    "acha","achha","achae","bura","buri","theek","thik","mast","maza","mazaa","mazedar","mazay","mazaydaar","bore","boring","bohot","bahut","bahot","zabardast","best",
    "gussa","naraz","khush","khushi","udaas","dil","pyar","pyaar","piyar","ishq","mohabbat","nafrat","dard","gham","rona","ro","hasna","hansa","hansi","rona aaya",
    "mujhe acha laga","bohot acha","bura laga","acha nahi","acha nhi","acha tha","acha hai","acha lagta","acha lagti","bahut acha","bura laga","bura lagta","bura tha",
    "acha nahi","bura nahi","achha nahi","achha tha","pasand","pasand aya","pasand aaya","pasand nahi","pasand nhi","shukriya","thanks","dhanyavaad","jazakallah","allah ka shukar",
    "dukh","sukoon","tension","relax","thanda","garam","thaka","thaki","thake","thak gaya","thak gyi","thak gaya hoon","so gaya","so gyi","thoda","zyada","kam",

    # Greetings & common talk
    "salam","assalamualaikum","walaikumassalam","hello","hi","hey","bye","khuda hafiz","allah hafiz","good night","good morning","subah bakhair","shab bakhair",
    "kaisa","kaisi","kaise","theek ho","kya haal","sab theek","mein theek","aap kaise","shukriya","thank you","welcome","maaf","sorry","maafi","acha laga","bohot din baad",

    # Days / time / place
    "aaj","kal","parso","subah","shaam","raat", "h", "din","savera","dopahar","abhi","pehle","baad","der","jaldi","ghar","office","school","college","bazaar","market","masjid",
    "mandir","ghar jana","ghar aa","ghar gaya","ghar gaya tha","ghar par","ghar mai","room","kamra","city","gaon","sheher","karachi","lahore","islamabad","mithi","hyderabad",
    "idhar","udhar","bahar","andar","upar","neeche","samne","piche","left","right","road","rasta","gali",

    # Objects / common nouns
    "kitab","mobile","phone","charger","light","fan","table","kursi","car","bike","bus","rickshaw","train","ticket","bag","pani","chai","coffee","roti","khana","biryani",
    "andaa","anda","chawal","sabzi","gosht","roti","daal","achar","doodh","meetha","namak","mirch","thanda","garam","pankha","switch","computer","laptop","tv","remote","ac",
    "internet","wifi","network","battery","charger","pen","paper","copy","notebook","book","camera","pic","photo","video","song","music","gaana","film","movie","game",

    # Family / people
    "maa","ammi","mama","abba","abu","baba","papa","dad","mom","behan","bhai","bhaiya","dost","friend","teacher","sir","madam","chacha","chachi","phuppo","mamu","khala",
    "beta","beti","bacha","bachi","uncle","aunty","cousin","nephew","niece","student","classmate","boss","colleague","neighbour","padosi","log","insan","aadmi","aurat",
    "ladka","ladki","banda","bandi","shaadi","dulha","dulhan","rishta","biwi","mian","patni","husband","wife",

    # Common adjectives / states
    "naya","purana","bada","chota","lamba","chhota","mota","patla","mehnga","mehengi","sasta","tez","dhimi","acha","bura","sundar","khoobsurat","pyara","mazedar","ajeeb",
    "normal","khaas","garam","thanda","gila","sukha","andaaz","acha","nahi","behtar","kharab","mast","fit","perfect","cute","sweet","nice","cool","amazing","awesome","boring",

    # Feelings / actions / daily phrases
    "mujhe lagta hai","lagta hai","shayad","pata nahi","samajh nahi","samjha nahi","samjhi nahi","yaar","bhai","behan","plz","please","krdo","kar do","mat karo","na karo",
    "zaroor","pukka","pata hai","mujhe pta","mujhe nahi pta","mujhe lagta","mujhe samajh","sochta","sochti","yaad","yaad aaya","yaad nahi","bhool gaya","bhool gayi",
    "likhna","padna","dekhna","bolna","sona","uthna","khana","peena","jeena","marna","daurna","chalna","rukna","khelna","parhna","kaam karna","kaam","job","mehnat",
    "thoda sa","bohot zyada","thoda kam","zyada nahi","bas itna","kitna","kitni","kitne","sab kuch","kuch nahi","sab theek","sab accha","sab khush","sab udaas",

    # Islamic / cultural words
    "allah","khuda","dua","aameen","ameen","inshallah","mashallah","subhanallah","alhamdulillah","astaghfirullah","jazakallah","allah ka shukar","khuda ka shukar",
    "namaz","roza","eid","jumma","sehri","iftar","quran","masjid","azaan","dua karo","duaen","iman","islam","deen","sabr","shukar","barkat","rizq","zindagi","maut",

    # Slang / chat abbreviations
    "lol","lmao","hahaha","hehe","haha","hmm","hmm","huh","uff","ufff","ok","okk","okey","oky","bro","sis","yar","yaar","plz","pls","k","kk","thx","tnx","ty","np",
    "btw","idk","ikr","f9","gr8","gud","nyc","luv","bcz","coz","frnd","msg","txt","ya","yaa","haan","han","nahi","nai","acha","achha","theek","thik","bura","bohot",
    "acha lagta","acha nahi","acha tha","acha hai","nahi laga","acha laga","sahi","galat","fine","bad","worst","best","amazing","superb","awesome","cool","decent",
    "mujhe samajh nahi aaya","mujhe pasand","mujhe nahi pasand","bahot maza aaya","bohot acha laga","bura laga","mujhe acha laga","acha tha","acha nhi laga","mast tha",
    "acha experience","kharab tha","bohot khushi hui","udaasi thi","dil khush ho gaya","bura mehsoos hua","maza aya","fun tha","enjoy kiya","enjoyed"
}


def contains_urdu_script(text: str) -> bool:
    return bool(re.search(r'[\u0600-\u06FF]', text))

def roman_urdu_score(text: str) -> int:
    # count roman urdu tokens present
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    count = 0
    for t in tokens:
        if t in _ROMAN_URDU_WORDS:
            count += 1
    return count

def is_english_like(text: str) -> bool:
    return bool(re.search(r'[a-zA-Z]', text))

def safe_langdetect_probs(text: str):
    try:
        return detect_langs(text)  # list of Language objects with prob
    except LangDetectException:
        return []

# =============================
# NEW: Spelling Correction Function
# =============================
def correct_spelling(text: str) -> str:
    """Corrects spelling for English-like text tokens."""
    words = re.findall(r"[a-zA-Z']+", text)
    corrected_words = []
    
    # Keep track of original non-word separators
    tokens = re.findall(r"([a-zA-Z']+|\s+|[^\w\s])", text)
    
    corrected_text = []
    for token in tokens:
        if re.match(r"[a-zA-Z']+", token):
            # Only correct if the word is flagged as unknown
            if token.lower() in spell.unknown([token.lower()]):
                # Get the best correction
                correction = spell.correction(token)
                # Preserve the original capitalization if possible
                if correction and token[0].isupper() and len(token) > 1:
                    correction = correction.capitalize()
                corrected_text.append(correction or token) # Use original if correction fails
            else:
                corrected_text.append(token)
        else:
            corrected_text.append(token)
            
    return "".join(corrected_text)

# =============================
# NEW: Pre-processing for Negations (MODIFIED)
# =============================
def preprocess_negations(text: str) -> str:
    """
    Handles simple negations before feeding to the model.
    """
    # Define common negations and their antonyms
    negation_map = {
        # 1. NOT + POSITIVE WORD (e.g., 'not good' -> 'bad')
        r"\b(not|n't|is not|was not|wasn't|are not|aren't|have not|haven't|do not|don't|did not|didn't)\s+(good|great|happy|pleased|satisfied|well|like|best|excellent)\b": "bad",
        
        # 2. NOT + NEGATIVE WORD (e.g., 'not bad' -> 'good')
        # Includes 'worst' here to cover general negated negatives
        r"\b(not|n't|is not|was not|wasn't|are not|aren't|have not|haven't|do not|don't|did not|didn't)\s+(bad|poor|sad|unhappy|disappointed|hate|terrible|awful)\b": "good",
        
        # 3. Explicitly handling 'not worst' which means POSITIVE (better than the worst)
        r"\b(not|n't|is not|was not|wasn't|are not|aren't|have not|haven't|do not|don't|did not|didn't)\s+(worst)\b": "best", 

        # 4. 'not bad' is often neutral or slightly positive, mapping to a neutral word
        r"\b(not|n't|is not|was not|wasn't|are not|aren't|have not|haven't|do not|don't|did not|didn't)\s+(bad)\b": "okay"
    }
    
    processed_text = text
    for pattern, replacement in negation_map.items():
        processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)
        
    return processed_text

# =============================
# Robust translation strategy
# (Keep your existing robust_translate function)
# =============================

def clean_text(text):
    # remove emojis, urls, punctuation noise
    text = re.sub(r"http\S+|www\S+|@\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

def robust_translate(text: str):
    """
    Returns a tuple:
      (detected_type, translator_src_lang, translated_text)
    detected_type: one of "urdu_script", "roman_urdu", "english"
    translator_src_lang: 'ur' or 'en' or None on failure
    translated_text: translated English text (if translation performed) or original text
    """

    # --- 1) Detect text type ---
    if contains_urdu_script(text):
        detected_type = "urdu_script"
    else:
        rscore = roman_urdu_score(text)
        english_like = is_english_like(text)

        if rscore >= 2 and english_like:
            detected_type = "roman_urdu"
        elif not english_like:
            detected_type = "unknown"
        else:
            probs = safe_langdetect_probs(text)
            if probs:
                top = probs[0]
                if top.lang == "en" and top.prob > 0.85:
                    detected_type = "english"
                elif top.lang in ["ur", "pa", "hi"]:
                    detected_type = "roman_urdu"
                else:
                    detected_type = "roman_urdu" if rscore >= 1 else "english"
            else:
                detected_type = "roman_urdu" if rscore >= 1 else "english"

    # --- 2) Translation phase ---
    translator_src = None
    translated = text.strip()

    try:
        if detected_type in ("roman_urdu", "urdu_script"):
            # Force Urdu source to correctly translate Roman Urdu
            translated = GoogleTranslator(source="ur", target="en").translate(text)
            
            translator_src = "ur"
        else:
            translated = text
            translator_src = "en"

        # If translation didn’t change text, try once more (rare)
        if translated.strip().lower() == text.strip().lower() and detected_type != "english":
            translated = GoogleTranslator(source="auto", target="en").translate(text)
            translator_src = "ur"
            

    except Exception as e:
        st.warning(f"Translation failed: {e}. Predicting on original text.")
        translated = text
        translator_src = None

    return detected_type, translator_src, translated

# =============================
# Main predict flow (MODIFIED)
# =============================
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a comment first!")
    else:
        detected_type, translator_src, translated_text = robust_translate(user_input.strip())

        if detected_type == "unknown":
            st.error("🚫 Input not recognized as English, Urdu (script) or Roman Urdu. Please enter text in English or Urdu (script or romanized).")
        else:
            # Show detection summary
            human_type = {
                "urdu_script": "Urdu (script)",
                "roman_urdu": "Roman Urdu (latn)",
                "english": "English"
            }.get(detected_type, "Unknown")
            st.info(f"Detected as: **{human_type}**")
            if translator_src:
                st.caption(f"Translator detected source language: `{translator_src}` (googletrans)")
            
            # Show both original and translated (if different)
            st.markdown("**Original Text:**")
            st.write(user_input)
           # st.write(translated_text)
             
            
            if translated_text.strip().lower() != user_input.strip().lower():
                st.markdown("Translated to English:")
                st.write(translated_text)
            else:
                st.markdown("No translation performed (text already English or translation identical to input).")

            # === START: MODIFIED SECTION ===
            
            # Use the translated_text for prediction
            input_after_translation = translated_text

            # NEW STEP: Apply spelling correction on the *translated* text
            input_after_spellcheck = correct_spelling(input_after_translation)
            
            if input_after_spellcheck.lower() != input_after_translation.lower():
                st.markdown("**Text after spelling correction:**")
                st.write(f"> {input_after_spellcheck}")
                st.caption("_This text is then used for the negation patch._")

            # Apply negation patch on the spell-checked text
            input_for_model = preprocess_negations(input_after_spellcheck)

            # Show the pre-processed text if it changed (negation)
            if input_for_model.lower() != input_after_spellcheck.lower():
                st.markdown("**Text after negation patch:**")
                st.write(f"> {input_for_model}")
                st.caption("_This 'patched' text is sent to the model._")
            
            # === END: MODIFIED SECTION ===

            # Vectorize and predict
            try:
                input_vec = vectorizer.transform([input_for_model])
                prediction = model.predict(input_vec)[0]
                sentiment_label = label_encoder.inverse_transform([prediction])[0]
                
                # probability
                if hasattr(model, "predict_proba"):
                    prob = np.max(model.predict_proba(input_vec)) * 100
                else:
                    prob = None # Model might not support predict_proba (e.g., LinearSVC)

                # Display result
                st.success(f"🎯 **Predicted Sentiment: {sentiment_label.upper()}**")
                if prob is not None:
                    st.info(f"Confidence: {prob:.2f}%")

                # Emoji visualization
                if "pos" in sentiment_label.lower():
                    st.markdown("## 😊")
                    st.markdown("The comment seems **Positive**!")
                elif "neg" in sentiment_label.lower():
                    st.markdown("## 😠")
                    st.markdown("The comment seems **Negative**!")
                elif "neutral" in sentiment_label.lower():
                    st.markdown("## 😐")
                    st.markdown("The comment seems **Neutral**!")
                else:
                    st.markdown("## 🤔")
                    st.markdown(f"The comment expresses **{sentiment_label.title()}** sentiment.")

            except Exception as e:
                st.error(f"Prediction error: {e}")

st.markdown("---")
st.caption("Developed with ❤️ — improved heuristics for Urdu / Roman Urdu detection and translation")
