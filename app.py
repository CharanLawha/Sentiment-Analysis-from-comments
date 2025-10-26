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
    "mujhe","mujhay","mujhko","mujko","mera","meri","mere","tum","tu","ap","aap","acha","acha","achae",
    "acha","acha","achha","acha","acha","bura","buri","nahi","nai","nhi","kyun","kyu","kyon","kya","kahan",
    "kab","bohot","bahut","bahot","bhut","pyar","piyar","pyaar","love","khushi","khush","udaas","sad",
    "gussa","gussah","gussa","acha laga","acha laga","pasand","nazar","nahi","theek","theek hai","acha hai",
    "acha nhi","acha nahi","shukriya","thanks","dhanyavaad","jazakallah","jazakallah","acha tha","acha tha",
    "kal","aaj","subah","shaam","raat","ghar","school","college","kam","mazaa"
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
def robust_translate(text: str):
    """
    Returns a tuple:
      (detected_type, translator_src_lang, translated_text)
    detected_type: one of "urdu_script", "roman_urdu", "english"
    translator_src_lang: what googletrans says it detected (e.g., 'ur','en') or None on failure
    translated_text: translated English text (if translation performed) or original text
    """
    # 1) Fast checks
    if contains_urdu_script(text):
        detected_type = "urdu_script"
    else:
        # compute roman-urdu score and simple english presence
        rscore = roman_urdu_score(text)
        english_like = is_english_like(text)
        # heuristics:
        # - if many roman-Urdu tokens present => roman_urdu
        # - if only ascii letters and few roman tokens => english
        if rscore >= 2 and english_like:
            detected_type = "roman_urdu"
        elif not english_like:
            # no latin letters and no urdu script -> unknown/unsupported
            detected_type = "unknown"
        else:
            # ask langdetect to help
            probs = safe_langdetect_probs(text)
            if probs:
                top = probs[0]
                # if top language is english with high prob -> english
                if top.lang == "en" and top.prob > 0.85:
                    detected_type = "english"
                # if top is urdu (rare) or others -> treat as roman or unknown
                elif top.lang == "ur" or top.lang == "pa" or top.lang == "hi":
                    # these likely indicate non-english -> treat as roman_urdu
                    detected_type = "roman_urdu"
                else:
                    # fallback based on roman score
                    detected_type = "roman_urdu" if rscore >= 1 else "english"
            else:
                detected_type = "roman_urdu" if rscore >= 1 else "english"

    # 2) Try translation with translator
    translator_src = None
    translated = text
    try:
        # ask translator to auto-detect first
        res = GoogleTranslator(source='auto', target='en').translate(text)
        translator_src = getattr(res, 'src', None)
        translated_candidate = getattr(res, 'text', text)
        
        # if translator thinks source is English but we flagged roman_urdu/urdu_script,
        # try forcing src='ur' (google translates roman-Urdu poorly unless forced)
        if translator_src in ('en', 'und') and detected_type in ("roman_urdu", "urdu_script"):
            # second attempt: force Urdu as source
            try:
                res2 = GoogleTranslator(source='auto', target='en').translate(text)
                translator_src = getattr(res2, 'src', translator_src) # Use 'ur' if successful
                translated = getattr(res2, 'text', translated_candidate)
            except Exception:
                # fallback to first candidate
                translated = translated_candidate
        else:
            translated = translated_candidate
            
    except Exception as e:
        # translator failed: keep original text and mark src None
        translator_src = None
        translated = text
        st.warning(f"Translation failed: {e}. Predicting on original text.")

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
            
            if translated_text.strip().lower() != user_input.strip().lower():
                st.markdown("**Translated to English:**")
                st.write(translated_text)
            else:
                st.markdown("_No translation performed (text already English or translation identical to input)._")

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
