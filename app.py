"""
Spam / Phishing Email Detector — Streamlit App
Run with: streamlit run app.py

One input box, one button — runs both models together via
unified_detector.analyze_message() and shows a combined verdict plus a
breakdown of what each model individually thinks.
"""

import streamlit as st

from unified_detector import analyze_message

st.set_page_config(page_title="Email Detector", page_icon="📧", layout="centered")

if "history" not in st.session_state:
    st.session_state.history = []

st.title("📧 Email / Message Detector")
st.caption(
    "Paste any email or message below. This checks it against two models at "
    "once: Spam/Ham (English) and Phishing (URL/email/SMS-aware)."
)

email = st.text_area(
    "Paste email or message text here",
    height=160,
    placeholder="e.g. Win a free iPhone now, click here!",
)

check = st.button("🔍 Analyze", type="primary", use_container_width=True)

VERDICT_STYLE = {
    "Phishing": ("error", "🎣 Possible Phishing"),
    "Spam": ("error", "🚨 Spam Detected"),
    "Ham": ("success", "✅ Looks Legit (Ham)"),
    "Unknown": ("warning", "❓ Couldn't determine — no models available"),
}

if check:
    if not email.strip():
        st.warning("Please paste some text first.")
    else:
        with st.spinner("Running spam and phishing checks..."):
            result = analyze_message(email)

        style, label = VERDICT_STYLE[result["verdict"]]
        getattr(st, style)(label)

        st.subheader("Breakdown by model")

        # --- Spam/Ham ---
        col1, col2 = st.columns(2)
        spam = result["spam"]
        with col1:
            st.markdown("**Spam/Ham** (English)")
            if spam["available"]:
                st.metric("Spam probability", f"{spam['spam_probability']}%")
                st.write("🚨 Spam" if spam["is_spam"] else "✅ Ham")
            else:
                st.caption(f"⚠️ Unavailable — {spam['error']}")

        # --- Phishing ---
        phishing = result["phishing"]
        with col2:
            st.markdown("**Phishing**")
            if phishing["available"]:
                st.metric("Phishing probability", f"{phishing['phishing_probability']}%")
                st.write("🎣 Phishing" if phishing["is_phishing"] else "✅ Not phishing")
            else:
                st.caption(f"⚠️ Unavailable — {phishing['error']}")

        st.session_state.history.insert(0, {
            "text": email.strip()[:100],
            "verdict": result["verdict"],
        })

if st.session_state.history:
    st.divider()
    st.subheader("🕘 Recent checks")
    icons = {"Phishing": "🎣", "Spam": "🚨", "Ham": "✅", "Unknown": "❓"}
    for item in st.session_state.history[:10]:
        st.write(f"{icons.get(item['verdict'], '❓')} **{item['verdict']}** — {item['text']}")