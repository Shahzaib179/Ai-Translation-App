import os

import streamlit as st
from google import genai


MODEL_NAME = "gemini-2.5-flash"

LANGUAGES = {
    "Urdu": "Urdu",
    "Arabic": "Arabic",
    "French": "French",
    "Spanish": "Spanish",
    "German": "German",
    "Italian": "Italian",
    "Portuguese": "Portuguese",
    "Turkish": "Turkish",
    "Chinese (Simplified)": "Simplified Chinese",
    "Japanese": "Japanese",
    "Korean": "Korean",
    "Hindi": "Hindi",
    "Bengali": "Bengali",
    "Persian": "Persian",
}


def get_api_key() -> str | None:
    """Read the Gemini API key from Streamlit secrets or an environment variable."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except (FileNotFoundError, KeyError):
        pass

    return os.getenv("GEMINI_API_KEY")


def read_uploaded_text(uploaded_file) -> str:
    """Decode a small text-based upload as UTF-8."""
    if uploaded_file is None:
        return ""

    raw = uploaded_file.getvalue()
    if not raw:
        return ""

    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("utf-8", errors="replace")


def translate_text(text: str, target_language: str, api_key: str) -> str:
    """Translate English text while preserving structure and meaning."""
    client = genai.Client(api_key=api_key)

    prompt = f"""
You are a professional translator.
Translate the English text below into {target_language}.

Rules:
- Return only the translated text.
- Preserve headings, bullet points, numbering, paragraph breaks, names, numbers, and formatting where possible.
- Keep the meaning and tone faithful to the original.
- Do not explain the translation and do not add commentary.

English text:
{text}
""".strip()

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")

    return response.text.strip()


def main() -> None:
    st.set_page_config(
        page_title="AI Language Translator",
        page_icon="🌐",
        layout="centered",
    )

    st.title("🌐 AI Language Translator")
    st.caption("Translate English text into multiple languages using Gemini Flash.")

    target_label = st.selectbox(
        "Translate into",
        options=list(LANGUAGES.keys()),
        index=0,
    )

    input_method = st.radio(
        "Choose input method",
        options=["Type or paste text", "Upload a text file"],
        horizontal=True,
    )

    source_text = ""

    if input_method == "Type or paste text":
        source_text = st.text_area(
            "English text",
            height=240,
            placeholder="Enter or paste English text here...",
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload an English text file",
            type=["txt", "md", "csv", "json"],
            help="Supported text-based formats: TXT, Markdown, CSV, and JSON.",
        )
        source_text = read_uploaded_text(uploaded_file)
        if source_text:
            st.text_area("Uploaded content", value=source_text, height=240, disabled=True)

    api_key = get_api_key()

    if not api_key:
        st.info(
            "Add your Gemini API key as GEMINI_API_KEY in Streamlit Secrets "
            "or as a local environment variable before translating."
        )

    if st.button("Translate", type="primary", use_container_width=True):
        if not source_text.strip():
            st.warning("Please enter English text or upload a text file first.")
        elif not api_key:
            st.error("Gemini API key is missing. Add GEMINI_API_KEY and try again.")
        else:
            try:
                with st.spinner(f"Translating to {target_label}..."):
                    translated = translate_text(
                        source_text.strip(),
                        LANGUAGES[target_label],
                        api_key,
                    )

                st.subheader("Translation")
                st.text_area(
                    "Translated text",
                    value=translated,
                    height=260,
                    label_visibility="collapsed",
                )
                st.download_button(
                    "Download translation",
                    data=translated.encode("utf-8"),
                    file_name=f"translation_{target_label.lower().replace(' ', '_')}.txt",
                    mime="text/plain; charset=utf-8",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"Translation failed: {exc}")

    with st.expander("About this app"):
        st.write(
            "This app uses Google's Gemini 2.5 Flash model. "
            "Your text is sent to the Gemini API only when you click Translate."
        )


if __name__ == "__main__":
    main()
