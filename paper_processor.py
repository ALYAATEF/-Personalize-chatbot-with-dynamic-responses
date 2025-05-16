from Explainantion import test_gemini, process_pdf
# Additional helper function for better error handling
def process_pdf_with_fallback(pdf_path):
    """
    Process PDF with fallback mechanisms
    """
    # First test if Gemini is working
    if not test_gemini():
        return {"error": "Gemini API is not properly configured"}
    
    return process_pdf(pdf_path)


def answer_question(query):
    """
    Answer the question based on the processed PDF content.
    """
    # Simulate answering a question
    answer = "This is the answer to your question."
    return answer