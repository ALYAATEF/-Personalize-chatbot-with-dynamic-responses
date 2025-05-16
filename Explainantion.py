# import necessary libraries
import re
import PyPDF2
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from dotenv import load_dotenv
import os
import io
import json

load_dotenv()  # Loads variables from .env into environment

# Get API key for Gemini
GEMINI_API_KEY = os.getenv("API_KEY")  # Make sure to set this in your .env file

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-1.5-flash-latest')

def generate_text(prompt, max_length=100, temperature=0.7):
    """Generate text using Gemini model"""
    try:
        # Configure generation parameters
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_length * 4,  # Gemini uses tokens differently than words
            temperature=temperature,
        )
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        return [response.text]
    except Exception as e:
        print(f"Error generating text: {e}")
        return [f"Error: {str(e)}"]

# the section patterns to identify different parts of a research paper
SECTION_PATTERNS = {
    "abstract": r'\babstract\b',
    "introduction": r'\bintroduction\b',
    "methodology": r'\b(methodology|methods|materials and methods)\b',
    "results": r'\b(results|findings)\b',
    "conclusion": r'\b(conclusion|conclusions|discussion and conclusion)\b'
}

# Compile the regex patterns for section headers
SECTION_REGEX = {k: re.compile(v, re.IGNORECASE) for k, v in SECTION_PATTERNS.items()}

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

def extract_sections(text):
    text = re.sub(r'\s+', ' ', text).strip()
    combined_pattern = r'|'.join(f'(?P<{k}>{v})' for k, v in SECTION_PATTERNS.items())
    section_header_regex = re.compile(combined_pattern, re.IGNORECASE)

    sections = {}
    matches = list(section_header_regex.finditer(text))

    for i, match in enumerate(matches):
        section_name = match.lastgroup
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end]
        sections[section_name] = content.strip()

    return sections

def extract_json_block(text):
    """Extract JSON from markdown code blocks"""
    pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    if matches:
        return f"```json\n{matches[-1]}\n```"
    
    # If no markdown json found, try to find JSON directly
    try:
        # Look for JSON pattern in the text
        json_pattern = r'\{[^}]*"explanation"[^}]*\}'
        json_match = re.search(json_pattern, text, re.DOTALL)
        if json_match:
            return f"```json\n{json_match.group()}\n```"
    except:
        pass
    
    # If still no JSON found, wrap the entire response as explanation
    return f'```json\n{{"explanation": "{text.strip()}"}}\n```'

def Simplify_research_part(content):
    """Simplify research content using Gemini"""
    explanation_schema = ResponseSchema(
        name="explanation",
        description="A simplified and clear explanation of the provided research text."
    )
    output_parser = StructuredOutputParser.from_response_schemas([explanation_schema])
    format_instructions = output_parser.get_format_instructions()

    prompt_template = """\
    You are a helpful assistant that explains research papers in a simple way.

    TASK:
    - Read the provided research content carefully.
    - Return a clear and simplified explanation in one paragraph using plain, accurate language.
    - Avoid adding any information that is not in the original content.
    - The goal is to help anyone understand this part, even without a research background, while staying true to the original meaning.
    - Format your response as JSON with the key "explanation".

    Content:
    {content}

    Please provide your response in the following JSON format:
    {{
        "explanation": "Your simplified explanation here"
    }}

    {format_instructions}
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["content", "format_instructions"]
    )

    formatted_prompt = prompt.format(content=content, format_instructions=format_instructions)

    # Generate response with increased max_length for better explanations
    response = generate_text(formatted_prompt, max_length=500, temperature=0.7)
    
    # Extract JSON block
    final_response = extract_json_block(response[0])
    
    try:
        # Parse the output
        output = output_parser.parse(final_response)
        return output
    except Exception as e:
        # If parsing fails, return a basic structure
        print(f"Parsing error: {e}")
        # Try to extract explanation manually
        try:
            # Clean up the response and extract explanation
            cleaned_response = response[0].strip()
            if cleaned_response.startswith('"') and cleaned_response.endswith('"'):
                cleaned_response = cleaned_response[1:-1]
            return {"explanation": cleaned_response}
        except:
            return {"explanation": "Unable to simplify this section"}

def process_pdf(pdf_content):
    """
    Process the PDF content to extract text and summarize it using Gemini.
    """
    try:

        # Create PDF reader from bytes
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        full_text = ""
        
        # Extract text from all pages
        for page in pdf_reader.pages:
            try:
                full_text += page.extract_text() + "\n"
            except Exception as e:
                print(f"Error extracting page: {e}")
                continue
        
        sections = extract_sections(full_text)
        results = {}

        for key in ['abstract', 'introduction', 'methodology', 'results', 'conclusion']:
            print(f"\n\n===== {key.capitalize()} =====\n")
            section_text = sections.get(key, "")
            if section_text:
                # Limit section text length to avoid token limits
                if len(section_text) > 3000:
                    section_text = section_text[:3000] + "..."
                
                try:
                    simplified = Simplify_research_part(section_text)
                    results[key] = simplified['explanation']
                    print(f"✅ Successfully simplified {key}")
                except Exception as e:
                    print(f"❌ Could not simplify {key}: {str(e)}")
                    # Add error handling to continue processing other sections
                    results[key] = f"Error processing {key}: {str(e)}"
            else:
                print("Section not found.")
                results[key] = "Section not found in the document."

        return results
    
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return {"error": str(e)}
    
    
# Test function to verify Gemini is working
def test_gemini():
    """Test if Gemini is properly configured"""
    try:
        response = model.generate_content("Please respond with 'Gemini is working correctly!'")
        print(f"✅ Gemini test successful: {response.text}")
        return True
    except Exception as e:
        print(f"❌ Gemini test failed: {e}")
        return False