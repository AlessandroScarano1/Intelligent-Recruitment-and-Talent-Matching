# document parser for cv and job postings
# uses unstructured library to handle PDF, Word, CSV, and text files
# usage:
#     from demo.scripts.document_parser import parse_document
#     text = parse_document('/path/to/cv.pdf')

import os
import time
import logging
from pathlib import Path

# configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# supported file extensions
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt', '.csv', '.rtf'}


def parse_document(filepath, timeout_seconds=60):
    # parse a document and extract text

    # args:
    #     filepath: path to document file
    #     timeout_seconds: maximum time for parsing (prevents hanging on corrupt files)

    # returns:
    #     dict with keys: text, filename, file_type, parse_time
    #     returns none if parsing fails

    # supported formats: PDF, DOCX, DOC, TXT, CSV, RTF

    filepath = Path(filepath)

    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        return None

    ext = filepath.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        logger.warning(f"Unsupported file type: {ext}")
        return None

    start_time = time.time()

    try:
        # use unstructured for all document types
        from unstructured.partition.auto import partition

        logger.info(f"Parsing {filepath.name}")

        # partition document (auto-detects format)
        elements = partition(filename=str(filepath))

        # extract text from elements
        text_parts = []
        for el in elements:
            el_text = str(el).strip()
            if el_text:
                text_parts.append(el_text)

        text = "\n".join(text_parts)

        # validate output
        if not text or len(text.strip()) < 10:
            logger.warning(f"Extracted text too short from {filepath.name}")
            return None

        parse_time = time.time() - start_time

        result = {
            'text': text,
            'filename': filepath.name,
            'file_type': ext,
            'parse_time': parse_time,
            'char_count': len(text),
            'word_count': len(text.split())
        }

        logger.info(f"Parsed {filepath.name}: {result['word_count']} words in {parse_time:.2f}s")
        return result

    except ImportError:
        logger.error("Unstructured library not installed. Run: pip install 'unstructured[all-docs]'")
        return None

    except Exception as e:
        logger.error(f"Failed to parse {filepath.name}: {e}")
        return None


def parse_with_fallback(filepath):
    # parse document with fallback to simple text extraction
    # uses unstructured first, falls back to basic extraction for text files
    # try unstructured first
    result = parse_document(filepath)
    if result:
        return result

    # fallback for text files
    filepath = Path(filepath)
    if filepath.suffix.lower() == '.txt':
        try:
            text = filepath.read_text(encoding='utf-8')
            return {
                'text': text,
                'filename': filepath.name,
                'file_type': '.txt',
                'parse_time': 0.01,
                'char_count': len(text),
                'word_count': len(text.split()),
                'fallback': True
            }
        except Exception as e:
            logger.error(f"Fallback text read failed: {e}")

    return None


def detect_document_type(text):
    # detect if text is a cv or job posting based on content

    # returns: 'cv' or 'job' (best guess based on keywords)

    text_lower = text.lower()

    # cv indicators
    cv_keywords = [
        'resume', 'curriculum vitae', 'work experience', 'education',
        'skills:', 'references', 'objective', 'career summary',
        'professional experience', 'employment history', 'qualifications'
    ]

    # job posting indicators
    job_keywords = [
        'job description', 'responsibilities', 'requirements:',
        'we are looking', 'apply now', 'salary range', 'benefits:',
        'about the role', 'about us', 'position:', 'how to apply',
        'equal opportunity', 'must have', 'nice to have'
    ]

    cv_score = sum(1 for kw in cv_keywords if kw in text_lower)
    job_score = sum(1 for kw in job_keywords if kw in text_lower)

    if cv_score > job_score:
        return 'cv'
    elif job_score > cv_score:
        return 'job'
    else:
        # default to cv (more common for interactive demo)
        return 'cv'


def test_parser():
    # test parser with sample files
    print("Document Parser Test")

    # test with a simple text file
    test_file = Path("incoming/cv/test_cv.txt")
    test_file.parent.mkdir(parents=True, exist_ok=True)

    test_content = '''
    John Doe
    Software Engineer

    Work Experience:
    - Senior Developer at TechCorp (2020-present)
    - Developer at StartupXYZ (2018-2020)

    Skills:
    Python, JavaScript, Django, React, PostgreSQL, AWS

    Education:
    B.S. Computer Science, State University, 2018
    '''

    test_file.write_text(test_content)

    try:
        # parse test file
        result = parse_document(test_file)
        if result:
            print(f"Parsed: {result['filename']}")
            print(f"  Words: {result['word_count']}")
            print(f"  Type detected: {detect_document_type(result['text'])}")
            print(f"  Time: {result['parse_time']:.2f}s")
            print("\nTest passed!")
        else:
            print("Parsing failed!")

    finally:
        # cleanup
        if test_file.exists():
            test_file.unlink()


if __name__ == "__main__":
    test_parser()
