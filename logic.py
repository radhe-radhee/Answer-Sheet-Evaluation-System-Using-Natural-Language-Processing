import re
import numpy as np
import cv2
import logging
from pdf2image import convert_from_path
import pytesseract
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from collections import defaultdict
from datetime import datetime
import json
from tabulate import tabulate

# Initialize models globally
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
nlp = spacy.load("en_core_web_md")

# Configure logging
logging.basicConfig(filename='exam_evaluation.log', level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_floats(obj):
    """Convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_floats(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_floats(item) for item in obj]
    return obj

def pdf_to_images(pdf_path):
    """Convert PDF pages to images and extract text using OCR."""
    try:
        images = convert_from_path(pdf_path)
        ocr_outputs = []
        for i, image in enumerate(images):
            image = image.convert("L")  # Convert to grayscale
            image = cv2.medianBlur(np.array(image), 3)
            _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ocr_output = pytesseract.image_to_string(image)
            ocr_outputs.append(ocr_output.strip())
        return images, ocr_outputs
    except Exception as e:
        logger.error(f"PDF to image conversion failed: {str(e)}")
        raise

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF using OCR."""
    try:
        _, ocr_outputs = pdf_to_images(pdf_path)
        return "\n".join(ocr_outputs)
    except Exception as e:
        logger.error(f"Text extraction failed: {str(e)}")
        raise

def extract_questions(text):
    """
    Extracts numbered questions with marks using regex.
    Returns: List of (question_number, question_text, marks) tuples.
    """
    questions = []
    last_marks = 10  # Default marks if missing

    question_pattern = r"(\d+)\.\s*((?:.*?(?:\n(?!\d+\.))*)?)\s*\[(\d+)\]"

    matches = re.finditer(question_pattern, text, re.MULTILINE | re.DOTALL)

    for match in matches:
        question_number = int(match.group(1))
        question_text = match.group(2).strip()
        marks = next((int(m) for m in match.groups()[2:] if m is not None and m.isdigit()), last_marks)
        questions.append((question_number, question_text, marks))

    return questions

def extract_answers(answer_pdf_path):
    """
    Extracts numbered answers from PDF using OCR with enhanced text processing.
    Returns: List of (answer_number, answer_text) tuples.
    """
    try:
        _, ocr_outputs = pdf_to_images(answer_pdf_path)
        text = "\n".join(ocr_outputs)

        answer_pattern = r"(\d+)\.\s*((?:.*?(?:\n|$))+?)\n*(?=\d+\.|\Z)"
        matches = re.finditer(answer_pattern, text, re.DOTALL)

        answers = []
        for match in matches:
            answer_number = int(match.group(1))
            answer_text = re.sub(r'\s+', ' ', match.group(2).strip())
            answer_text = re.sub(r"(?<!\n)\n(?!\n)", " ", answer_text)
            answers.append((answer_number, answer_text))

        return answers

    except Exception as e:
        print(f"Error processing answers: {e}")
        return []

def get_embedding(text, max_len=512, pooling='mean'):
    """Get BERT embedding for text with error handling."""
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len, padding='max_length')
        with torch.no_grad():
            outputs = bert_model(**inputs)

        last_hidden_states = outputs.last_hidden_state.squeeze(0)

        if pooling == 'mean':
            embedding = torch.mean(last_hidden_states, dim=0)
        elif pooling == 'max':
            embedding, _ = torch.max(last_hidden_states, dim=0)
        else:  # CLS token
            embedding = last_hidden_states[0]

        return embedding.numpy()
    except Exception as e:
        logger.error(f"Embedding generation failed for text: {text[:100]}... Error: {str(e)}")
        raise

def semantic_keyword_overlap(student_answer, correct_answer, threshold=0.7):
    """Compute semantic keyword overlap with enhanced matching."""
    try:
        student_doc = nlp(student_answer.lower())
        correct_doc = nlp(correct_answer.lower())

        student_answer = re.sub(r"(?<!\n)\n(?!\n)", " ", student_answer).strip()
        correct_answer = re.sub(r"(?<!\n)\n(?!\n)", " ", correct_answer).strip()

        student_words = {token.lemma_ for token in student_doc
                       if token.is_alpha and not token.is_stop and token.pos_ in ['NOUN', 'VERB', 'ADJ']}
        correct_words = {token.lemma_ for token in correct_doc
                        if token.is_alpha and not token.is_stop and token.pos_ in ['NOUN', 'VERB', 'ADJ']}

        exact_match = len(student_words & correct_words) / len(correct_words) if correct_words else 0

        semantic_match = 0
        matched = set()
        for c_word in correct_words:
            for s_word in student_words:
                if s_word not in matched and nlp.vocab[c_word].similarity(nlp.vocab[s_word]) >= threshold:
                    semantic_match += 1
                    matched.add(s_word)
                    break

        semantic_score = semantic_match / len(correct_words) if correct_words else 0
        return (exact_match + semantic_score) / 2
    except Exception as e:
        logger.error(f"Keyword overlap calculation failed: {str(e)}")
        raise

def generate_feedback(student_answer, correct_answer, score, max_marks):
    """Generate detailed feedback for student answers with enhanced analysis."""
    feedback = {
        'score': f"{score:.1f}/{max_marks}",
        'strengths': [],
        'improvements': [],
        'specific_suggestions': [],
        'key_concepts': []
    }

    score_percentage = score / max_marks
    if score_percentage >= 0.8:
        feedback['strengths'].append("Excellent answer - comprehensive and well-structured")
    elif score_percentage >= 0.6:
        feedback['strengths'].append("Good answer showing solid understanding")
    elif score_percentage >= 0.4:
        feedback['improvements'].append("Fair attempt but needs more depth and accuracy")
    else:
        feedback['improvements'].append("Needs significant improvement in fundamental concepts")

    try:
        student_doc = nlp(student_answer.lower())
        correct_doc = nlp(correct_answer.lower())

        correct_keywords = {token.lemma_ for token in correct_doc
                          if token.is_alpha and not token.is_stop and token.pos_ in ['NOUN', 'VERB']}
        student_keywords = {token.lemma_ for token in student_doc
                          if token.is_alpha and not token.is_stop and token.pos_ in ['NOUN', 'VERB']}

        missing_keywords = correct_keywords - student_keywords
        present_keywords = correct_keywords & student_keywords

        if present_keywords:
            feedback['strengths'].append(f"Contains key concepts: {', '.join(sorted(present_keywords)[:5])}")
            feedback['key_concepts'].extend(sorted(present_keywords))

        if missing_keywords:
            feedback['improvements'].append(f"Missing important concepts: {', '.join(sorted(missing_keywords)[:5])}")
            feedback['specific_suggestions'].append(f"Study these concepts: {', '.join(sorted(missing_keywords)[:3])}")

        student_length = len(student_answer.split())
        correct_length = len(correct_answer.split())

        if student_length < correct_length * 0.5:
            feedback['improvements'].append("Answer is too brief - consider expanding your explanation")
        elif student_length > correct_length * 1.5:
            feedback['improvements'].append("Answer is too verbose - focus on being more concise")

        return feedback
    except Exception as e:
        logger.error(f"Feedback generation failed: {str(e)}")
        feedback['error'] = str(e)
        return feedback

def evaluate_answer(student_answer, correct_answer, max_marks):
    """Evaluate student answer against correct answer with multiple metrics."""
    try:
        student_embedding = get_embedding(student_answer)
        correct_embedding = get_embedding(correct_answer)
        similarity = cosine_similarity([student_embedding], [correct_embedding])[0][0]

        keyword_score = semantic_keyword_overlap(student_answer, correct_answer)

        score = (0.6 * similarity + 0.4 * keyword_score) * max_marks

        feedback = generate_feedback(student_answer, correct_answer, score, max_marks)

        return min(score, max_marks), feedback
    except Exception as e:
        logger.error(f"Answer evaluation failed: {str(e)}")
        return 0, {'error': str(e)}

def evaluate_exam(question_pdf_path, correct_answer_pdf_path, student_answer_pdf_path):
    """Main evaluation function with comprehensive results."""
    results = {
        "metadata": {
            "question_pdf": question_pdf_path,
            "correct_answer_pdf": correct_answer_pdf_path,
            "student_answer_pdf": student_answer_pdf_path,
            "timestamp": datetime.now().isoformat(),
            "system_version": "1.0"
        },
        "questions": [],
        "statistics": {
            "total_questions": 0,
            "answered_questions": 0,
            "unanswered_questions": 0,
            "score_distribution": defaultdict(int)
        },
        "summary": {},
        "warnings": [],
        "errors": []
    }

    try:
        question_text = extract_text_from_pdf(question_pdf_path)
        questions = extract_questions(question_text)
        results['statistics']['total_questions'] = len(questions)

        correct_answers = extract_answers(correct_answer_pdf_path)
        correct_answers_dict = {num: ans for num, ans in correct_answers}
        student_answers = extract_answers(student_answer_pdf_path)
        student_answers_dict = {num: ans for num, ans in student_answers}

        total_obtained = 0
        total_possible = 0
        score_distribution = defaultdict(int)

        for question in questions:
            q_num, q_text, marks = question
            result = {
                "question_number": q_num,
                "question_text": q_text,
                "max_marks": float(marks),
                "evaluation": {}
            }

            if q_num not in student_answers_dict:
                result["evaluation"] = {
                    "status": "unanswered",
                    "marks_obtained": 0,
                    "feedback": {
                        "score": f"0/{marks}",
                        "improvements": ["Question not attempted"]
                    }
                }
                results['statistics']['unanswered_questions'] += 1
                results['warnings'].append(f"Question {q_num} was not answered")
            else:
                if q_num not in correct_answers_dict:
                    results['errors'].append(f"No correct answer for question {q_num}")
                    continue

                score, feedback = evaluate_answer(
                    student_answers_dict[q_num],
                    correct_answers_dict[q_num],
                    float(marks)
                )

                result["evaluation"] = {
                    "status": "evaluated",
                    "marks_obtained": float(score),
                    "feedback": feedback,
                    "student_answer": student_answers_dict[q_num],
                    "correct_answer": correct_answers_dict[q_num]
                }
                total_obtained += score
                results['statistics']['answered_questions'] += 1

                percentage = (score / marks) * 100
                if percentage >= 80:
                    score_distribution['excellent'] += 1
                elif percentage >= 60:
                    score_distribution['good'] += 1
                elif percentage >= 40:
                    score_distribution['fair'] += 1
                else:
                    score_distribution['poor'] += 1

            total_possible += float(marks)
            results["questions"].append(result)

        results['statistics']['score_distribution'] = dict(score_distribution)

        total_obtained = float(total_obtained)
        total_possible = float(total_possible)

        results['summary'] = {
            "total_marks_obtained": float(total_obtained),
            "total_possible_marks": float(total_possible),
            "percentage_score": float(total_obtained / total_possible * 100) if total_possible > 0 else 0.0,
            "average_score_per_question": float(total_obtained / len(questions)) if len(questions) > 0 else 0.0,
            "performance_category": "Excellent" if (total_obtained / total_possible) >= 0.8 else
                                  "Good" if (total_obtained / total_possible) >= 0.6 else
                                  "Fair" if (total_obtained / total_possible) >= 0.4 else "Poor"
        }

    except Exception as e:
        results['errors'].append(str(e))
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)

    return results

def generate_report(results, output_format='both'):
    """Generate comprehensive evaluation report in multiple formats."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        def convert_floats(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_floats(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_floats(item) for item in obj]
            return obj

        results = convert_floats(results)

        if output_format in ('text', 'both'):
            report = []
            report.append("="*80)
            report.append("EXAM EVALUATION REPORT".center(80))
            report.append("="*80)
            report.append(f"\n📅 Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"\n📊 Summary:")
            report.append(f"  - Total Questions: {results['statistics']['total_questions']}")
            report.append(f"  - Answered: {results['statistics']['answered_questions']}")
            report.append(f"  - Unanswered: {results['statistics']['unanswered_questions']}")
            report.append(f"  - Total Score: {results['summary']['total_marks_obtained']:.1f}/{results['summary']['total_possible_marks']}")
            report.append(f"  - Percentage: {results['summary']['percentage_score']:.1f}%")
            report.append(f"  - Performance: {results['summary']['performance_category']}")

            report.append("\n📝 Question-wise Results:")
            table_data = []
            for q in results['questions']:
                eval = q['evaluation']
                status = "✅" if eval['status'] == 'evaluated' else "❌"
                feedback = eval.get('feedback', {})
                key_feedback = "\n".join(feedback.get('improvements', [])[:1] or ["-"])

                table_data.append([
                    q["question_number"],
                    f"{eval.get('marks_obtained', 0):.1f}/{q['max_marks']}",
                    status,
                    key_feedback
                ])

            report.append(tabulate(table_data,
                                headers=["Q#", "Score", "Status", "Key Feedback"],
                                tablefmt="grid",
                                maxcolwidths=[None, None, None, 40]))

            report.append("\n📈 Performance Analysis:")
            dist = results['statistics']['score_distribution']
            report.append(f"  - Excellent (≥80%): {dist.get('excellent', 0)} questions")
            report.append(f"  - Good (60-79%): {dist.get('good', 0)} questions")
            report.append(f"  - Fair (40-59%): {dist.get('fair', 0)} questions")
            report.append(f"  - Poor (<40%): {dist.get('poor', 0)} questions")

            all_concepts = set()
            for q in results['questions']:
                if 'feedback' in q['evaluation'] and 'key_concepts' in q['evaluation']['feedback']:
                    all_concepts.update(q['evaluation']['feedback']['key_concepts'])

            if all_concepts:
                report.append("\n🔑 Key Concepts Covered:")
                report.append(", ".join(sorted(all_concepts)[:15]) + ("..." if len(all_concepts) > 15 else ""))

            text_report = "\n".join(report)

            if output_format == 'text':
                print(text_report)
            else:
                with open(f"evaluation_report_{timestamp}.txt", 'w') as f:
                    f.write(text_report)

        if output_format in ('json', 'both'):
            json_filename = f"evaluation_report_{timestamp}.json"
            with open(json_filename, 'w') as f:
                json.dump(results, f, indent=2)

            if output_format == 'json':
                print(f"Report saved to {json_filename}")

        return True

    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        return False
