# CiteMind Evaluation Questions

This file contains test questions used to evaluate CiteMind’s retrieval-augmented generation pipeline.

The goal is to check whether CiteMind can:
- retrieve relevant paper sections
- answer using only uploaded paper content
- include source citations
- avoid hallucinating unsupported details
- handle vague study requests properly

---

## 1. General Understanding

### Question
What is the main idea of this paper?

### Expected Behavior
The answer should explain the paper’s core problem, proposed approach, and why it matters.

### Checks
- Uses retrieved paper context
- Includes citations
- Does not invent unsupported claims

---

## 2. Research Problem

### Question
What problem does this paper solve?

### Expected Behavior
The answer should identify the research problem or motivation described in the paper.

### Checks
- Mentions the problem clearly
- Uses source citations
- Avoids generic summaries

---

## 3. Methodology

### Question
Explain the methodology in simple terms.

### Expected Behavior
The answer should break down the method, model, algorithm, architecture, or experimental process in a student-friendly way.

### Checks
- Explains steps clearly
- Mentions technical details only if found in the paper
- Includes citations

---

## 4. Contributions

### Question
What are the main contributions of this paper?

### Expected Behavior
The answer should list the major contributions claimed or supported by the paper.

### Checks
- Uses bullet points if useful
- Keeps claims grounded in the text
- Includes citations

---

## 5. Results

### Question
What are the key results?

### Expected Behavior
The answer should describe experiments, metrics, comparisons, and findings only if present in the retrieved paper sections.

### Checks
- Does not invent numbers
- Explains what the results mean
- Includes citations

---

## 6. Limitations

### Question
What are the limitations of this paper?

### Expected Behavior
The answer should identify stated limitations or say if the retrieved sections do not clearly mention limitations.

### Checks
- Does not invent criticism
- Clearly separates supported information from missing information
- Includes citations if limitations are found

---

## 7. Study Mode Clarification

### Question
study mode

### Expected Behavior
CiteMind should ask what type of study mode the user wants.

### Expected Response Example
What kind of study mode would you like?

1. Flashcards
2. Quiz questions
3. Simple summary notes
4. Key concepts
5. Exam-style questions

### Checks
- Does not immediately generate a long answer
- Offers clear options
- Waits for the user to clarify

---

## 8. Flashcards

### Question
Give me 5 flashcards from this paper.

### Expected Behavior
The answer should generate flashcards using the required format.

### Expected Format
Q1: question  
A1: answer

Q2: question  
A2: answer

### Checks
- Uses Q1/A1 format
- Avoids markdown headings
- Uses information from the paper only

---

## 9. Hallucination Check

### Question
What does this paper say about blockchain?

### Expected Behavior
If blockchain is not discussed in the retrieved paper sections, CiteMind should say it cannot find enough relevant information.

### Checks
- Does not make up an answer
- Does not use outside knowledge
- Clearly says the retrieved paper sections do not support the answer

---

## 10. Source Grounding

### Question
Which source supports your answer?

### Expected Behavior
The answer should refer to the source labels returned by the system, such as [Source 1] or [Source 2].

### Checks
- Sources match retrieved chunks
- No fake citations
- Source labels are readable