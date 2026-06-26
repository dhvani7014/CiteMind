const pdfInput = document.getElementById("pdfInput");
const fileLabel = document.getElementById("fileLabel");
const uploadButton = document.getElementById("uploadButton");
const uploadStatus = document.getElementById("uploadStatus");
const questionInput = document.getElementById("questionInput");
const askButton = document.getElementById("askButton");
const chatMessages = document.getElementById("chatMessages");

pdfInput.addEventListener("change", () => {
    if (pdfInput.files.length > 0) {
        fileLabel.textContent = pdfInput.files[0].name;
    }
});

uploadButton.addEventListener("click", async () => {
    if (pdfInput.files.length === 0) {
        showUploadStatus("Please choose a PDF file first.", "error");
        return;
    }

    const formData = new FormData();
    formData.append("file", pdfInput.files[0]);

    showUploadStatus("Uploading and processing your paper...", "");

    try {
        const response = await fetch("/upload-pdf", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Upload failed.");
        }

        showUploadStatus(
            `Paper processed successfully. ${data.total_pages} pages and ${data.total_chunks} chunks indexed.`,
            "success"
        );

        addAssistantMessage(
            `I processed **${data.filename}**. You can now ask questions about this paper.`
        );
    } catch (error) {
        showUploadStatus(error.message, "error");
    }
});

askButton.addEventListener("click", askQuestion);

questionInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        askQuestion();
    }
});

async function askQuestion() {
    const question = questionInput.value.trim();

    if (!question) {
        return;
    }

    addUserMessage(question);
    questionInput.value = "";

    const clarification = getClarifyingResponse(question);

    if (clarification) {
        addAssistantMessage(clarification);
        return;
    }

    const loadingMessage = addAssistantMessage("Thinking through the paper...");

    try {
        const response = await fetch("/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                question: question,
                top_k: 5
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Could not answer the question.");
        }

        loadingMessage.remove();

        addAssistantMessage(
            data.answer,
            data.sources || [],
            data.answer_mode,
            data.llm_provider,
            data.llm_model
        );
    } catch (error) {
        loadingMessage.remove();
        addAssistantMessage(error.message);
    }
}

function getClarifyingResponse(question) {
    const cleanQuestion = question.trim().toLowerCase();

    const vagueStudyRequests = [
        "study mode",
        "study",
        "help me study",
        "quiz",
        "quiz me",
        "important",
        "important points",
        "what should i know",
        "prepare me",
        "exam prep",
        "test prep"
    ];

    if (vagueStudyRequests.includes(cleanQuestion)) {
        return `
What kind of study mode would you like?

- Flashcards
- Quiz questions with answers
- Simple study notes
- Key concepts only
- Exam-style questions
- A short summary first, then questions

For example, you can type: **Give me flashcards for the important concepts**.
`;
    }

    const veryShortVagueQuestions = [
        "explain",
        "summary",
        "summarize",
        "notes",
        "concepts",
        "questions"
    ];

    if (veryShortVagueQuestions.includes(cleanQuestion)) {
        return `
Can you be a little more specific?

You can ask something like:

- Explain the main idea of this paper
- Summarize the paper in simple terms
- Give me key concepts from the paper
- Create quiz questions from the paper
- Make flashcards from the paper
`;
    }

    return null;
}

function showUploadStatus(message, type) {
    uploadStatus.textContent = message;
    uploadStatus.className = "status-box";

    if (type) {
        uploadStatus.classList.add(type);
    }
}

function addUserMessage(text) {
    const message = document.createElement("div");
    message.className = "message user";
    message.innerHTML = `<p>${escapeHtml(text)}</p>`;
    chatMessages.appendChild(message);
    scrollChatToBottom();
}

function addAssistantMessage(text, sources = [], mode = null, provider = null, model = null) {
    const message = document.createElement("div");
    message.className = "message assistant";

    const answerHtml = isFlashcardResponse(text)
        ? buildFlashcards(text)
        : formatAssistantText(text);

    const modeText = getModeText(mode, provider, model);
    const sourcesHtml = buildCompactSources(sources);

    message.innerHTML = `
        <div class="assistant-answer">
            ${answerHtml}
        </div>

        ${modeText ? `<div class="answer-meta">${escapeHtml(modeText)}</div>` : ""}

        ${sourcesHtml}
    `;

    chatMessages.appendChild(message);

    setupFlashcardInteractions(message);

    scrollChatToBottom();

    return message;
}

function isFlashcardResponse(text) {
    return /Q\d+:/i.test(text) && /A\d+:/i.test(text);
}

function buildFlashcards(text) {
    const flashcards = [];
    const regex = /Q(\d+):\s*([\s\S]*?)(?=A\1:)\s*A\1:\s*([\s\S]*?)(?=Q\d+:|$)/gi;

    let match;

    while ((match = regex.exec(text)) !== null) {
        const number = match[1];
        const question = match[2].trim();
        const answer = match[3].trim();

        flashcards.push({
            number,
            question,
            answer
        });
    }

    if (flashcards.length === 0) {
        return formatAssistantText(text);
    }

    const cardsJson = encodeURIComponent(JSON.stringify(flashcards));

    return `
        <div class="flashcard-study-mode" data-cards="${cardsJson}" data-current-index="0">
            <div class="flashcard-intro">
                Study mode: click the card to reveal the answer.
            </div>

            <div class="single-flashcard-wrapper">
                ${renderSingleFlashcard(flashcards[0])}
            </div>

            <div class="flashcard-controls">
                <button type="button" class="flashcard-nav prev-card" disabled>Previous</button>
                <span class="flashcard-counter">Card 1 of ${flashcards.length}</span>
                <button type="button" class="flashcard-nav next-card">Next</button>
            </div>
        </div>
    `;
}

function renderSingleFlashcard(card) {
    return `
        <div class="flashcard single-flashcard">
            <div class="flashcard-inner">
                <div class="flashcard-face flashcard-front">
                    <span class="flashcard-label">Question ${escapeHtml(card.number)}</span>
                    <p>${escapeHtml(card.question)}</p>
                    <small>Click to reveal answer</small>
                </div>

                <div class="flashcard-face flashcard-back">
                    <span class="flashcard-label">Answer ${escapeHtml(card.number)}</span>
                    <p>${escapeHtml(card.answer)}</p>
                    <small>Click to flip back</small>
                </div>
            </div>
        </div>
    `;
}

function setupFlashcardInteractions(message) {
    const studyModes = message.querySelectorAll(".flashcard-study-mode");

    studyModes.forEach(studyMode => {
        const cards = JSON.parse(decodeURIComponent(studyMode.dataset.cards));
        const wrapper = studyMode.querySelector(".single-flashcard-wrapper");
        const counter = studyMode.querySelector(".flashcard-counter");
        const prevButton = studyMode.querySelector(".prev-card");
        const nextButton = studyMode.querySelector(".next-card");

        function attachFlipHandler() {
            const cardElement = wrapper.querySelector(".flashcard");

            if (!cardElement) {
                return;
            }

            cardElement.addEventListener("click", () => {
                cardElement.classList.toggle("flipped");
            });
        }

        function updateCard(newIndex) {
            studyMode.dataset.currentIndex = String(newIndex);

            wrapper.innerHTML = renderSingleFlashcard(cards[newIndex]);

            counter.textContent = `Card ${newIndex + 1} of ${cards.length}`;

            prevButton.disabled = newIndex === 0;
            nextButton.disabled = newIndex === cards.length - 1;

            attachFlipHandler();
        }

        attachFlipHandler();

        prevButton.addEventListener("click", event => {
            event.preventDefault();
            event.stopPropagation();

            const currentIndex = Number(studyMode.dataset.currentIndex);

            if (currentIndex > 0) {
                updateCard(currentIndex - 1);
            }
        });

        nextButton.addEventListener("click", event => {
            event.preventDefault();
            event.stopPropagation();

            const currentIndex = Number(studyMode.dataset.currentIndex);

            if (currentIndex < cards.length - 1) {
                updateCard(currentIndex + 1);
            }
        });
    });
}

function getModeText(mode, provider, model) {
    if (mode === "llm") {
        let providerName = "AI";

        if (provider === "openai") {
            providerName = "OpenAI";
        } else if (provider === "groq") {
            providerName = "Groq";
        }

        return model
            ? `Answered with ${providerName} (${model}) using retrieved paper sections`
            : `Answered with ${providerName} using retrieved paper sections`;
    }

    if (mode === "local_fallback") {
        return "Answered with local fallback using retrieved paper sections";
    }

    return "";
}

function buildCompactSources(sources) {
    if (!sources || sources.length === 0) {
        return "";
    }

    const sourceLabels = sources
        .map(source => source.citation || `Source ${source.source_number}`)
        .join(" · ");

    return `
        <details class="sources-compact">
            <summary>Sources used</summary>
            <p>${escapeHtml(sourceLabels)}</p>
        </details>
    `;
}

function formatAssistantText(text) {
    const lines = text.split("\n");
    let html = "";
    let inList = false;

    lines.forEach(rawLine => {
        const line = rawLine.trim();

        if (!line) {
            if (inList) {
                html += "</ul>";
                inList = false;
            }
            return;
        }

        const isBullet =
            line.startsWith("- ") ||
            line.startsWith("* ") ||
            line.startsWith("• ") ||
            line.startsWith("◦ ") ||
            line.startsWith("▪ ");

        if (isBullet) {
            if (!inList) {
                html += "<ul>";
                inList = true;
            }

            const cleanBullet = line
                .replace(/^[-*•◦▪]\s+/, "")
                .trim();

            html += `<li>${formatInlineText(cleanBullet)}</li>`;
        } else {
            if (inList) {
                html += "</ul>";
                inList = false;
            }

            html += `<p>${formatInlineText(line)}</p>`;
        }
    });

    if (inList) {
        html += "</ul>";
    }

    return html;
}

function formatInlineText(text) {
    let safeText = escapeHtml(text);
    safeText = safeText.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    return safeText;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text || "";
    return div.innerHTML;
}

function scrollChatToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}