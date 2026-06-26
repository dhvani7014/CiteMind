const THEME_KEY = "citemind-theme";

function applySavedTheme() {
    const savedTheme = localStorage.getItem(THEME_KEY);

    if (savedTheme === "mono") {
        document.body.classList.add("mono-theme");
    } else {
        document.body.classList.remove("mono-theme");
    }
}

function createThemeToggle() {
    const toggle = document.createElement("button");
    toggle.className = "theme-toggle";
    toggle.type = "button";

    function updateLabel() {
        const isMono = document.body.classList.contains("mono-theme");
        toggle.textContent = isMono ? "Default Theme" : "Mono Theme";
    }

    toggle.addEventListener("click", () => {
        const isMono = document.body.classList.toggle("mono-theme");

        if (isMono) {
            localStorage.setItem(THEME_KEY, "mono");
        } else {
            localStorage.setItem(THEME_KEY, "default");
        }

        updateLabel();
    });

    updateLabel();
    document.body.appendChild(toggle);
}

applySavedTheme();

document.addEventListener("DOMContentLoaded", () => {
    createThemeToggle();
});