FROM python:3.11-slim

# ── System dependencies (TeX + fonts) ────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        cabextract \
        fontconfig \
        texlive-xetex \
        texlive-fonts-recommended \
        texlive-latex-extra \
    && rm -rf /var/lib/apt/lists/*

# Install Microsoft core fonts (Times New Roman) used by main.tex
RUN mkdir -p /tmp/fonts && cd /tmp/fonts \
    && curl -fsSL -o times32.exe \
       "https://sourceforge.net/projects/corefonts/files/the%20fonts/final/times32.exe/download" \
    && cabextract times32.exe \
    && mkdir -p /usr/share/fonts/truetype/msttcorefonts \
    && find . -iname "*.ttf" -exec cp {} /usr/share/fonts/truetype/msttcorefonts/ \; \
    && fc-cache -fv \
    && cd / && rm -rf /tmp/fonts

# ── Application ───────────────────────────────────────────────────────────────
WORKDIR /app

COPY . .

RUN pip install --no-cache-dir . streamlit

EXPOSE 8080

# Cloud Run sets PORT=8080 by default
CMD ["streamlit", "run", "main_code/app.py", \
     "--server.port=8080", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
