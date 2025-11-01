# Second-Me Installation Guide für Ubuntu/Debian

Eine vollautomatische, benutzerfreundliche Installation für Ubuntu und Debian Linux-Systeme.

## 🚀 Schnellstart (1-Befehl-Installation)

```bash
curl -fsSL https://raw.githubusercontent.com/mindverse/Second-Me/master/install-ubuntu.sh | bash
```

**Oder** wenn Sie das Repository bereits geklont haben:

```bash
cd Second-Me
chmod +x install-ubuntu.sh
./install-ubuntu.sh
```

## 📋 Was das Script macht

Das Installations-Script führt automatisch folgende Schritte durch:

### 1. **System-Prüfung**
- ✅ Betriebssystem-Erkennung (Ubuntu/Debian)
- ✅ RAM-Prüfung (Minimum 8 GB empfohlen)
- ✅ Festplatten-Platz (Minimum 20 GB empfohlen)
- ✅ GPU-Erkennung (NVIDIA CUDA)

### 2. **Interaktive Installation**

Das Script fragt Sie nach Ihrer bevorzugten Installationsmethode:

**Option 1: Docker Installation (Empfohlen)**
- ✓ Einfachste Installation
- ✓ Alle Abhängigkeiten vorinstalliert
- ✓ Isolierte Umgebung
- ✓ Automatische GPU-Erkennung

**Option 2: Lokale Installation**
- ✓ Bessere Performance
- ✓ Mehr Kontrolle
- ✓ Direkter System-Zugriff

**Option 3: Beides**
- ✓ Maximale Flexibilität

### 3. **Automatische Abhängigkeits-Installation**

#### Für Docker:
- Docker Engine
- Docker Compose
- NVIDIA Container Toolkit (wenn GPU vorhanden)

#### Für Lokale Installation:
- Python 3.12+
- Node.js 20.x LTS
- Poetry (Python Dependency Management)
- CMake
- SQLite3
- Build-Tools

### 4. **Projekt-Setup**
- Repository-Konfiguration
- Abhängigkeiten installieren
- Umgebungsvariablen einrichten
- Optimierungen für Ihr System

### 5. **Convenience Scripts**
- `start-second-me.sh` - Startet Second-Me
- `stop-second-me.sh` - Stoppt Second-Me

## 💻 System-Anforderungen

### Minimum (CPU-Only):
- **OS:** Ubuntu 20.04+ oder Debian 11+
- **RAM:** 8 GB (mit "low" Datensynthese-Modus)
- **Disk:** 20 GB freier Speicher
- **CPU:** 4 Kerne empfohlen

### Empfohlen:
- **RAM:** 16 GB+
- **Disk:** 50 GB+ freier Speicher
- **GPU:** NVIDIA GPU mit 6+ GB VRAM (optional, aber empfohlen)
- **CPU:** 8+ Kerne

### Optimal (für größere Modelle):
- **RAM:** 32 GB+
- **GPU:** NVIDIA GPU mit 12+ GB VRAM
- **Disk:** 100 GB+ NVMe SSD

## 📖 Detaillierte Installationsanleitung

### Schritt 1: Repository klonen (optional)

Wenn Sie das Script nicht direkt von GitHub ausführen möchten:

```bash
git clone https://github.com/mindverse/Second-Me.git
cd Second-Me
```

### Schritt 2: Installations-Script ausführen

```bash
chmod +x install-ubuntu.sh
./install-ubuntu.sh
```

### Schritt 3: Interaktive Fragen beantworten

Das Script wird Sie fragen:

1. **Installationstyp wählen** (Docker/Lokal/Beides)
2. **GPU-Support aktivieren?** (Nur wenn NVIDIA GPU erkannt wurde)
3. Das Script führt dann alle erforderlichen Installationen durch

### Schritt 4: Nach der Installation

#### Docker-Installation:
```bash
# Starten
./start-second-me.sh
# oder
make docker-up

# Zugriff
# Web-Interface: http://localhost:3000
# Backend API:   http://localhost:8002

# Stoppen
./stop-second-me.sh
# oder
make docker-down
```

#### Lokale Installation:
```bash
# Starten
./start-second-me.sh
# oder
make start

# Zugriff
# Web-Interface: http://localhost:3000
# Backend API:   http://localhost:8002

# Stoppen
./stop-second-me.sh
# oder
make stop
```

## 🔧 Fehlerbehebung

### Problem: "Permission denied" bei Docker

**Lösung:**
```bash
# Nach Docker-Installation:
newgrp docker
# oder ausloggen und wieder einloggen
```

### Problem: "Port already in use"

**Lösung:**
```bash
# Prüfen, welcher Prozess Port 3000/8002 verwendet
sudo lsof -i :3000
sudo lsof -i :8002

# Prozess beenden oder anderen Port in .env konfigurieren
```

### Problem: Installation schlägt fehl

**Lösung:**
```bash
# Log-Datei überprüfen
cat install-ubuntu.log

# Manuelle Installation versuchen
make setup
```

### Problem: Training bricht ab oder wirkt "hängengeblieben"

**Lösungen:**

1. **Log-Datei überwachen:**
```bash
# Docker
docker logs -f second-me-backend

# Lokal
tail -f logs/train/train.log
```

2. **Speicher-Optimierung (bei wenig RAM):**

Bearbeiten Sie `.env`:
```bash
DATA_SYNTHESIS_MODE=low
CONCURRENCY_THREADS=1
```

3. **Starten Sie mit kleinen Daten:**
- Laden Sie zuerst nur 1-2 kleine Dokumente (< 10 Seiten) hoch
- Lassen Sie den kompletten Training-Prozess durchlaufen
- Fügen Sie dann mehr Daten hinzu

### Problem: GPU wird nicht erkannt (Docker)

**Lösung:**
```bash
# NVIDIA Container Toolkit installieren
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# GPU-Modus aktivieren
make docker-use-gpu
make docker-up
```

## 🎯 Performance-Tipps

### Für 8 GB RAM Systeme:
```bash
# In .env setzen:
DATA_SYNTHESIS_MODE=low
CONCURRENCY_THREADS=1

# Kleines Modell wählen (0.5B oder 1.5B)
# In der Web-UI beim Training
```

### Für 16+ GB RAM Systeme:
```bash
DATA_SYNTHESIS_MODE=medium
CONCURRENCY_THREADS=2
```

### Für 32+ GB RAM Systeme mit GPU:
```bash
DATA_SYNTHESIS_MODE=high
CONCURRENCY_THREADS=4
```

## 📝 Nützliche Befehle

```bash
# Alle verfügbaren Befehle anzeigen
make help

# Service-Status prüfen
make status

# Services neustarten
make restart

# Docker-Container prüfen
docker ps

# GPU-Support in Docker prüfen
make docker-check-cuda

# Logs anzeigen (Docker)
docker logs -f second-me-backend
docker logs -f second-me-frontend

# Logs anzeigen (Lokal)
tail -f logs/train/train.log
tail -f logs/app.log
```

## 🔄 Updates

Um Second-Me zu aktualisieren:

```bash
# Repository aktualisieren
git pull origin master

# Docker: Container neu bauen
make docker-down
make docker-build
make docker-up

# Lokal: Abhängigkeiten aktualisieren
make setup
make restart
```

## 🆘 Hilfe bekommen

Wenn Sie auf Probleme stoßen:

1. **Prüfen Sie die Log-Dateien:**
   - Installation: `install-ubuntu.log`
   - Docker Backend: `docker logs second-me-backend`
   - Lokales Training: `logs/train/train.log`

2. **Suchen Sie in der FAQ:**
   - https://secondme.gitbook.io/secondme/faq

3. **Fragen Sie in der Community:**
   - Discord: https://discord.gg/GpWHQNUwrg
   - GitHub Issues: https://github.com/mindverse/Second-Me/issues

4. **Dokumentation:**
   - GitBook: https://secondme.gitbook.io/secondme/
   - Homepage: https://home.second.me/

## 📊 Installations-Zeitplan

Ungefähre Installationszeiten:

| Schritt | Docker | Lokal |
|---------|--------|-------|
| System-Abhängigkeiten | 5-10 Min | 5-10 Min |
| Docker Installation | 5 Min | - |
| Python/Node.js | - | 10 Min |
| Image Build / Setup | 10-30 Min | 15-30 Min |
| **Gesamt** | **20-45 Min** | **30-50 Min** |

*Zeiten abhängig von Internet-Geschwindigkeit und System-Performance*

## ✅ Installations-Checkliste

- [ ] System-Anforderungen geprüft (8+ GB RAM, 20+ GB Disk)
- [ ] Installations-Script heruntergeladen/geklont
- [ ] Script ausgeführt (`./install-ubuntu.sh`)
- [ ] Installationstyp gewählt (Docker/Lokal)
- [ ] Installation erfolgreich abgeschlossen
- [ ] Second-Me gestartet (`./start-second-me.sh`)
- [ ] Web-Interface erreichbar (http://localhost:3000)
- [ ] Erste Dokumente hochgeladen
- [ ] Training-Prozess gestartet und überwacht

## 🎉 Nach der Installation

Herzlichen Glückwunsch! Second-Me ist jetzt installiert.

### Nächste Schritte:

1. **Öffnen Sie die Web-Oberfläche:** http://localhost:3000
2. **Folgen Sie dem User Tutorial:** https://secondme.gitbook.io/secondme/getting-started
3. **Laden Sie Ihre ersten Dokumente hoch**
4. **Starten Sie das Training Ihres AI-Selbst**
5. **Erkunden Sie die Features:**
   - Roleplay Mode
   - AI Space (Network Features)
   - Memory Management

**Viel Erfolg mit Second-Me!** 🚀
