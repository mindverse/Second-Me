# 🇩🇪 Second-Me Installation für Ubuntu/Debian (Deutsch)

## ⚡ Schnellinstallation (Empfohlen)

### Eine Zeile - Fertig!

```bash
curl -fsSL https://raw.githubusercontent.com/mindverse/Second-Me/master/install-ubuntu.sh | bash
```

### Oder Repository zuerst klonen:

```bash
git clone https://github.com/mindverse/Second-Me.git
cd Second-Me
chmod +x install-ubuntu.sh
./install-ubuntu.sh
```

Das Script führt Sie durch eine **vollautomatische Installation** mit interaktiven Fragen.

---

## 📋 Was macht das Installations-Script?

### ✨ Automatische Erkennung:
- ✅ Ubuntu/Debian Version
- ✅ RAM und Festplatten-Platz
- ✅ NVIDIA GPU (für CUDA-Beschleunigung)
- ✅ Bestehende Software (Docker, Python, Node.js)

### 🎯 Interaktive Installation:

Das Script fragt Sie:

**1. Welche Installation möchten Sie?**
- **Docker** (Empfohlen - Am einfachsten)
- **Lokal** (Bessere Performance)
- **Beides** (Maximale Flexibilität)

**2. GPU-Support aktivieren?** (Falls NVIDIA GPU vorhanden)

### 🔧 Was wird installiert:

#### Docker-Installation:
- Docker Engine & Docker Compose
- NVIDIA Container Toolkit (bei GPU)
- Alle Second-Me Container

#### Lokale Installation:
- Python 3.12+
- Node.js 20.x LTS
- Poetry (Python Package Manager)
- CMake & Build-Tools
- Alle Second-Me Abhängigkeiten

---

## 💻 System-Anforderungen

### Minimum:
- **OS:** Ubuntu 20.04+ oder Debian 11+
- **RAM:** 8 GB
- **Disk:** 20 GB frei
- **CPU:** 4 Kerne

### Empfohlen:
- **RAM:** 16 GB
- **Disk:** 50 GB frei
- **GPU:** NVIDIA mit 6+ GB VRAM (optional)
- **CPU:** 8 Kerne

---

## 🚀 Nach der Installation

### Starten:
```bash
./start-second-me.sh
```

### Zugriff:
- **Web-Interface:** http://localhost:3000
- **Backend API:** http://localhost:8002

### Stoppen:
```bash
./stop-second-me.sh
```

---

## 🔍 Training überwachen

Das Training kann **1-6 Stunden** dauern, je nach Datenmenge und Hardware.

### Fortschritt anzeigen:

**Docker:**
```bash
docker logs -f second-me-backend
```

**Lokal:**
```bash
tail -f logs/train/train.log
```

Sie sollten Ausgaben wie diese sehen:
```
Training progress: 15% (36/240)
Training progress: 25% (60/240)
...
```

---

## 🛠️ Häufige Probleme & Lösungen

### Problem: "Training hängt fest"

**Ursache:** Der Prozess läuft wahrscheinlich noch, braucht aber Zeit.

**Lösung:**
1. **Log-Datei prüfen** (siehe oben)
2. Wenn wirklich hängt: Mit kleinen Daten starten
3. RAM-Optimierung in `.env`:
   ```bash
   DATA_SYNTHESIS_MODE=low
   CONCURRENCY_THREADS=1
   ```

### Problem: "Permission denied" bei Docker

**Lösung:**
```bash
newgrp docker
# oder ausloggen und wieder einloggen
```

### Problem: "Port 3000 or 8002 already in use"

**Lösung:**
```bash
# Prüfen welcher Prozess den Port verwendet
sudo lsof -i :3000
sudo lsof -i :8002

# Prozess beenden oder andere Ports konfigurieren
```

### Problem: Nicht genug RAM (nur 8 GB)

**Lösung:**
Bearbeiten Sie die `.env` Datei:
```bash
# Öffnen
nano .env

# Hinzufügen/Ändern:
DATA_SYNTHESIS_MODE=low
CONCURRENCY_THREADS=1

# Speichern: Ctrl+O, Enter, Ctrl+X
```

Dann neu starten:
```bash
./stop-second-me.sh
./start-second-me.sh
```

---

## 💡 Performance-Tipps

### Bei 8 GB RAM:
- ✓ `DATA_SYNTHESIS_MODE=low` setzen
- ✓ Mit 1-2 kleinen Dokumenten starten
- ✓ Kleineres Modell wählen (0.5B oder 1.5B)

### Bei 16+ GB RAM:
- ✓ `DATA_SYNTHESIS_MODE=medium`
- ✓ Mittlere Modelle möglich (2B-3B)

### Bei 32+ GB RAM + GPU:
- ✓ `DATA_SYNTHESIS_MODE=high`
- ✓ Große Modelle möglich (7B+)
- ✓ GPU-Beschleunigung aktivieren

---

## 📚 Nützliche Befehle

```bash
# Alle Befehle anzeigen
make help

# Status prüfen
make status

# Neu starten
make restart

# GPU-Support prüfen (Docker)
make docker-check-cuda

# Logs anzeigen
docker logs -f second-me-backend  # Docker
tail -f logs/train/train.log       # Lokal
```

---

## 📖 Dokumentation & Hilfe

- **Homepage:** https://home.second.me/
- **Ausführliche Anleitung (Englisch):** [docs/INSTALLATION-UBUNTU.md](docs/INSTALLATION-UBUNTU.md)
- **User Tutorial:** https://secondme.gitbook.io/secondme/getting-started
- **FAQ:** https://secondme.gitbook.io/secondme/faq
- **Discord Community:** https://discord.gg/GpWHQNUwrg
- **GitHub Issues:** https://github.com/mindverse/Second-Me/issues

---

## 🎯 Schritt-für-Schritt Guide

### 1. Installation starten
```bash
./install-ubuntu.sh
```

### 2. Fragen beantworten
- Installationstyp wählen: **1** (Docker - Empfohlen)
- GPU aktivieren: **Y** (wenn vorhanden)

### 3. Warten
- Installation dauert ca. 20-45 Minuten
- Kaffee holen ☕

### 4. Starten
```bash
./start-second-me.sh
```

### 5. Browser öffnen
- Gehen Sie zu: http://localhost:3000

### 6. Erste Schritte
1. ✓ Account erstellen / Einloggen
2. ✓ Profil einrichten
3. ✓ **Kleine** Dokumente hochladen (1-2 PDFs, je <10 Seiten)
4. ✓ Training starten
5. ✓ Logs überwachen (siehe oben)
6. ✓ Nach 1-2 Stunden: Ihr AI-Selbst ist bereit!

### 7. Mehr Daten hinzufügen
Nach erfolgreichem ersten Training:
- Laden Sie mehr Dokumente hoch
- Starten Sie erneut das Training
- Ihr AI-Selbst wird besser und persönlicher

---

## ✅ Installations-Checkliste

- [ ] System-Anforderungen geprüft
- [ ] Repository geklont oder Script heruntergeladen
- [ ] `./install-ubuntu.sh` ausgeführt
- [ ] Installation erfolgreich
- [ ] `./start-second-me.sh` ausgeführt
- [ ] http://localhost:3000 erreichbar
- [ ] Profil erstellt
- [ ] Erste Dokumente hochgeladen
- [ ] Training gestartet
- [ ] Logs zeigen Fortschritt

---

## 🎉 Fertig!

Willkommen bei Second-Me! Sie haben erfolgreich Ihr eigenes AI-System installiert.

**Nächste Schritte:**
1. Erkunden Sie die Web-Oberfläche
2. Laden Sie Ihre Dokumente hoch
3. Trainieren Sie Ihr AI-Selbst
4. Testen Sie die Features (Roleplay, AI Space)
5. Teilen Sie Ihre Erfahrungen in der Community!

**Viel Spaß mit Second-Me!** 🚀

---

## 🐛 Probleme?

1. **Prüfen Sie:** `install-ubuntu.log`
2. **Lesen Sie:** [docs/INSTALLATION-UBUNTU.md](docs/INSTALLATION-UBUNTU.md) (Englisch)
3. **Fragen Sie:** [Discord Community](https://discord.gg/GpWHQNUwrg)
4. **Melden Sie:** [GitHub Issues](https://github.com/mindverse/Second-Me/issues)
