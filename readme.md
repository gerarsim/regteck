# LexAI - Analyseur de conformité réglementaire

LexAI est un outil d'analyse de conformité qui vérifie les documents juridiques et financiers selon les réglementations luxembourgeoises, notamment RGPD, CSSF et AML.

## 🚀 Fonctionnalités

- **Analyse de documents** : Téléchargez et analysez des PDF, DOCX et fichiers texte
- **Détection intelligente** : Identification automatique du type de document et de la langue
- **Conformité multi-réglementations** : Vérification selon RGPD, CSSF 18/698, AML et plus
- **Recommandations personnalisées** : Suggestions pour améliorer la conformité
- **Tableau de bord administrateur** : Gestion des utilisateurs et des règles de conformité
- **Interface multilingue** : Support du français, anglais, allemand et espagnol

## 📋 Types de documents pris en charge

- Contrats et accords juridiques
- Politiques et procédures de conformité
- Rapports annuels
- Bilans financiers
- Documents de conformité bancaire

## 🔧 Installation

### Prérequis

- Python 3.9+
- Tesseract OCR (optionnel, pour l'extraction de texte à partir d'images)

### Installation des dépendances

```bash
# Cloner le dépôt
git clone https://github.com/yourusername/lexai.git
cd lexai

# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement virtuel
# Sur Windows:
venv\Scripts\activate
# Sur macOS/Linux:
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt