# GridWorld RL Agent - 100% Success Rate & 90% Generalization

Questo progetto implementa un agente di Reinforcement Learning capace di risolvere complessi livelli GridWorld generati proceduralmente. L'agente deve navigare in una griglia, evitare ostacoli e zone di rischio, trovare una chiave per sbloccare una porta e raggiungere l'obiettivo finale.

Il progetto è stato sviluppato e ottimizzato per raggiungere prestazioni di livello umano (e oltre) in termini di efficienza e robustezza.

## 🏆 Risultati Raggiunti (26/11/2025)

Dopo un training intensivo di **5 Milioni di step** su hardware accelerato (GPU + 12 CPU cores), il modello finale (`dqn_final.zip`) ha ottenuto:

*   **100% Success Rate** sulla Test Suite ufficiale (Livelli 1-5).
*   **90% Generalization Rate** su livelli procedurali mai visti prima (Test Set di 10 livelli).
*   **Efficienza Ottimale:** L'agente risolve i livelli con un numero di passi vicino al minimo teorico, senza loop o esitazioni.

## 🧠 Architettura e Configurazione

La soluzione vincente si basa su una combinazione di tecniche avanzate di RL:

*   **Algoritmo:** **Dueling DQN** (Deep Q-Network con architettura Dueling per separare la stima del valore dello stato dal vantaggio dell'azione).
*   **Policy:** **CNN** (Convolutional Neural Network) personalizzata per processare la griglia come un'immagine a 6 canali.
*   **Input:** **Frame Stacking (4 frames)** per dare all'agente la percezione del movimento e della direzione.
*   **Ambiente:**
    *   Griglia **8x8**.
    *   **Reward Shaping:** Chiave (+10.0), Goal (+50.0), Penalità Ripetizione (-1.0) per eliminare i loop.
*   **Training Strategy:** **Mixed Training** (80% Livelli Procedurali / 20% Livelli Fissi) per massimizzare la generalizzazione mantenendo la stabilità.

## 🛠️ Installazione

Assicurati di avere Python 3.10+ installato.

```bash
pip install -r requirements.txt
```

## 🚀 Utilizzo

### 1. Addestramento (Training)
Per replicare il training del modello finale:

```bash
python agents/train.py --train-mixed --timesteps 5000000
```
*Nota: Questo comando utilizzerà automaticamente tutti i core della CPU disponibili per la generazione dei dati (SubprocVecEnv) e la GPU per l'aggiornamento della rete.*

### 2. Valutazione (Evaluation)
Per valutare il modello sulla suite di test standard (Livelli 1-5):

```bash
python agents/evaluate.py --model_path output/models/dqn_final.zip --run_suite
```

Per testare la generalizzazione su nuovi livelli procedurali:

```bash
# Genera nuovi livelli di test
python generate_test_set.py

# Valuta il modello su questi nuovi livelli
python agents/evaluate.py --model_path output/models/dqn_final.zip --test_folder levels/test_set
```

### 3. Visualizzazione
Per vedere l'agente in azione e salvare una GIF del miglior episodio:

```bash
python agents/evaluate.py --model_path output/models/dqn_final.zip --level 5 --save_gif
```

## 📂 Struttura del Progetto

*   `agents/`: Codice sorgente per il training (`train.py`) e la valutazione (`evaluate.py`).
*   `gridworld/`: Logica dell'ambiente, configurazione e generatore procedurale dei livelli.
*   `levels/`: File JSON dei livelli (Training set e Test set).
*   `output/`:
    *   `models/`: Checkpoint dei modelli addestrati.
    *   `logs/`: Log di training per TensorBoard/CSV.
    *   `plots/`: Grafici delle performance (Reward, Success Rate).
    *   `gifs/`: Replay visivi degli episodi.

## 📊 Performance Visiva

L'agente dimostra un comportamento intelligente: esplora in modo sicuro, identifica la chiave, pianifica il percorso di ritorno verso la porta e scatta verso l'obiettivo.

*(Inserire qui le GIF generate nella cartella output/gifs)*

---
**Autore:** FrancescoFalcon
**Data:** Novembre 2025
