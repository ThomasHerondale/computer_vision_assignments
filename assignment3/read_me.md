# Per vedere la progress bar nel terminale:
    1. in alto, sulla configurazione accanto ai tasti play/debug, cliccate sul nome della run configuration
    2. cliccate su edit configurations
    3. cliccate su modify options
    4. nella sezione 'python' abilitate 'Emulate terminal in output console'

# Path del dataset
Il codice assume che le cartelle siano combinate così: \
**assingment3/MOT17/test e assignment3/MOT17/train**

# Funzioni da usare
In pratica, l'unica funzione che dovreste usare è get_detections. Fa tutto lei,
capisce se c'è già un file di cache e lo carica, altrimenti genera le detection
e nel frattempo le salva nel file.