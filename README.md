# Neural-Networks-in-Electromyography

Ten projekt ma na celu opracowanie modelu klasyfikacyjnego rozpoznającego typ chwytu dłoni na podstawie sygnałów EMG (elektromiograficznych), przy użyciu nowoczesnej architektury Transformera.

Dane wejściowe
Zbiór danych: 1200 próbek (plików), każda przypisana do jednej z 6 klas chwytów. Dane zostały zebrane w ramach [Pracy Inżynierskiej](https://github.com/mburakowski/EMG-Signals-Analysis) 

Struktura próbki:

200 punktów czasowych

8 kanałów EMG (wektor 1×8)

Finalna reprezentacja danych wejściowych: (200, 8)

Rodzaj klasyfikacji: wieloklasowa (6 klas chwytów)

Cel projektu
Zbudowanie modelu typu Transformer, który:

poprawnie klasyfikuje chwyt na podstawie krótkiej sekwencji EMG,

wyciąga informacje czasowe oraz międzykanałowe z danych,

umożliwia łatwe rozszerzanie i interpretację (np. przez attention maps).

Wybór architektury: Transformer
Dlaczego Transformer?

Potrafi modelować długoterminowe zależności czasowe lepiej niż LSTM/GRU.

Umożliwia równoległe przetwarzanie danych, co znacząco skraca czas uczenia.

Pozwala na interpretację modelu dzięki mechanizmowi attention.

Elastyczny — łatwo go rozbudować, dostosować lub zintegrować z warstwami CNN/MLP.

Udowodniona skuteczność w analizie sekwencji bio-sygnałów (EMG, EEG, itp.)

Plan działania (skrót)
Wczytanie i preprocessing danych

Normalizacja kanałów EMG

Etykietowanie klas chwytów

Budowa modelu Transformer

Wejście: sekwencja 200×8

Embedding czasowy + warstwy attention

Warstwa klasyfikacyjna (softmax)

Trenowanie i ewaluacja

Loss: CrossEntropy

Metryki: accuracy, confusion matrix

Walidacja i interpretacja wyników

Attention heatmaps

Eksperymenty z augmentacją