# Notes for Fri Sep 13

* [TODO] Sammle Relevante Runs
* [TODO] CBO run 60 vs 60/4 (implementiere sub swarms) und 100 vs 100/4
* [NOTE] Bei 100 statt 50 Partikeln loss/acc spike bei epoche 3 (r 121 vs 122)
* [NOTE] Sub Swarms (100/4) schlecht (r 120 vs 121), und (60/4) schlecht (r 117 vs 118)

* [IDEA] layerwise 0th order estimation + backprop? (wird auch im overview angesprochen)

* [TODO] cold start: kleine schrittweite/self.lr am Anfang bzw permanent auf 0.1

### RL results
* [TODO] fasse RL ergebnisse zusammen (siehe nachricht in slack)
* [NOTE] supposed to be good for model free policy search (laut overview)

* [NOTE] exploration in zwei sinnen ..
* [TODO] globale benchmarksfunktionen um exploration zu testen (implementiere entweder in c++ oder in pytorch; füge für CBO/PSO momentum hinzu in c++)

### Applications of 0th Order:

* [NOTE] wir brauchen wahrscheinlich nicht nur 0th order applications, sondern solche, wo auch exploration wichtiger ist als exploitation
* [META] [overview of applications](https://ieeexplore.ieee.org/ielaam/79/9186128/9186148-aam.pdf)
* [RLPS] [good for (model free?) policy search methods (exploration in parameter statt action space)](http://proceedings.mlr.press/v89/vemula19a/vemula19a.pdf)
* [AUTO] [automl (intro siehe overview)](https://ojs.aaai.org/index.php/AAAI/article/download/5926/5782)
* [READ] ES with threading https://arxiv.org/abs/1703.03864
* [READ] Nathan https://arxiv.org/abs/2006.01759

# Mögliche Konferenzen/Journals

* [TODO] AISTATS 03.10. https://virtual.aistats.org/Conferences/2025
* [TODO]
* Foundations of Computational Mathematics (FOCM)
* Statistics & Computing ? 



# Notes Mon Aug 26

* quadrierte norm detrimental
* resampling detrimental für sga

### Hierarchical

* flat first
* M subswarms
* different init centers?
* permute within subswarm only first, then within whole swarm
* merge after convergence?




# Notes Mon Aug 12

* logsumexp
* determnism
* batch size
* e hoch df + norm 
* gateaux
* paper ?
* comparison?
* pso 14%

# Notizen Meeting
* frag simon RL (diskret, nicht diffbar, exploration notwendig)
* quadriere norm von h
* teste xi slack oder hierarchisch




