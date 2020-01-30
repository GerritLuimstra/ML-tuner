# ML-tuner
This is a simple framework I use for my hyper parameter tuning.

It currently supports:
Bayesian optimization for the very slow models (scikit-optimize)
Randomized Selection (scikit-learn)
GridSearch (exhaustive) (scikit-learn)

Note that bayesian optimization mode also supports different auxiliary functions if gaussian processes are not desired.
This framework is not secure; it relies on ```eval``` in some instances!

