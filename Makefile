doc:
	rm -f doc/gradients.pdf
	pdflatex -output-directory doc doc/gradients.tex
	rm -f doc/gradients.aux
	rm -f doc/gradients.log

mle:
	rm -f examples/MLE/estimates.tsv
	rm -f examples/MLE/performance.csv
	julia examples/MLE/fit.jl > examples/MLE/estimates.tsv
	Rscript examples/MLE/analyze_estimates.R

cross_validation:
	rm -rf examples/cross_validation/graphs
	rm -f examples/cross_validation/replicates.tsv
	rm -f examples/cross_validation/performance.csv
	julia examples/cross_validation/generate_replicates.jl examples/cross_validation/replicates.tsv
	mkdir examples/cross_validation/graphs
	Rscript examples/cross_validation/analyze_replicates.R

bootstrap:
	rm -rf examples/bootstrap/graphs
	rm -f examples/bootstrap/replicates.tsv
	julia examples/bootstrap/generate_replicates.jl examples/bootstrap/replicates.tsv
	mkdir examples/bootstrap/graphs
	Rscript examples/bootstrap/analyze_replicates.R

profile_likelihood:
	rm -rf examples/profile_likelihood/graphs
	rm -f examples/profile_likelihood/grid.tsv
	julia examples/profile_likelihood/generate_grid.jl > examples/profile_likelihood/grid.tsv
	mkdir examples/profile_likelihood/graphs
	Rscript examples/profile_likelihood/analyze_grid.R

clean:
	rm -f doc/derivations.pdf
	rm -f examples/mle/estimates.tsv
	rm -f examples/mle/performance.csv
	rm -f examples/cross_validation/replicates.tsv
	rm -f examples/cross_validation/performance.csv
	rm -f examples/bootstrap/replicates.tsv
	rm -rf examples/cross_validation/graphs
	rm -rf examples/bootstrap/graphs
	rm -rf examples/profile_likelihood/graphs
	rm -f examples/profile_likelihood/grid.tsv

all: doc mle cross_validation bootstrap profile_likelihood
