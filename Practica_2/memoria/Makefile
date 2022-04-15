latex: dissertation_main.tex chap1/complejidad_h.bib chap1/complejidad_h.tex chap2/modelos_lineales.tex
	pdflatex -shell-escape dissertation_main
	bibtex dissertation_main
	pdflatex -shell-escape dissertation_main

clean:
	rm -r *~ _* *.ilg *.ind *.pyg *.aux  *.bcf *.lof *.idx *.lot *.xml *.log *.out *.toc *.bbl *.blg *.dvi *-blx.bib *.fls *.fdb_latexmk
  

