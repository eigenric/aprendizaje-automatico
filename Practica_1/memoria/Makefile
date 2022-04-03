latex: dissertation_main.tex chap1/gradient_biblio.bib chap1/gradiente_descendente.tex chap2/regresion_lineal.tex
	pdflatex -shell-escape dissertation_main
	bibtex dissertation_main
	pdflatex -shell-escape dissertation_main

clean:
	rm -r *~ _*  *.aux  *.bcf *.lof *.lot *.xml *.log *.out *.toc *.bbl *.blg *.dvi *-blx.bib *.fls *.fdb_latexmk
  

