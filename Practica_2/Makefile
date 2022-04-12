NUMERO_PR = 2
MEMORIA_PDF = memoria/dissertation_main.pdf

run: 
	python practica2.py

mem: memoria
	make -C memoria latex
	xdg-open memoria/dissertation_main.pdf

clean: memoria
	make -C memoria clean

entrega: memoria/dissertation_main.pdf memoria/practica.py
	zip -r Practica_${NUMERO_PR}.zip memoria/dissertation_main.pdf practica2.py	