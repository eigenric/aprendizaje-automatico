NUMERO_PR = 2
MEMORIA_PDF = memoria/dissertation_main.pdf

run: 
	python practica2.py

mem: memoria
	make -C memoria latex clean

entrega: memoria/dissertation_main.pdf memoria/practica.py
	zip -r Practica_${NUMERO_PR}.zip memoria/dissertation_main.pdf practica2.py	