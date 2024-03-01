all: user_manual.pdf

user_manual.pdf: *.md
	python3 merge_manuals.py -o user_manual.md
	pandoc user_manual.md -o user_manual.pdf -V colorlinks=true -V linkcolor=blue -V urlcolor=blue -V toccolor=gray -fmarkdown-implicit_figures --highlight-style=tango -V geometry:margin=0.5in
	rm -f user_manual.md

clean:
	rm -f user_manual.pdf user_manual.md

