all: user_manual.pdf

user_manual.pdf: *.md
	pandoc index.md install.md cabrnet.md applications.md model.md data.md training.md visualize.md mnist.md -o user_manual.pdf -V colorlinks=true -V linkcolor=blue -V urlcolor=blue -V toccolor=gray -fmarkdown-implicit_figures --highlight-style=tango

