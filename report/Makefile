### VARS Output directory
OUT_DIR=texout
IN_DIR=input/content

# Name - the latex main document
TEX_MAIN=main

# PDF-file name
PDF_TITLE=trinomial-option

.PHONY: main.pdf all clean

all: main.pdf

main.pdf:	$(TEX_MAIN).tex $(IN_DIR)/*
		latexmk -pdf \
			-jobname=$(PDF_TITLE) \
			-outdir=$(OUT_DIR) \
			$(TEX_MAIN).tex

clean:
	rm $(OUT_DIR)/*
