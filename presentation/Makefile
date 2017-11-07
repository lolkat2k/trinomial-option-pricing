### VARS# Output directory
OUT_DIR=texout
IN_DIR=input/content

# Name - the latex main document
TEX_MAIN=main

# PDF-file name
PDF_TITLE=trinomial-option-pricing

# You want latexmk to *always* run, because make does not have all the info.
# Also, include non-file targets in .PHONY so they are run regardless of any
# file of the given name existing.

.PHONY: main.pdf all clean spellcheck cleanin cleanout cleanall

# The first rule in a Makefile is the one executed by default ("make"). It
# should always be the "all" rule, so that "make" and "make all" are identical.
all: main.pdf

# MAIN LATEXMK RULE

main.pdf:	$(TEX_MAIN).tex $(IN_DIR)/*
			latexmk -pdf -jobname=$(PDF_TITLE) -outdir=$(OUT_DIR) $(TEX_MAIN).tex

spellcheck: $(IN_DIR)/*
			find $(IN_DIR) -maxdepth 1 -type f -name "*.tex" -exec \
				aspell \
					--conf="./aspell/aspell.conf" \
					--extra-dicts="./aspell/soma.pws" \
					check {} \;

cleanin:
			find $(IN_DIR) -maxdepth 1 -type f -name "*.bak" -delete

cleanout:
			rm $(OUT_DIR)/*

cleanall:	cleanin cleanout
