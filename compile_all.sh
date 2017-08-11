#!/bin/bash
pdflatex dissertation.tex
bibtex pubs
bibtex dissertation
makeglossaries dissertation
pdflatex dissertation.tex
