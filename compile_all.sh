#!/bin/bash
pdflatex dissertation.tex
bibtex dissertation
makeglossaries dissertation
pdflatex dissertation.tex
