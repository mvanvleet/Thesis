#!/bin/bash
pdflatex dissertation.tex
bib dissertation
makeglossaries dissertation
pdflatex dissertation.tex
