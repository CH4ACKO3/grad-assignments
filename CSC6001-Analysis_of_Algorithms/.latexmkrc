# Run latexmk from this course directory; generated files stay under tex/.
$pdf_mode = 1;
$out_dir = 'tex';
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -halt-on-error %O %S';
