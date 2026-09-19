# Run latexmk from this course directory; generated files stay under tex/.
$pdf_mode = 5;
$out_dir = 'tex';
$xelatex = 'xelatex -synctex=1 -interaction=nonstopmode -halt-on-error %O %S';
