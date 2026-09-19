# Run latexmk from the document directory.
$pdf_mode = 1;
$out_dir = 'tex';
$pdflatex = 'pdflatex -synctex=1 -interaction=nonstopmode -halt-on-error %O %S';
