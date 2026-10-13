# Self-contained build: pdfLaTeX + biber, output in output/ (no dependency on a global latexmkrc)
@default_files = ("QC_TN_Contraction.tex");   # only the main file (not preamble.tex, sections/)
$pdf_mode = 1;
$bibtex_use = 2;
$out_dir = 'output';
$aux_dir = 'output';
