# Compiling the manuscript and the Supporting Information

Everything needed is in this directory. Run:

```bash
cd /home/biazzin/git/AIedes/paper
./build.sh          # both documents
./build.sh main     # manuscript only
./build.sh SI       # Supporting Information only
```

The script uses the local TeX Live 2024 installation
(`/usr/local/texlive/2024/bin/x86_64-linux`), which provides `xelatex`,
`bibtex` and `biber` 2.20. Override the location with `TEXBIN=/path/to/bin
./build.sh` if TeX Live moves.

Bibliographies: `main.tex` uses `bibtex` with `elsarticle-num`, so the sequence
is xelatex, bibtex, xelatex, xelatex. `SI.tex` uses `biblatex`, so the sequence
is xelatex, biber, xelatex, xelatex. The script runs both and then reports the
page count and the number of undefined-reference warnings per document.

The outputs are written to:

```text
paper/main.pdf
paper/SI.pdf
```

Expected after a clean run: `main.pdf` 26 pages, `SI.pdf` 8 pages, zero
undefined references.

## Alternative: Tectonic

A Tectonic binary is also available and needs no TeX Live:

```bash
export LD_LIBRARY_PATH=/home/biazzin/conda-envs/aiedes-env/lib
/home/biazzin/miniconda3/pkgs/tectonic-0.17.0-ha39f199_0/bin/tectonic --keep-logs main.tex
```

It compiles `main.tex`, but `SI.tex` fails unless a Biber matching Tectonic's
cached BibLaTeX (2.17) is on `PATH`; the TeX Live Biber 2.20 is not used by
that path. `./build.sh` is therefore the supported route for the SI.

Useful check:

```bash
stat -c '%y %s %n' main.pdf SI.pdf
```
