# Rapport LaTeX (UE Chef d'oeuvre - Livrable 2)

## Structure
- `main.tex`: point d'entree du rapport
- `sections/`: chapitres
- `refs.bib`: bibliographie BibLaTeX
- `figures/`: images/captures

## Compilation (recommande)
Depuis le dossier `report/`:

```bash
latexmk -pdf -interaction=nonstopmode -file-line-error main.tex
```

Avec bibliographie:

```bash
latexmk -pdf -interaction=nonstopmode -file-line-error main.tex
biber main
latexmk -pdf -interaction=nonstopmode -file-line-error main.tex
```

## Alternative sans latexmk
```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

## Overleaf
1. Compresser le dossier `report/` en `.zip`.
2. Sur Overleaf: `New Project` -> `Upload Project`.
3. Verifier que le document principal est `main.tex`.
4. Compiler.

## Personnalisation minimale avant rendu
- page de garde (`sections/00_frontmatter.tex`)
- noms et roles des membres du groupe
- nom de l'encadrant
- captures de l'application
- details de gestion de projet (organisation, suivi, contributions)

