# Principles for Turning Readings Into Slides

These principles are the default slide workflow for `PLSC 40601`.
They are intended to keep decks teachable, visually consistent, and easy to revise across quarters.

## 1) Start from the reading's structure

- identify the core section headings first
- convert those headings into slide-sized chunks
- do not begin by copying text from the paper

## 2) Build a spine of core slides

- aim for 6 to 10 core slides per major reading block
- keep one major claim, equation, or definition per slide
- use a logical chain: definition, implication, example, bridge

## 3) Use one running example when possible

- reuse the same table, dataset, or empirical example across multiple ideas
- let the example absorb notation so later slides can move faster
- change the imputation rule, estimator, or assumption before introducing a new example

## 4) Convert math into a teaching sequence

- begin with observed quantities
- introduce notation once
- then move to assumptions, identification, and estimation

## 5) Pair each assumption with a visual or calculation

- if a claim can be shown numerically, show it numerically
- if an assumption changes identification status, make that change explicit
- avoid leaving the audience with only symbols

## 6) Separate definition slides from implication slides

- definition slides should be formal and spare
- implication slides should answer: what does this buy us?

## 7) Make identification status explicit

- say whether a target is set identified or point identified
- repeat the same identified quantity across related slides
- do not force students to infer what changed

## 8) Use short transitions

- bridge topics with one short slide or one short frame title shift
- explain why the next tool or assumption matters

## 9) Keep notation consistent

- use `\\E[...]`, `\\Pr[...]`, and `\\Supp[...]` throughout
- reuse notation across slides instead of renaming objects casually
- check vector and matrix notation for consistency with `assets/pres-template_MOW.sty`
- use sentence case for frame titles

## 10) Put overflow material in notes

- alternate derivations, side remarks, and optional examples belong in notes
- do not overload the visible slide with text

## 11) Quote sparingly

- use quotations only for a framing line worth discussing directly
- otherwise paraphrase into the course's own language

## 12) Keep the workflow repeatable

- sketch the deck sequence before editing the `.Rnw`
- make small edits and compile often
- set and report a random seed for any randomized code

## 13) Use course deck conventions

- keep `%-------------------------------------------------------------------------------%` separators between frames and major blocks
- keep the standard references block at the end of each deck
- prefer deck-internal visuals and reproducible figures over pasted screenshots
