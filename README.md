# SAYCOU Semi-Analytical Yield Calculator fOr Uniform samples

Python code for estimating neutron capture yield including direct capture, one-scatter contribution, two-scatter contribution, and an approximate higher-order tail.

# website

https://ipnp.cz/valenta/index_en.html#page=yield_en.html

## Scripts and Files

- `SAYCOU_fastY2.py` : compute script
- `SAYCOU_plot.py` : plotting script
- `Mn55_xs_ENDFBVIII0_myCustomGrid.csv` : example input

## Usage

```bash
python3 SAYCOU_fastY2.py XS_FILE N_AREAL THICKNESS_MM SAMPLE_DIAMETER_MM BEAM_DIAMETER_MM OUTPUT_FILE
python3 plot_yields.py OUTPUT_FILE N_AREAL
