# Sound Textures

Generate sound textures using Python, store metadata in PostgreSQL, and analyze with R.

## Setup

### Python (3.11)
```bash
pip install -r requirements.txt
```

### R packages
```r
install.packages(c("DBI", "RPostgres", "ggplot2"))
```

### PostgreSQL
Make sure PostgreSQL is running. Update the password in `db_setup.py`, `db_insert.py`, and `analyze.R` if needed.

## Usage

1. **Generate textures:**
   ```bash
   python generate.py -n 20 -s 42
   ```
   Produces `.wav` files in `output/` and a `manifest.json`.

2. **Set up database:**
   ```bash
   python db_setup.py
   ```

3. **Insert metadata:**
   ```bash
   python db_insert.py
   ```

4. **Analyze and plot:**
   ```bash
   Rscript analyze.R
   ```
   Prints stats to console and saves plots to `plots/`.

## Parameters per texture

| Parameter       | Range           |
|----------------|-----------------|
| noise_type     | white, pink, brown |
| base_frequency | 50–500 Hz       |
| duration       | 2–8 seconds     |
| amplitude      | 0.3–1.0         |
| filter_cutoff  | 200–8000 Hz     |
| mod_rate       | 0.5–10 Hz       |
| mod_depth      | 0.0–1.0         |
