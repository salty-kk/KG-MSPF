# METABRIC input

Obtain `METABRIC_RNA_Mutation.csv` from the Kaggle dataset [Breast Cancer Gene Expression Profiles (METABRIC)](https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric), which derives from cBioPortal study `brca_metabric`. Save the original CSV in this directory without changing its row order. Downloading may require a provider account.

The evaluated input has 1,904 rows and 693 columns. Excluding non-positive follow-up leaves 1,903 patients with 1,103 deaths. The exact column names and source-file SHA256 are recorded in `../data_sources.json` so that another researcher can identify the evaluated table.

Clinical covariates, expression values, protein-change mutation strings, PAM50 labels and cohort indicators are parsed by `hetero_data.py`. Mutation string `0` means wild type, a protein-change string means mutated and an unavailable value remains missing. Times are in months. Overall-survival events use `death_from_cancer`, not the input field named `overall_survival`.

The original METABRIC cohort publications are Curtis et al., Nature 486, 346–352 (2012), and Pereira et al., Nature Communications 7, 11479 (2016). Use the original providers' data terms and cite the cohort publications when reusing their data.
